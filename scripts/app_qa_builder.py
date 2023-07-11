import os
import textwrap
from typing import Optional
from urllib.request import pathname2url
import logging

from deep_translator import GoogleTranslator
from langchain import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.retrieval_qa.base import BaseRetrievalQA
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.callbacks.base import BaseCallbackHandler
from langchain import HuggingFacePipeline

from langchain.llms import LlamaCpp, GPT4All, OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, LlamaForCausalLM, GenerationConfig, pipeline
from openai.error import AuthenticationError

from .app_environment import translate_dst, translate_src, translate_docs, translate_q, ingest_target_source_chunks, args, openai_use, ingest_embeddings_model, gpu_is_enabled, \
    chromaDB_manager

from .app_environment import model_type, openai_api_key, model_n_ctx, model_temperature, model_top_p, model_n_batch, model_use_mlock, model_verbose, \
    db_get_only_relevant_docs, gpt4all_backend, model_path_or_id, cpu_model_n_threads, gpu_model_n_threads, model_n_answer_words, huggingface_model_base_name


# Ensure TOKENIZERS_PARALLELISM is set before importing any HuggingFace module.
os.environ["TOKENIZERS_PARALLELISM"] = "true"
# load environment variables

try:
    load_dotenv()
except Exception as e:
    logging.error("Error loading .env file, create one from example.env:", str(e))


# noinspection PyPep8Naming
def calculate_layer_count() -> None | int | float:
    """
    How many layers of a neural network model you can fit into the GPU memory,
    rather than determining the number of threads.
    The layer size is specified as a constant (120.6 MB), and the available GPU memory is divided by this to determine the maximum number of layers that can be fit onto the GPU.
    Some additional memory (the size of 6 layers) is reserved for other uses.
    The maximum layer count is capped at 32.
    """
    if not gpu_is_enabled:
        return None
    LAYER_SIZE_MB = 120.6  # This is the size of a single layer on VRAM, and is an approximation.
    # The current set value is for 7B models. For other models, this value should be changed.
    LAYERS_TO_REDUCE = 6  # About 700 MB is needed for the LLM to run, so we reduce the layer count by 6 to be safe.
    if (get_gpu_memory() // LAYER_SIZE_MB) - LAYERS_TO_REDUCE > 32:
        return 32
    else:
        return get_gpu_memory() // LAYER_SIZE_MB - LAYERS_TO_REDUCE


def get_llm_instance(*callback_handler: BaseCallbackHandler):
    logging.debug(f"Initializing model...")

    callbacks = [] if args.mute_stream else callback_handler

    if model_type == "gpt4all":
        if gpu_is_enabled:
            logging.warn("GPU is enabled, but GPT4All does not support GPU acceleration. Please use LlamaCpp instead.")
            exit(1)
        return GPT4All(
            model=model_path_or_id,
            n_ctx=model_n_ctx,
            backend=gpt4all_backend,
            callbacks=callbacks,
            use_mlock=model_use_mlock,
            n_threads=gpu_model_n_threads if gpu_is_enabled else cpu_model_n_threads,
            n_predict=1000,
            n_batch=model_n_batch,
            top_p=model_top_p,
            temp=model_temperature,
            streaming=False,
            verbose=False
        )
    elif model_type == "llamacpp":
        return LlamaCpp(
            model_path=model_path_or_id,
            temperature=model_temperature,
            n_ctx=model_n_ctx,
            top_p=model_top_p,
            n_batch=model_n_batch,
            use_mlock=model_use_mlock,
            n_threads=gpu_model_n_threads if gpu_is_enabled else cpu_model_n_threads,
            verbose=model_verbose,
            n_gpu_layers=calculate_layer_count() if gpu_is_enabled else None,
            callbacks=callbacks,
        )
    elif model_type == "huggingface":
        if gpu_is_enabled and huggingface_model_base_name is not None:
            logging.info("Tokenizer loaded")
            tokenizer = AutoTokenizer.from_pretrained(model_path_or_id, use_fast=True)
            model = AutoGPTQForCausalLM.from_quantized(
                model_name_or_path=model_path_or_id,
                model_basename=huggingface_model_base_name if ".safetensors" not in huggingface_model_base_name else huggingface_model_base_name.replace(".safetensors", ""),
                use_safetensors=True,
                trust_remote_code=True,
                device="cuda:0",
                use_triton=False,
                quantize_config=None,
            )
        elif gpu_is_enabled:
            logging.info("Using AutoModelForCausalLM for full models")
            tokenizer = AutoTokenizer.from_pretrained(model_path_or_id)
            model = AutoModelForCausalLM.from_pretrained(
                model_path_or_id,
                device_map="auto",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                # max_memory={0: "15GB"} # Uncomment this line if you encounter CUDA out of memory errors
            )
            model.tie_weights()
        else:
            logging.info("Using LlamaTokenizer")
            tokenizer = LlamaTokenizer.from_pretrained(model_path_or_id)
            model = LlamaForCausalLM.from_pretrained(model_path_or_id)

        return HuggingFacePipeline(pipeline=pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=2048,
            temperature=0,
            top_p=model_top_p,
            repetition_penalty=1.15,
            generation_config=GenerationConfig.from_pretrained(model_path_or_id),
        ))
    elif model_type == "openai":
        assert openai_api_key is not None, "Set ENV OPENAI_API_KEY, Get one here: https://platform.openai.com/account/api-keys"
        return OpenAI(openai_api_key=openai_api_key, callbacks=callbacks)
    else:
        logging.error(f"Model {model_type} not supported!")
        raise Exception(f"Model type {model_type} is not supported. Please choose one of the following: LlamaCpp, GPT4All")


async def process_database_question(database_name, llm, collection_name: Optional[str]):
    embeddings_kwargs = {'device': 'cuda'} if gpu_is_enabled else {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = OpenAIEmbeddings() if openai_use else HuggingFaceEmbeddings(
        model_name=ingest_embeddings_model, model_kwargs=embeddings_kwargs, encode_kwargs=encode_kwargs
    )
    persist_dir = f"./db/{database_name}"

    db = Chroma(persist_directory=persist_dir,
                embedding_function=embeddings,
                collection_name=collection_name if collection_name else args.collection,
                client_settings=chromaDB_manager.get_chroma_setting(persist_dir)
                )

    retriever = db.as_retriever(search_kwargs={"k": ingest_target_source_chunks if ingest_target_source_chunks else args.ingest_target_source_chunks})

    template = """You are a an AI assistant providing helpful advice. You are given the following extracted parts of a long document and a question.
    Provide a conversational answer (about {answer_length} words) based on the context provided.
    If you can't find the answer in the context below, just say
    "Hmm, I'm not sure." Don't try to make up an answer. If the question is not related to the context, politely respond
    that you are tuned to only answer questions that are related to the context.

    Question: {question}
    =========
    {context}
    =========
    Answer:"""
    question_prompt = PromptTemplate(template=template, input_variables=["question", "answer_length", "context"])

    qa = ConversationalRetrievalChain.from_llm(llm=llm, condense_question_prompt=question_prompt, retriever=retriever, chain_type="stuff", return_source_documents=not args.hide_source)
    return qa


def process_query(qa: BaseRetrievalQA, query: str, answer_length: int, chat_history, chromadb_get_only_relevant_docs: bool, translate_answer: bool):
    try:

        if chromadb_get_only_relevant_docs:
            docs = qa.retriever.get_relevant_documents(query)
            return None, docs

        if translate_q:
            query_en = GoogleTranslator(source=translate_dst, target=translate_src).translate(query)
            res = qa({"question": query_en, "answer_length": answer_length, "chat_history": chat_history})
        else:
            res = qa({"question": query, "answer_length": answer_length, "chat_history": chat_history})

        # Print the question
        print(f"\nQuestion: {query}\n")

        answer, docs = res['answer'], res['source_documents']
        # Translate answer if necessary
        if translate_answer:
            answer = GoogleTranslator(source=translate_src, target=translate_dst).translate(answer)

        print(f"\n\033[1m\033[97mAnswer: \"{answer}\"\033[0m\n")

        return answer, docs
    except AuthenticationError as e:
        print(f"Warning: Looks like your OPENAI_API_KEY is invalid: {e.error}")
        return None, []
