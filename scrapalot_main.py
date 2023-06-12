#!/usr/bin/env python3
import os
from time import monotonic

import torch
from dotenv import load_dotenv
from langchain import HuggingFacePipeline, HuggingFaceHub, LLMChain, PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import LlamaCpp, GPT4All, OpenAI
from torch import cuda as torch_cuda
from transformers import LlamaForCausalLM
from transformers import LlamaTokenizer
from transformers import pipeline

from scripts import app_logs
from scripts.app_environment import model_type, openai_api_key, model_n_ctx, model_temperature, model_top_p, model_n_batch, model_use_mlock, model_n_threads, model_verbose, \
    huggingface_hub_key, args, db_get_only_relevant_docs, gpt4all_backend, model_path_or_id, gpu_is_enabled
from scripts.app_qa_builder import print_document_chunk, print_hyperlink, process_database_question, process_query
from scripts.app_user_prompt import prompt

# Ensure TOKENIZERS_PARALLELISM is set before importing any HuggingFace module.
os.environ["TOKENIZERS_PARALLELISM"] = "true"
# load environment variables

try:
    load_dotenv()
except Exception as e:
    print("Error loading .env file, create one from example.env:", str(e))


def get_gpu_memory() -> int:
    """
    Returns the amount of free memory in MB for each GPU.
    """
    return int(torch_cuda.mem_get_info()[0] / (1024 ** 2))


# noinspection PyPep8Naming
def calculate_layer_count() -> None | int | float:
    """
    Calculates the number of layers that can be used on the GPU.
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


def get_llm_instance():
    print(f"Initializing model...")

    callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]

    if model_type == "gpt4all":
        if gpu_is_enabled:
            print("GPU is enabled, but GPT4All does not support GPU acceleration. Please use LlamaCpp instead.")
            exit(1)
        return GPT4All(
            model=model_path_or_id,
            n_ctx=model_n_ctx,
            backend=gpt4all_backend,
            callbacks=callbacks,
            use_mlock=model_use_mlock,
            n_threads=model_n_threads,
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
            n_threads=model_n_threads,
            verbose=model_verbose,
            n_gpu_layers=calculate_layer_count(),
            callbacks=callbacks,
        )
    elif model_type == "huggingface-local":
        return HuggingFacePipeline(pipeline=pipeline(
            "text-generation",
            model=LlamaForCausalLM.from_pretrained(
                model_path_or_id,
                load_in_8bit=gpu_is_enabled if gpu_is_enabled else False,
                device_map={
                    '': 'cuda' if gpu_is_enabled else 'cpu',
                    'transformer': 'cuda' if gpu_is_enabled else 'cpu',
                    'lm_head': 'cuda' if gpu_is_enabled else 'cpu',
                },  # if GPU: device_map='auto',
                torch_dtype=torch.float16 if gpu_is_enabled else torch.float32,
                low_cpu_mem_usage=True
            ),
            tokenizer=LlamaTokenizer.from_pretrained(model_path_or_id),
            max_length=2048,
            temperature=model_temperature,
            top_p=model_top_p,
            repetition_penalty=1.15
        ))
    elif model_type == "huggingface-hub":
        return LLMChain(
            prompt=PromptTemplate(template="""<|prompter|>{question}<|endoftext|><|assistant|>""", input_variables=["question"]),
            llm=HuggingFaceHub(
                repo_id=model_path_or_id,
                huggingfacehub_api_token=huggingface_hub_key
            ),
        )
    #     return HuggingFaceHub(
    #         repo_id=model_path_or_id,
    #         task="summarization",
    #         huggingfacehub_api_token=huggingface_hub_key,
    #         model_kwargs={"temperature": model_temperature, "max_length": 1000})
    elif model_type == "openai":
        assert openai_api_key is not None, "Set ENV OPENAI_API_KEY, Get one here: https://platform.openai.com/account/api-keys"
        return OpenAI(openai_api_key=openai_api_key, callbacks=callbacks)
    else:
        print(f"Model {model_type} not supported!")
        exit()


def main():
    llm = get_llm_instance()
    if llm is None:
        print("Could not initialize LLM instance.")
        return

    selected_directory_list = prompt()

    # Initialize a chat history list
    chat_history = []

    while True:
        query = input("\nEnter question (q for quit): ")
        if query.strip() == "":
            continue
        if query == "q":
            break

        qa_list = []
        for i in range(len(selected_directory_list)):
            qa_list.append(process_database_question(selected_directory_list[i], llm, selected_directory_list[i]))

        # Doesn't work very well for some reason won't send proper collection name to process_database_question?
        # def worker(j):
        #     return process_database_question(selected_directory_list[j], llm, selected_directory_list[j])
        #
        # with ThreadPoolExecutor() as executor:
        #     qa_list = list(executor.map(worker, range(len(selected_directory_list))))

        for i in range(len(qa_list)):
            start_time = monotonic()
            qa = qa_list[i]

            print(f"\n\033[94mSeeking for answer from: [{selected_directory_list[i]}]. May take some minutes...\033[0m")
            answer, docs = process_query(qa, query, chat_history, db_get_only_relevant_docs)
            print(f"\033[94mTook {round(((monotonic() - start_time) / 60), 2)} min to process the answer!\n\033[0m")

            if docs is not None:
                for doc in docs:
                    print_hyperlink(doc)
                    print_document_chunk(doc)


if __name__ == "__main__":
    if args.log_level is not None:
        app_logs.initialize_logging()

    main()
