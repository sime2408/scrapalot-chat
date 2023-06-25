#!/usr/bin/env python3
import logging
import os
from time import monotonic

import torch
from auto_gptq import AutoGPTQForCausalLM
from dotenv import load_dotenv
from langchain import HuggingFacePipeline
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import LlamaCpp, GPT4All, OpenAI
from langchain.schema import Document
from torch import cuda as torch_cuda
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, LlamaForCausalLM, GenerationConfig, pipeline

from scripts import app_logs
from scripts.app_environment import model_type, openai_api_key, model_n_ctx, model_temperature, model_top_p, model_n_batch, model_use_mlock, model_verbose, \
    args, db_get_only_relevant_docs, gpt4all_backend, model_path_or_id, gpu_is_enabled, cpu_model_n_threads, gpu_model_n_threads, model_n_answer_words, huggingface_model_base_name
from scripts.app_qa_builder import print_document_chunk, print_hyperlink, process_database_question, process_query
from scripts.app_user_prompt import prompt

# Ensure TOKENIZERS_PARALLELISM is set before importing any HuggingFace module.
os.environ["TOKENIZERS_PARALLELISM"] = "true"
# load environment variables

try:
    load_dotenv()
except Exception as e:
    logging.error("Error loading .env file, create one from example.env:", str(e))


def get_gpu_memory() -> int:
    """
    Returns the amount of free memory in MB for each GPU.
    """
    return int(torch_cuda.mem_get_info()[0] / (1024 ** 2))


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


def main():
    llm = get_llm_instance(StreamingStdOutCallbackHandler())

    if llm is None:
        logging.error("Could not initialize LLM instance.")
        return

    logging.info(f"Running on: {'cuda' if gpu_is_enabled else 'cpu'}")
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
        for dir_name in selected_directory_list:
            # Check if the directory name contains a slash, indicating a sub-collection
            if "/" in dir_name:
                # If so, split the string to separate the database name and the collection name
                database_name, collection_name = dir_name.split("/")
            else:
                # If not, the database name and the collection name are the same
                database_name, collection_name = dir_name, dir_name

            processed_answer = await process_database_question(database_name=database_name, llm=llm, collection_name=collection_name)
            qa_list.append(processed_answer)

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
            answer, docs = process_query(qa, query, model_n_answer_words, chat_history, db_get_only_relevant_docs, translate_answer=True)
            print(f"\033[94mTook {round(((monotonic() - start_time) / 60), 2)} min to process the answer!\n\033[0m")

            if isinstance(docs, Document):
                doc = docs
                print_hyperlink(doc)
                print_document_chunk(doc)
            else:
                for doc in docs:
                    print_hyperlink(doc)
                    print_document_chunk(doc)


if __name__ == "__main__":
    app_logs.initialize_logging()

    main()
