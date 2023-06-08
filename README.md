# Scrapalot Chat
Ask questions about your documents without an internet connection, using the power of LLMs. 100% private, no data leaves your execution environment at any point. 
You can ingest documents and ask questions without an internet connection!

# Environment Setup
In order to set your environment up to run the code here, first install all requirements. 

## GPU (Linux):

If you have an Nvidia GPU, you can speed things up by installing the llama-cpp-python version with CUDA by setting these flags:
- On windows: `🚧 WIP`
- On linux: `export LLAMA_CUBLAS=1`

```shell
pip3 install -r requirements_win.txt
```

First, you have to uninstall old torch installation and install CUDA one:
Install a proper torch version:
```shell
pip3 uninstall torch
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

llama.cpp doesn't work on windows with GPU, only with CPU, so you should try with a linux distro
Installing torch with CUDA, will only speed up the vector search, not the writing by llama.cpp.

You should install the latest cuda toolkit:
```shell
conda install -c conda-forge cudatoolkitpip uninstall llama-cpp-python
```

Now, set environment variables:
```shell
set LLAMA_CUBLAS=1
set CMAKE_ARGS=-DLLAMA_CUBLAS=on
set FORCE_CMAKE=1
```

Install llama:
```shell
pip install llama-cpp-python --no-cache-dir --verbose
```

Modify LLM code to accept `n_gpu_layers`:
```shell
llm = LlamaCpp(model_path=model_path, ..., n_gpu_layers=20)
```

Change environment variable model:
```shell
MODEL_TYPE=llamacpp
MODEL_ID_OR_PATH=models/ggml-vic13b-q5_1.bin
```

## GPU (Windows)

You can use the included installer batch file to install the required dependencies for GPU acceleration, or:

1. Install [NVidia CUDA 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive)
2. Install `llama-cpp-python` package with cuBLAS enabled. Run the code below in the directory you want to build the package in.
   - Powershell:

    ```powershell
    $Env:CMAKE_ARGS="-DLLAMA_CUBLAS=on"; $Env:FORCE_CMAKE=1; pip3 install llama-cpp-python --force-reinstall --upgrade --no-cache-dir
    ```

   - Bash:

    ```bash
    CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip3 install llama-cpp-python --force-reinstall --upgrade --no-cache-dir
    ```

3. Enable GPU acceleration in `.env` file by setting `IS_GPU_ENABLED` to `True`
4. Run `scrapalot_ingest.py` and `scrapalot_main.py` as usual


## For CPU only setup:
```shell
pip3 install -r requirements_mac.txt
```

- For conda environment:
```shell
conda create --name scrapalot-chat python=3.10.11 && conda activate scrapalot-chat
```

If you want to remove the conda environment, run this:
```shell
conda remove -n scrapalot-chat --all
```

# LLM Models

Then, download the LLM model and place it in a directory of your choice (for example: `models`):
- `gpt4all`: [ggml-gpt4all-j-v1.3-groovy.bin](https://gpt4all.io/models/ggml-gpt4all-j-v1.3-groovy.bin)
  - If you prefer a different GPT4All-J compatible model, just download it and reference it in your `.env` file.
  - If you prefer a llama model, download [ggml-model-q4_0.bin](https://huggingface.co/Pi3141/alpaca-native-7B-ggml/tree/main)
    - NOTE: you need to adapt `GPT4ALL_BACKEND`
- `llamacpp`: [WizardLM-7B-uncensored.ggmlv3.q8_0.bin](https://huggingface.co/TheBloke/WizardLM-7B-uncensored-GGML/tree/8917029d1fecd37d2c3a395d399868bfd225ff36)
- `llamacpp`: [ggml-vicuna-13b-1.1](https://huggingface.co/vicuna/ggml-vicuna-13b-1.1/tree/main)
- `llamacpp`: [koala-7B.ggmlv3.q8_0.bin](https://huggingface.co/TheBloke/koala-7B-GGML/tree/main)
- `huggingface-local`: [TheBloke/guanaco-7B-HF](https://huggingface.co/TheBloke/guanaco-7B-HF)
- `huggingface-hub`: Not yet implemented!
- `openai`: Uses OpenAI API and gpt-4 model

# Env variables

Rename `example.env` to `.env` and edit the variables appropriately.
```ini
OS_RUNNING_ENVIRONMENT: Operating system your application is running on.

INGEST_PERSIST_DIRECTORY: is the folder you want your vectorstore in
INGEST_SOURCE_DIRECTORY: from where books will be parsed
INGEST_EMBEDDINGS_MODEL: SentenceTransformers embeddings model name (see https://www.sbert.net/docs/pretrained_models.html)
INGEST_CHUNK_SIZE: default chunk size of texts when performing an ingest
INGEST_OVERLAP: default chunk overlap of texts when performing an ingest
INGEST_TARGET_SOURCE_CHUNKS: The amount of chunks (sources) that will be used to answer a question

MODEL_TYPE: supports llamacpp, gpt4all, openai, huggingface-local, huggingface-hub
MODEL_ID_OR_PATH: Path to your gpt4all or llamacpp supported LLM
MODEL_N_CTX: Token context window. Maximum token limit for the LLM model
MODEL_TEMPERATURE: Temperature between 0.0 & 1.0. If 0 it will return exact answers from the books
MODEL_USE_MLOCK: If this value is set to 1, the entire model will be loaded into RAM (avoid using the disk but use more RAM), 
if you have little RAM, set this value to 0
MODEL_VERBOSE: Turn on or off model debugging
MODEL_N_BATCH:  the number of tokens processed at any one time. The lower this value, the less hardware resources will be required, 
but the query may be very slow; a high value, on the other hand, speeds things up at the cost of higher memory usage.
MODEL_N_THREADS: How much threads will be used when model process the data
MODEL_TOP_P: The top-p value to use for sampling.

TRANSLATE_QUESTION: Whether or not turn on translation of questionto english. Based on GoogleTranslate HTTP calls.
TRANSLATE_ANSWER: Whether or not turn on translation of answers from english to your language
TRANSLATE_SRC_LANG: If you want to translate answers from this language
TRANSLATE_DST_LANG: If you want to translate answers to this language

CLI_COLUMN_WIDTH: How wide will be each column when printing subdirectories of database or source documenets
CLI_COLUMN_NUMBER: How many columns by default will be shown in CLI

DB_GET_ONLY_RELEVANT_DOCS: If this is set to `true` only documents will be returned from the database. Program won't go through the process of sending chunks to the LLM.

OPENAI_USE: Whether to use this model or not, if yes, different embeddings should be used

GPU_IS_ENABLED: Whether or not your GPU environment is enabled.

OPENAI_API_KEY: OpenAI key for http calls to OpenAI GPT-4 API
HUGGINGFACEHUB_API_TOKEN: Token to connect to huggingface and download the models
GPT4ALL_BACKEND: backend type of GPT4All model. Can be gptj or llama (ggml-model-q4_0.bin)
```

Note: because of the way `langchain` loads the `SentenceTransformers` embeddings, the first time you run the script it will require internet connection to download the embeddings model itself.

# Instructions for ingesting your own dataset

For each set of documents, create a new `subfolder` in the `source_documents` folder and place the files inside.

The supported extensions are:
- `.csv`: CSV,
- `.docx`: Word Document,
- `.doc`: Word Document,
- `.enex`: EverNote,
- `.eml`: Email,
- `.epub`: EPub,
- `.html`: HTML File,
- `.md`: Markdown,
- `.msg`: Outlook Message,
- `.odt`: Open Document Text,
- `.pdf`: Portable Document Format (PDF),
- `.pptx` : PowerPoint Document,
- `.ppt` : PowerPoint Document,
- `.txt`: Text file (UTF-8),
- `.json`: Text file (jq_schema),

If you use conda environment, and you want to parse `epub` books, you have to install `pypandoc` inside conda environment.

```shell
conda install -c conda-forge pypandoc
```

Run the following command to ingest all the data.
Follow the instructions to select the correct set of source documents. The screen will ask you which `subfolder` you want to ingest.

```shell
python scrapalot_ingest.py
```

Output should look like this:

```shell
Select an option or 'q' to quit:

1. Select existing directory
2. Create a new directory
3. Use current source_directory: ./source_documents/ufo


Enter your choice [1]: 1

Seconds remaining: 10 to choose option [1]. 

Existing directories in ./source_documents:
1. psychology
2. medicine

Enter the number of the existing directory: 1
Selected directory: ./source_documents/psychology
Creating new vectorstore
Loading documents from source_documents
Loading new documents: 100%|██████████████████████| 1/1 [00:01<00:00,  1.73s/it]
Creating embeddings. May take some minutes...
Ingestion complete! You can now run scrapalot_main.py to query your documents
```

It will create a subfolder in the `db` folder containing the local vectorstore. It Will take 20-30 seconds per document (much less if you use an Nvidia GPU), depending on the size of the document.
You can ingest as many documents as you want, and all will be accumulated in the selected embeddings' database.
If you want to start from an empty database, delete the subfolder inside the `db` folder, or create a new one using the scrapalot_ingest.py script.

Note: during the ingest process, no data leaves your local environment. You could ingest without an internet connection, except for the first time you run the ingest script, when the embedding model is downloaded.

## Ask questions to your documents!

### Console CLI
In order to ask a question, run a command like this. Mute stream is the flag to hide generation of answer in the console:
```shell
python scrapalot_main.py --mute-stream
```

When you run this script, you'll have to choose to which database you want to perform conversation.
If you want your question returned from multiple databases, you will have to put the index numbers separated by comma.
For example:

```shell
Existing databases in ./db folder:

1. psychology
2. medicine
 
Enter the index number of the database (or more of them separated by comma): 1,2

Enter question (q for quit): How to be happy?

Seeking for answer from: [psychology]. May take some minutes...
```

The script also supports other optional command-line arguments to modify its behavior. 
You can see a full list of these arguments by running this command in your terminal:
```shell 
python scrapalot_main.py --help
```

### Web app

Scrapalot has REST API built by `fastapi` (`scrapalot_main_api_run.py`), that API has to be running if you want to run the UI (`scrapalot_main_web.py`), which is based on `streamlit` / `streamlit-chat`.

To run the web:

```shell
streamlit run scrapalot_main_web.py
```

# How does it work?
Selecting the right local models and the power of `LangChain` you can run the entire pipeline locally, without any data leaving your environment, and with reasonable performance.

- `scrapalot_ingest.py` uses `LangChain` tools to parse the document and create embeddings locally using `HuggingFaceEmbeddings` (`SentenceTransformers`). It then stores the result in a local vector database using `Chroma` vector store. 
- `scrapalot_main.py` uses a local LLM based on `llamacpp, gpt4all, openai` to understand questions and create answers. The context for the answers is extracted from the local vector store using a similarity search to locate the right piece of context from the docs.

Note: you could turn off your internet connection, and the script inference would still work. No data gets out of your local environment.

### Docker

1. Put your data in `models` / `source_documents` in the project root folder (Can be customized changing the corresponding value in the `docker-compose.yaml`)

2. If you want to do it manually, you can service by service, with docker compose
```
docker-compose up --build scrapalot-chat
```

# System Requirements

## Python libraries

1. [x] langchain: LangChain is a framework for developing applications powered by language models
2. [x] gpt4all: A free-to-use, locally running, privacy-aware chatbot. No GPU or internet is required. 
3. [x] chromadb: A vector database, capable of embedding text
4. [x] llama-cpp-python: Python bindings for CPP. Offers a web server which aims to act as a drop-in replacement for the OpenAI API
5. [x] urllib3: A powerful, sanity-friendly HTTP client for Python.
6. [x] pdfminer.six: A library for extracting text, images, and metadata from PDF files.
7. [x] python-dotenv: Reads key-value pairs from a .env file and adds them to the environment variables.
8. [x] unstructured, extract-msg, tabulate, pandoc, pypandoc, tqdm: Libraries related to handling and manipulating various data formats, tabulating data, and providing progress bars.
9. [x] deep-translator: A flexible free and unlimited library to translate between different languages in a simple way using multiple translators.
10. [x] openai, huggingface, huggingface_hub, sentence_transformers, transformers: Libraries related to machine learning and natural language processing, particularly for working with transformer models like GPT and BERT.
11. [x] bitsandbytes, safetensors: Libraries that seem related to operations with bits, bytes, and tensors, but I can't find more detailed information as of my last update.
12. [x] pyttsx3: A text-to-speech conversion library.
13. [x] fastapi, uvicorn, gunicorn, python-multipart: Libraries for building APIs with Python and deploying them.
14. [x] streamlit, streamlit-chat: Libraries to quickly create custom web apps for machine learning and data science projects.
15. [x] psutil: A cross-platform library for accessing system details and process utilities.

## Python Version
To use this software, you must have minimum Python `3.10` or later installed. Earlier versions of Python will not compile.

## C++ Compiler
If you encounter an error while building a wheel during the `pip install` process, you may need to install a C++ compiler on your computer.

## Windows 10/11
### installation of packages

Install the required packages (on MacOS):
```shell
pip3 install -r requirements_win.txt
```

To install a C++ compiler on Windows 10/11, follow these steps:

1. Install Visual Studio 2022.
2. Make sure the following components are selected:
   * Universal Windows Platform development
   * C++ CMake tools for Windows
3. Download the MinGW installer from the [MinGW website](https://sourceforge.net/projects/mingw/).
4. Run the installer and select the `gcc` component.

## Mac OS 
### installation of packages

Install the required packages (on MacOS):
```shell
pip3 install -r requirements_mac.txt
```

### Intel Chip
When running a Mac with Intel hardware (not M1), you may run into `_clang: error: the clang compiler does not support '-march=native'_ during pip install`.

If so, set your `archflags` during pip install. Eg: `_ARCHFLAGS="-arch x86_64" pip3 install -r requirements_mac.txt_`

# Disclaimer
This is a test project to validate the feasibility of a fully private solution for question answering using LLMs and Vector embeddings. It is not production ready, and it is not meant to be used in production. The model selection is not optimized for performance, but for privacy; but it is possible to use different models and vector stores to improve performance.
