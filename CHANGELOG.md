# CHANGELOG

- **19.06.2023**
    - Collections now work by default
    - Increased INGEST_TARGET_SOURCE_CHUNKS from 4 to 6 for better LLM answers
    - extended answer length to around 200 words, controlled by MODEL_ANSWER_N_WORDS
    - added support to ingest code files, like .py, .java, .js, .html
    - added support to display collection subdirectory in the CLI script
    - get_llm_instance now accepts callback handlers as subclass of BaseCallbackHandler, this means that you can stream answer to other handlers
    - fixed is cuda available function

- **15.06.2023**
    - Upgrade of langchain, gpt4all
    - the number of threads for the model now calculated using multiprocessing, by default model exhausts 80% of CPU
    - GPU is determined from is_cuda_available() function, no need for .env property
    - logging turned on by default and generates a file under `logs` directory in the root folder structure, improved logging for errors and warn
    - more granulated code in scrapalot_ingest.py, separated main method into more reusable ones
    - started working on collections of sub-databases (WIP)
    - added funding.yml, .editorconfig
    - added `CONTRIBUTING.md`, updated `README.md`
    - merged requirements-*.txt files into single one with checks per system like using ; sys_platform == 'darwin' and platform_machine == 'x86_64'
    - UI changes:
        - streamlit now has default .streamlit folder with defaults and custom theme you can enable
        - translations support for multiple languages (`streamlit-option-menu` library)

- **12.06.2023**
    - Upgrade of langchain, gpt4all, llama-cpp-python
    - Increased MODEL_N_BATCH from 512 to default 1024, better works for LLAMA, 8 works for GPT4All
    - added support to set the speed of text-to-speech via TTS_SPEED env variable, moved tts code to a separate py file
    - speed up ingestion of PDF files by using PyMuPDF instead of pdfminer.six
    - utilization of code functions now goes to app_utils.py
    - improved exception handling
    - tagging collections before storing to the ChromaDB was not necessary, removed that code
    - support for ingestion to a multiple database removed due to race condition multithreading
    - removed images from readme, because of constant changes of features
    - Improved README.md

- **10.06.2023**
    - CUDA GPU now works on windows, updated instructions inside README.md
    - CUDA GPU linux testing

- **09.06.2023**
    - adding collections to the DB means that not only you can have separate databases, but you can have sub-collections inside them. An example: a database named "medicine" can have collections: of
      allergies, immunology, anesthesiology, dermatology, and radiology .... which you can choose from the UI when asking questions. When you perform and ingest you can specify --ingest-dbname and
      --collection, if you don't specify --collection it will be named as the database name. If you don't specify any of these arguments, users will be prompted to enter the database (for now only
      supported in the terminal.
    - GPU-enabled flags to turn it on or off
    - add some images to the README.md on how the app works
    - added separate requirements.txt logic using sys_platform with package differences for each OS,
    - fixes in ingested file to skip unparsable files and removed default countdown in the terminal when the prompt is waiting for user input
    - more descriptive messages in the command line
    - translation of answers, and source chunks from books are disabled by default because they contact Google Translate over the network, and not all languages are supported for now, the same states
      for text-to-speech (for now only English)
    - added embeddings normalization settings, fixed embeddings_kwargs to support CPU
    - looks like pandoc is also needed to be installed separately from requirements.txt when parsing epub files (added to README.md)
    - added CHANGELOG file to track the changes
    - CUDA testing on Linux
    - added support for a text-to-speech espeak library on linux
