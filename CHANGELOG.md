# CHANGELOG

- **09.06.2023**
    - adding collections to the DB, means that, not only you can have separate databases, but you can have sub-collections inside them. An example: a database named "medicine" can have collections: of
      allergy, immunology, anesthesiology, dermatology, and radiology .... which you can choose from the UI when asking questions. When you perform and ingest you can specify --ingest-dbname and
      --collection, if you don't specify --collection it will be named as the database name. If you don't specify any of these arguments user will be prompted to enter the database (for now only
      supported in the terminal.
    - GPU-enabled flags to turn it on or off
    - added some images to the README.md on how the app works
    - added separate requirements_linux.txt, and requirements_windows.txt with package differences for each OS,
    - fixes in ingest file to skip unparsable files and removed default countdown in the terminal when the prompt is waiting for user input
    - more descriptive messages in the command line
    - translation of answers, and source chunks from books are disabled by default because they contact Google Translate over the network, and not all languages are supported for now, the same states
      for text-to-speech (for now only English)
    - added embeddings normalization settings, fixed embeddings_kwargs to support CPU
    - looks like pandoc is also needed to be installed separately from requirements.txt when parsing epub files (added to README.md)
    - added CHANGELOG file to track the changes
    - CUDA testing on Linux
    - added support for text-to-speach espeak library on linux
    - ingest of files can now be performed in batch, meaning you can separate by comma indexes of source directories you want to ingest (only available in terminal)

- **10.06.2023**
    - CUDA GPU now works on windows, updated instructions inside README.md
    - CUDA GPU linux testing