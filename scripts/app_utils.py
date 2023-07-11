import logging
import math
import os
import platform
import sys
import textwrap
from typing import List

from langchain.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    PyMuPDFLoader,
    TextLoader,
    JSONLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredEmailLoader,
)
from langchain.schema import Document


def print_platform_version():
    """
    The sys.platform for macOS is 'darwin', for Windows it's 'win32', and for Linux it's 'linux'
    (it can be more specific like 'linux2' or 'linux3', depending on the Linux version you're running).
    The platform.machine() returns the machine type, like 'x86_64' or 'amd64' for an Intel x64 machine, and 'arm64' for an ARM64 machine.
    """
    logging.debug("sys_platform:", sys.platform)
    logging.debug("platform_machine:", platform.machine())


######################################################################
# INGEST
######################################################################

# Custom document loaders
class MyElmLoader(UnstructuredEmailLoader):
    """Wrapper to fall back to text/plain when default does not work"""

    def load(self) -> List[Document]:
        """Wrapper adding fallback for elm without html"""
        try:
            try:
                doc = UnstructuredEmailLoader.load(self)
            except ValueError as e:
                if 'text/html content not found in email' in str(e):
                    # Try plain text
                    self.unstructured_kwargs["content_source"] = "text/plain"
                    doc = UnstructuredEmailLoader.load(self)
                else:
                    raise
        except Exception as e:
            # Add file_path to exception message
            raise type(e)(f"{self.file_path}: {e}") from e

        return doc


# Map file extensions to document loaders and their arguments
LOADER_MAPPING = {
    ".csv": (CSVLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    ".eml": (MyElmLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    ".pdf": (PyMuPDFLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8", "autodetect_encoding": True}),
    ".json": (JSONLoader, {"jq_schema": ".[].full_text"}),
    ".py": (TextLoader, {}),
    ".java": (TextLoader, {}),
    ".js": (TextLoader, {}),
    ".html": (TextLoader, {}),
    # Add more mappings for other file extensions and loaders as needed
}


def load_single_document(file_path: str) -> List[Document]:
    """
    The function takes a single file and loads its data using the appropriate loader based on its extension.
    :param file_path: The path of the file to load.
    :return: A list of Document objects loaded from the file.
    """
    ext = (os.path.splitext(file_path)[-1]).lower()
    if ext in LOADER_MAPPING:
        try:
            loader_class, loader_args = LOADER_MAPPING[ext]
            loader = loader_class(file_path, **loader_args)
            return loader.load()
        except Exception as e:
            raise ValueError(f"Problem with document {file_path}: \n'{e}'")
    raise ValueError(f"Unsupported file extension '{ext}'")


######################################################################
# DISPLAY
######################################################################

def display_directories():
    """
    This function displays the list of existing directories in the parent directory.
    It also explores one level of subdirectories for each directory.
    :return: The list of existing directories.
    """
    base_dir = "./source_documents"
    directories = []

    # Fetch directories and their direct subdirectories
    sorted_list = sorted(os.listdir(base_dir))
    for dir_name in sorted_list:
        if not dir_name.startswith("."):
            dir_path = os.path.join(base_dir, dir_name)

            if os.path.isdir(dir_path):
                directories.append(dir_name)
                subdirectories = [f"{dir_name}/{sub_dir}" for sub_dir in sorted(os.listdir(dir_path)) if os.path.isdir(os.path.join(dir_path, sub_dir))]
                directories.extend(subdirectories)

    cli_column_number = 4  # Number of columns to be displayed
    cli_column_width = 30  # Width of the column

    # Calculate the number of rows needed based on the number of directories
    num_rows = math.ceil(len(directories) / cli_column_number)

    # Print directories in multiple columns
    for row in range(num_rows):
        for column in range(cli_column_number):
            # Calculate the index of the directory based on the current row and column
            index = row + column * num_rows

            if index < len(directories):
                directory = directories[index]
                wrapped_directory = textwrap.shorten(directory, width=cli_column_width - 1, placeholder="...")
                print(f"{index + 1:2d}. {wrapped_directory:{cli_column_width}}", end=" ")
        print()  # Print a new line after each row

    return directories
