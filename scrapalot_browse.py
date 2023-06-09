import importlib
import math
import os
import textwrap

import pyttsx3
from deep_translator import GoogleTranslator
from langchain.docstore.document import Document
from langchain.document_loaders import (
    UnstructuredEPubLoader,
    UnstructuredWordDocumentLoader,
    PDFMinerLoader,
)

from scripts.user_environment import translate_src, translate_dst, translate_docs, ingest_chunk_size, os_running_environment

if os_running_environment == 'macOs':
    driver = "pyttsx3.drivers.nsss"
    importlib.import_module(driver)
elif os_running_environment == 'windows':
    driver = 'pyttsx3.drivers.sapi5'
    importlib.import_module(driver)
elif os_running_environment == 'linux':
    driver = 'pyttsx3.drivers.espeak'
    importlib.import_module(driver)

engine = pyttsx3.init()
engine.setProperty('rate', 205)
voices = engine.getProperty('voices')  # getting details of current voice


def print_all_voices_helper():
    for index, voice in enumerate(voices):
        print(f"Voice {index}: ID: {voice.id}, Name: {voice.name}, Languages: {voice.languages}, Gender: {voice.gender}, Age: {voice.age}")
        engine.setProperty('voice', voice.id)


LOADER_MAPPING = {
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".pdf": (PDFMinerLoader, {}),
}


def get_directories(directory):
    return sorted([name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))])


def get_files(directory):
    return [name for name in os.listdir(directory) if
            os.path.isfile(os.path.join(directory, name)) and not name.startswith('.')]


def print_in_grid(items, num_columns, column_width, indexed=True):
    num_rows = math.ceil(len(items) / num_columns)
    for i in range(num_rows):
        for j in range(num_columns):
            index = j * num_rows + i
            if index < len(items):
                item_index = f"{index + 1:2d}. " if indexed else ""
                print(f"{item_index}{textwrap.shorten(items[index], width=column_width):{column_width}}", end="")
        print()


def print_files_in_source_directory(files):
    visible_files = [file for file in files if not file.startswith('.')]
    print()
    for i, file in enumerate(visible_files):
        print(f"{i + 1}. {file}")


def load_single_document(file_path: str) -> Document:
    ext = (os.path.splitext(file_path)[-1]).lower()
    if ext in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext]
        loader = loader_class(file_path, **loader_args)
        return loader.load()[0]

    raise ValueError(f"Unsupported file extension '{ext}'")


def stop_voice():
    engine.stop()


def speak(content):
    if not translate_docs:
        engine.setProperty('voice', voices[0].id)
    else:
        if os_running_environment == 'macOs':
            engine.setProperty('voice', voices[74].id)
        elif os_running_environment == 'windows':
            engine.setProperty('voice', voices[1].id)

    engine.say(content)
    engine.runAndWait()


def run_program():
    source_dir = 'source_documents'
    column_width = 30
    num_columns = 4
    current_directory = source_dir

    while True:
        if current_directory == source_dir:
            print("\n\033[32m[!]\033[0m Positioning to source directory \033[32m[!]\033[0m\n")
            sub_dirs = get_directories(current_directory)
            print_in_grid(sub_dirs, num_columns, column_width)

            valid_input = False
            while not valid_input:
                user_input = input(f'\n\033[94mChoose a directory ("q" to quit): \033[0m')

                if user_input.lower() == 'q':
                    print('\n\033[91m\033[1m[!]\033[0m Quitting program \033[91m\033[1m[!] \033[0m')
                    exit(0)
                try:
                    if 0 < int(user_input) <= len(sub_dirs):
                        valid_input = True
                    else:
                        print("Invalid input. Please enter a valid directory number.")
                except ValueError:
                    print("Invalid input. Please enter a number.")
                    continue

                current_directory = os.path.join(current_directory, sub_dirs[int(user_input) - 1])
        else:
            files = get_files(current_directory)
            filter_option = input(
                f'\nDo you want to search for specific files in \033[94m[{os.path.basename(current_directory)}]\033[0m? (y/n) ').lower()

            while filter_option not in ['y', 'n']:
                print("Invalid option. Please enter 'y' or 'n'.")
                filter_option = input(f'\nYou must enter y (yes) or n (no)? (y/n) ').lower()

            if filter_option == 'y':
                substrings = input('\nEnter filter (if more, separate by comma): ').lower().split(',')
                files = [filename for filename in files if
                         any(substring.strip() in filename.lower() for substring in substrings)]

            if len(files) == 0:
                print("\nNo files found matching your filter. Going back to directory listing.")
                current_directory = source_dir
            else:
                print_files_in_source_directory(files)

            user_input = input(f'\n\033[94mChoose file index ("b" to go back, "q" to quit): \033[0m')
            if user_input.lower() == 'b':
                current_directory = source_dir
            elif user_input.lower() == 'q':
                break
            elif user_input.isnumeric():
                if 0 < int(user_input) <= len(files):
                    print(f"\n\033[32m[!]\033[0m Opening document. May take some minutes! \033[32m[!]\033[0m")
                    file_path = os.path.join(current_directory, files[int(user_input) - 1])
                    document = load_single_document(file_path)

                    # Convert document content into a single string for processing
                    doc_content = "".join(document.page_content)

                    # Start and end indices for printing
                    start_index = 0

                    while True:
                        # Ensure that the end_index doesn't exceed the length of doc_content
                        end_index = min(len(doc_content), start_index + ingest_chunk_size)

                        # If start_index and end_index are the same, it means that there are no more characters to read
                        if start_index == end_index:
                            break

                        content = doc_content[start_index:end_index]

                        try:
                            console_width = os.get_terminal_size().columns  # Get console width
                        except OSError:
                            console_width = 200  # A console width

                        # Split the content by line breaks, apply the fill function to each part separately
                        paragraphs = content.split('\n')
                        justified_content = '\n'.join(textwrap.fill(p, width=console_width) for p in paragraphs)

                        if translate_docs:
                            justified_content = GoogleTranslator(source=translate_src, target=translate_dst).translate(
                                justified_content)

                        wrapper = textwrap.TextWrapper(initial_indent='\033[37m', subsequent_indent='\033[37m',
                                                       width=120)
                        print(f"{wrapper.fill(justified_content)}\033[0m\n")
                        print(f'\n\033[94mPress "n" -> next, "b" -> back, "s" -> speak, or any other key to go back '
                              f'to the main directory: \033[0m')

                        # Use the input function to wait for user input
                        user_input = input()

                        if user_input.lower() == 'n':
                            start_index = end_index
                            continue
                        elif user_input.lower() == 's':
                            speak(justified_content)
                            start_index = end_index
                            continue
                        elif user_input.lower() == 'b':
                            break
                        else:
                            current_directory = source_dir
                            break

                    print(
                        f'\n\033[94mPress "b" to go back to the book list or any other key to go back to the main directory: \033[0m')
                    user_input = input()
                    if user_input.lower() != 'b':
                        current_directory = source_dir
                else:
                    current_directory = source_dir
                    continue


if __name__ == "__main__":
    run_program()
