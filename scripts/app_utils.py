import math
import os
import textwrap

from scripts.app_environment import cli_column_number, cli_column_width


def display_source_directories(folder: str) -> list[str]:
    """
    Displays the list of existing directories in the folder directory.
    :return: The list of existing directories.
    """
    print(f"Existing directories in ./{folder}:\n\033[0m")
    return sorted((f for f in os.listdir(f"./{folder}") if not f.startswith(".")), key=str.lower)


def display_directories():
    """
    This function displays the list of existing directories in the parent directory.
    :return: The list of existing directories.
    """
    directories = display_source_directories("source_documents")

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
                print(f"{index + 1:2d}. {wrapped_directory:{cli_column_width}}", end="")
        print()  # Print a new line after each row

    return directories
