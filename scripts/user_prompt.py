import math
import os
import textwrap

from dotenv import set_key

from .user_environment import cli_column_number, cli_column_width


def prompt():
    """
    This function prompts the user to select an existing directory or create a new one to store source material.
    If an existing directory is selected, it checks if the directory is empty and prompts the user to create files
    in the directory if it is empty. It sets the directory paths as environment variables and returns them.
    Returns:
     - selected_directory_path (str): The path of the selected directory.
     - selected_db_path (str): The path of the database directory for the selected directory.
    """

    def _display_directories():
        """
        This function displays the list of existing database directories in the ./db directory.
        """
        print(f"\n\033[94mExisting databases in ./db folder:\n\033[0m")
        directories = sorted((f for f in os.listdir("db") if not f.startswith(".")), key=str.lower)

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

    directories = _display_directories()
    valid_input = False

    while not valid_input:
        user_input = input("\n\033[94mEnter the index number of the database (or more of them separated by comma) (q for quit): \033[0m")

        # Check if the user wants to quit
        if user_input.strip().lower() == 'q':
            exit(0)

        # Split the input by comma and create a list
        selected_directory_list = user_input.split(',')

        # Validate user input
        try:
            for dir_index in selected_directory_list:
                # Check if the input can be converted to integer and it's within valid range
                dir_index = int(dir_index.strip())
                if 1 <= dir_index <= len(directories):
                    valid_input = True
                else:
                    print("Invalid input: Index is out of range. Please try again.")
                    valid_input = False
                    break
        except ValueError:
            print("Invalid input: Please enter a number. Try again.")

        if valid_input:
            # Split the input by comma and create a list
            selected_directory_list = user_input.split(',')
            # Use the latest directory from the list
            selected_directory = directories[int(selected_directory_list[-1]) - 1]
            # let's store only the latest as a path variable
            selected_directory_path = f"./source_documents/{selected_directory}"
            selected_db_path = f"./db/{selected_directory}"
            set_key('.env', 'INGEST_SOURCE_DIRECTORY', selected_directory_path)
            set_key('.env', 'INGEST_PERSIST_DIRECTORY', selected_db_path)
            print(f"Storing env variable defaults: {selected_db_path}")
            return [directories[int(dir) - 1] for dir in selected_directory_list]
