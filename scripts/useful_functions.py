import os

def createFolder(directory):
    """
    Create a folder.
    :param directory: Path to the new directory. This is the parent directory at which the new folder will be created.
    :return: A folder at the specified directory.

    # Example
    createFolder('./data/')  # Creates a folder in the current directory called data
    """
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)


