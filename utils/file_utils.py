"""
some helper functions for setting up directories  etc.
"""

import os


def make_unique_directory(base_path, base_name):
    """
    create a directory in base_path starting wih base_name.
    If that directory already exists, create one with a name
    that has added -# with # being the next free number
    """
    counter = 0
    while True:
        new_path = f"{base_path}/{base_name}-{counter}"
        if not os.path.exists(new_path):
            # os.mkdir(new_path)
            return new_path, f"{base_name}-{counter}"
        counter += 1


if __name__ == "__main__":
    # Code to run when the script is executed directly
    pass
