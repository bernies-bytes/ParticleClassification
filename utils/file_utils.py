import os

def make_unique_directory(base_path, base_name):
    counter = 1
    while True:
        new_path = f"{base_path}/{base_name}-{counter}"
        if not os.path.exists(new_path):
            # os.mkdir(new_path)
            return new_path, f"{base_name}-{counter}"
        counter += 1


if __name__ == "__main__":
    # Code to run when the script is executed directly
    pass
