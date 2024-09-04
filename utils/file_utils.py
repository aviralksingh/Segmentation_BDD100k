import os

# check_files_exist(file_list, folder_path)
def check_files_exist(file_list, folder_path):
    missing_files = [f for f in file_list if not os.path.exists(os.path.join(folder_path, f))]

    if missing_files:
        print(f"Missing files: {missing_files}")
        return False
    else:
        print("All files are present.")
        return True
