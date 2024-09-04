import sys
import os

# Add the path to the utils folder
sys.path.append(os.path.abspath('../utils'))

from zip_utils import download_7z_dropbox, extract_7z_file
from file_utils import check_files_exist

# Usage
dropbox_url = "https://www.dropbox.com/scl/fi/u4teuyczq9tkwt7hews20/leftImg8bit.7z?rlkey=u7z1l81h89adzsqefmmr2h63j&st=fb6pmpno&dl=1"  # Replace with your Dropbox link
output_path = "leftImg8bit.7z"
dropbox_url_2 = "https://www.dropbox.com/scl/fi/kijgz0jkranj3050oztqh/gtFine.7z?rlkey=tdit7rchuexmm6gz938uuzl97&st=e7lk5g7s&dl=1"  # Replace with your Dropbox link
output_path_2 = "gtFine.7z"
extract_to = "./../dataset"

file_list= ["leftImg8bit/train/aachen_000000_000019.png"]
missing_file= check_files_exist(file_list,extract_to)
print(missing_file)

if(missing_file is False):
    download_7z_dropbox(dropbox_url, output_path)
    extract_7z_file(output_path, extract_to)
    download_7z_dropbox(dropbox_url_2, output_path_2)
    extract_7z_file(output_path_2, extract_to)
else:
    pass
