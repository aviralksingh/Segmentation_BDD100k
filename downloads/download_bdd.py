
import sys
import os

# Add the path to the utils folder
sys.path.append(os.path.abspath('../utils'))

from zip_utils import download_7z_dropbox, extract_7z_file
from file_utils import check_files_exist

# Usage
dropbox_url = "https://www.dropbox.com/scl/fi/fuqhwgk26r4xrzyu700ko/bdd_sample_dataset.7z?rlkey=tdkdlrd4lt92gdpfy7gpps3g4&st=ic85fr7u&dl=1"  # Replace with your Dropbox link
output_path = "bdd_sample_dataset.7z"
extract_to = "./../dataset"

file_list= ["bdd_image_180_320.npy","bdd_label_180_320.npy","bdd_metric_demo_gt.npy"]
missing_file= check_files_exist(file_list,extract_to)
print(missing_file)

if(missing_file is False):
    download_7z_dropbox(dropbox_url, output_path)
    extract_7z_file(output_path, extract_to)
else:
    pass

