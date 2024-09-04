import sys
import os

# Add the path to the utils folder
sys.path.append(os.path.abspath('../utils'))

from zip_utils import download_7z_dropbox, extract_7z_file
from file_utils import check_files_exist

# Usage
dropbox_url = "https://www.dropbox.com/scl/fi/r3mshswcvzx9wo3znssl5/weights.7z?rlkey=1fqlxp5f05m1cgk4lii4herv2&st=b1wfycuf&dl=1"  # Replace with your Dropbox link
output_path = "weights.7z"
extract_to = "./../weights"

file_list= ["deeplabv3_plus_resnet50_os.pt",
            "FCN_resnet50_baseline.pt",
            "PSPNet_resnet50_aux.pt",
            "segformer_mit_b2.pt",
            "segformer_mit_b3",
            "segformer_mit_b3_imagenet_weights.pt",
            "segformer_mit_b3_cs_pretrain_19CLS_512_1024_CE_loss",
            "UNet_baseline.pt",
            "upsample_test_img.pt"]
missing_file= check_files_exist(file_list,extract_to)
print(missing_file)

if(missing_file is False):
    download_7z_dropbox(dropbox_url, output_path)
    extract_7z_file(output_path, extract_to)
else:
    pass

