import sys
import os

# Add the path to the utils folder
sys.path.append(os.path.abspath('../utils'))

from zip_utils import download_7z_dropbox, extract_7z_file
from file_utils import check_files_exist

# Usage
dropbox_url = "https://www.dropbox.com/scl/fi/890ig06vdlbqbbjblix1f/weights.7z?rlkey=d626peei17y9mo9v75bbtv1vf&st=sthayfh8&dl=1"  # Replace with your Dropbox link
output_path = "weights.7z"
extract_to = "./../weights1"

file_list= ["deeplabv3_plus_resnet50_os.pt",
            "FCN_resnet50_baseline.pt",
            "PSPNet_resnet50_aux.pt",
            "segformer_mit_b2.pt",
            "segformer_mit_b3_imagenet_weights.pt",
            "UNet_baseline.pt",
            "upsample_test_img.pt"]
missing_file= check_files_exist(file_list,extract_to)
print(missing_file)

if(missing_file is False):
    download_7z_dropbox(dropbox_url, output_path)
    extract_7z_file(output_path, extract_to)
else:
    pass

