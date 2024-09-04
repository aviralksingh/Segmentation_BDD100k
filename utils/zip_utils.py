import requests
import py7zr
import zipfile

def download_7z_dropbox(dropbox_url, output_path):
    direct_link = dropbox_url.replace("?dl=0", "?dl=1")
    response = requests.get(direct_link, stream=True)

    if response.status_code == 200:
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print(f"Downloaded {output_path}")
    else:
        print(f"Failed to download file. Status code: {response.status_code}")

def extract_7z_file(file_path, extract_to):
    with py7zr.SevenZipFile(file_path, mode='r') as archive:
        archive.extractall(path=extract_to)
    print(f"Extracted {file_path} to {extract_to}")

def download_zip_dropbox(dropbox_url, output_path):
    # Convert the Dropbox URL to a direct download link
    direct_link = dropbox_url.replace("?dl=0", "?dl=1")

    # Send a GET request to the Dropbox link
    response = requests.get(direct_link, stream=True)

    # Check if the request was successful
    if response.status_code == 200:
        # Write the content to a file
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:  # Filter out keep-alive chunks
                    f.write(chunk)
        print(f"Downloaded {output_path}")
    else:
        print(f"Failed to download file. Status code: {response.status_code}")


def extract_zip_file(file_path, extract_to):
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted {file_path} to {extract_to}")
