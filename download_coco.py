import os
import requests
from tqdm import tqdm
import zipfile


def download_file(url, save_path, chunk_size=1024):
    """
    Download a file from a URL with progress bar.

    Args:
        url (str): The URL to download from.
        save_path (str): The file path to save the downloaded file.
        chunk_size (int): The chunk size for downloading.
    """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    if os.path.exists(save_path):
        current_size = os.path.getsize(save_path)
        if current_size == total_size:
            print(f"{save_path} already downloaded.")
            return
        else:
            print(f"Resuming download of {save_path}...")
            headers = {'Range': f'bytes={current_size}-'}
            response = requests.get(url, headers=headers, stream=True)
    else:
        print(f"Downloading {save_path}...")

    with open(save_path, 'ab') as file, tqdm(
        desc=save_path,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
        initial=os.path.getsize(save_path)
    ) as bar:
        for data in response.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)


def extract_zip(file_path, extract_to):
    """
    Extract a zip file to a directory.

    Args:
        file_path (str): The path to the zip file.
        extract_to (str): The directory to extract files to.
    """
    print(f"Extracting {file_path}...")
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extraction completed: {extract_to}")


def main():
    # Define URLs for the dataset parts
    urls = {
        "train_images": {
            "url": "http://images.cocodataset.org/zips/train2017.zip",
            "save_path": "train2017.zip",
            "extract_to": "train2017"
        },
        "val_images": {
            "url": "http://images.cocodataset.org/zips/val2017.zip",
            "save_path": "val2017.zip",
            "extract_to": "val2017"
        },
        "test_images": {
            "url": "http://images.cocodataset.org/zips/test2017.zip",
            "save_path": "test2017.zip",
            "extract_to": "test2017"
        },
        "annotations": {
            "url": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
            "save_path": "annotations_trainval2017.zip",
            "extract_to": "annotations"
        },
        "test_annotations": {
            "url": "http://images.cocodataset.org/annotations/image_info_test2017.zip",
            "save_path": "image_info_test2017.zip",
            "extract_to": "annotations_test"
        }
    }

    # Choose which parts to download
    print("Select the parts of the MS COCO 2017 dataset to download:")
    print("1. Training Images [118K/18GB]")
    print("2. Validation Images [5K/1GB]")
    print("3. Test Images [41K/6GB]")
    print("4. Annotations [241MB]")
    print("5. Test Annotations [1MB]")
    print("6. Download All")

    choices = input(
        "Enter the numbers corresponding to your choices (e.g., 1 2 4): ").split()
    selected_parts = []
    all_choices = {'1', '2', '3', '4', '5', '6'}

    if not set(choices).issubset(all_choices):
        print("Invalid selection. Please enter valid numbers.")
        return

    if '6' in choices:
        selected_parts = ['train_images', 'val_images',
                          'test_images', 'annotations', 'test_annotations']
    else:
        if '1' in choices:
            selected_parts.append('train_images')
        if '2' in choices:
            selected_parts.append('val_images')
        if '3' in choices:
            selected_parts.append('test_images')
        if '4' in choices:
            selected_parts.append('annotations')
        if '5' in choices:
            selected_parts.append('test_annotations')

    # Create a directory to store the dataset
    os.makedirs('coco2017', exist_ok=True)
    os.chdir('coco2017')

    # Download and extract selected parts
    for part in selected_parts:
        url = urls[part]['url']
        save_path = urls[part]['save_path']
        extract_to = urls[part]['extract_to']

        # Download the file
        download_file(url, save_path)

        # Extract the file
        extract_zip(save_path, extract_to)

        # Optionally, remove the zip file to save space
        os.remove(save_path)
        print(f"Removed zip file: {save_path}")

    print("All selected parts have been downloaded and extracted successfully.")


if __name__ == "__main__":
    main()
