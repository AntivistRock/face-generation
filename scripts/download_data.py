import os

import fire


def download_and_extract(target_dir: str) -> None:
    """Downloads a ZIP file from a URL using wget, extracts it using unzip,
    and deletes the ZIP file after extraction.
    """

    url = "https://www.dropbox.com/scl/fi/di9wevuxwd4ks0cvhk0lg/\
        img_align_celeba.zip?rlkey=yzi4rakarun7fxj8yl0oy5v6t&st=8demnpch&dl=0"

    os.makedirs(target_dir, exist_ok=True)

    zip_path = os.path.join(target_dir, "data.zip")
    wget_command = f"wget -nc -O {zip_path} '{url}'"
    print(f"Executing: {wget_command}")
    os.system(wget_command)

    unzip_command = f"unzip -n -qq {zip_path} -d {target_dir}"
    print(f"Executing: {unzip_command}")
    os.system(unzip_command)

    print(f"Deleting {zip_path}...")
    os.remove(zip_path)
    print(f"Deleted {zip_path}.")


def check_dataset_size():
    os.system("ls -1 ../data/val/dog | wc -l")


if __name__ == "__main__":
    fire.Fire(download_and_extract)
