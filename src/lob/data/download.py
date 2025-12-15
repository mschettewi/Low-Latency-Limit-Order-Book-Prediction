import os
import zipfile
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[3]
os.chdir(PROJECT_ROOT)

# Make data/raw and data/processed folders
Path("data/raw").mkdir(parents=True, exist_ok=True)
Path("data/processed").mkdir(parents=True, exist_ok=True)
Path("checkpoints").mkdir(parents=True, exist_ok=True)
Path("logs").mkdir(parents=True, exist_ok=True)

print("CWD:", os.getcwd())
print("Folders in CWD:", os.listdir())


def download_data():
    zip_path = PROJECT_ROOT / "data/raw/BenchmarkDatasets.zip"


    if zip_path.exists():
        print(f"File {zip_path} already exists, skipping download.")
    else:
        print(f"Downloading data to {zip_path}.")

        # get link from command line since requires auth each time
        url = input("Please provide the download URL from https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649/data: ")

        response = requests.get(url)

        if response.status_code != 200:
            print(f"Error: Received status code {response.status_code}")
            raise Exception(f"Failed to download file from {url}")

        with open(zip_path, "wb") as f:
            f.write(response.content)

    print("Unzipping data")
    # Unzip into data/raw
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall("data/raw")

    print("data/raw contains:", os.listdir("data/raw/BenchmarkDatasets"))

    OUT_PATH = "data/processed/fi2010_processed.pt"
    print("Will save to:", OUT_PATH)


if __name__ == "__main__":
    download_data()
