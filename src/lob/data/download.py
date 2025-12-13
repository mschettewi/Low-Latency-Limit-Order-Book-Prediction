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
    zip_path = PROJECT_ROOT / "data/raw/LOBSTER_SampleFile_AAPL_2012-06-21_10.zip"

    if zip_path.exists():
        print(f"File {zip_path} already exists, skipping download.")
        return

    url = "https://data.lobsterdata.com/info/sample/LOBSTER_SampleFile_AAPL_2012-06-21_10.zip"
    response = requests.get(url)

    if response.status_code != 200:
        raise Exception(f"Failed to download file from {url}")

    with open(zip_path, "wb") as f:
        f.write(response.content)

    # Unzip into data/raw
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall("data/raw")

    print("data/raw contains:", os.listdir("data/raw"))

    # Pick out the orderbook and message file names automatically
    files_in_raw = os.listdir("data/raw")
    RAW_ORDERBOOK_NAME = [f for f in files_in_raw if "orderbook" in f][0]
    RAW_MESSAGE_NAME = [f for f in files_in_raw if "message" in f][0]

    RAW_ORDERBOOK_PATH = f"data/raw/{RAW_ORDERBOOK_NAME}"
    RAW_MESSAGE_PATH = f"data/raw/{RAW_MESSAGE_NAME}"
    OUT_PATH = "data/processed/aapl_lobster_balanced.pt"

    print("Orderbook path:", RAW_ORDERBOOK_PATH)
    print("Message path:", RAW_MESSAGE_PATH)
    print("Will save to:", OUT_PATH)


if __name__ == "__main__":
    download_data()
