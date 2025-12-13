import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.lob.data.download import download_data
from src.lob.data.dataset import analyze_data_distributions
from src.lob.data.preprocess import process_data

if __name__ == "__main__":
    download_data()
    process_data()
    analyze_data_distributions()
