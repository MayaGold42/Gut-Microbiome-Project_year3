import pandas as pd
import os
import sys
from pathlib import Path

"""
paper2_data.py
Loads and exports datasets from Carlino et al. 2024 (Cell)
Source: https://github.com/SegataLab/cFMD
"""

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR

# Base path to the cFMD data folder
base_cfmd = DATA_DIR / "cFMD-main" / "cFMD-main" / "cFMD_data"

# Dataset of microbial species  vs fecal samples
data_base = []
# Iterate on each dataset in the database
for dataset in os.listdir(base_cfmd):
    # Create path to dataset
    path = base_cfmd / dataset / f"{dataset}_taxonomic_profiles.tsv"
    # If the path is correct add the dataset to the data base
    if path.exists():
        df = pd.read_csv(path, sep="\t", index_col=0)
        data_base.append(df)

# Join the data sets to one single data set
combined = pd.concat(data_base, axis=1)
# Fill null value with zero
combined = combined.fillna(0)
print("Shape:", combined.shape)

# save the combined datasets to csv
combined.to_csv(DATA_DIR / "combined_datasets.csv")

# Metadata on the microbiome
metadata = pd.read_csv(DATA_DIR / "cFMD-main" / "cFMD-main" / "cFMD_metadata.tsv", sep="\t")
# Export the metadata to csv
metadata.to_csv(DATA_DIR / "cfmd_metadata.csv", index=False)
