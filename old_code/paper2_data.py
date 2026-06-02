import pandas as pd
import os

"""
paper2_data.py
Loads and exports datasets from Carlino et al. 2024 (Cell)
Source: https://github.com/SegataLab/cFMD
"""

# Base link to database
base_cfmd = "cFMD/cFMD_data"

# Dataset of microbial species  vs fecal samples
data_base = []
# Iterate on each dataset in the database
for dataset in os.listdir(base_cfmd):
    # Create path to dataset
    path = os.path.join(base_cfmd, dataset, f"{dataset}_taxonomic_profiles.tsv")
    # If the path is correct add the dataset to the data base
    if os.path.exists(path):
        df = pd.read_csv(path, sep="\t", index_col=0)
        data_base.append(df)

# Join the data sets to one single data set
combined = pd.concat(data_base, axis=1)
# Fill null value with zero
combined = combined.fillna(0)
print("Shape:", combined.shape)

# save the combined datasets to csv
combined.to_csv("combined_datasets.csv")

# Metadata on the microbiome
metadata = pd.read_csv("cFMD/cFMD_metadata.tsv", sep="\t")
# Export the metadata to csv
metadata.to_csv("cfmd_metadata.csv", index=False)

