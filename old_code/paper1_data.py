import pandas as pd
import sys
from pathlib import Path

"""
paper1_data.py
Loads and exports datasets from Johnson et al. 2019 (Cell Host & Microbe)
Source: https://github.com/knights-lab/dietstudy_analyses
"""

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR

# Base path to the dietstudy_analyses data folder
base = DATA_DIR / "dietstudy_analyses-master" / "data"

# Take relevant datasets print top rows and  export them to csv after
#  Fecal microbiome measurements - species level
microbiome = pd.read_csv(base / "microbiome/processed_sample/taxonomy_counts_s.txt", sep="\t", index_col=0)
print("=== microbiome ===")
print(microbiome.shape)
print(microbiome.head())
print("***********************************")


# Metadata for each fecal sample: subject info, study day, demographics
sample_meta = pd.read_csv(base / "maps/food_map.txt", sep="\t").rename(columns={"#SampleID": "SampleID"})
print("=== sample_meta ===")
print(sample_meta.shape)
print(sample_meta.head())
print("***********************************")

# Daily food intake in grams per sample
diet_raw = pd.read_csv(base / "diet/processed_food/dhydrt.txt", sep="\t", index_col=0)
print("=== DIET RAW ===")
print(diet_raw.shape)
print(diet_raw.head())
print("***********************************")


# Daily food intake with exponential decay weighting of dietary history
diet_decay = pd.read_csv(base / "diet/processed_food/masterdecaydiet.txt", sep="\t", index_col=0)
print("=== diet decay ===")
print(diet_decay.shape)
print(diet_decay.head())
print("***********************************")

# Nutritional totals per sample
nutrition_totals = pd.read_csv(base / "diet/nutrition_totals.txt", sep="\t", index_col=0)
print("=== nutrition totals ===")
print(nutrition_totals.shape)
print(nutrition_totals.head())
print("***********************************")

# CLR-transformed microbiome data
microbiome_clr = pd.read_csv(base / "microbiome/processed_sample/taxonomy_clr_s.txt", sep="\t", index_col=0)
print("=== microbiome CLR ===")
print(microbiome_clr.shape)
print(microbiome_clr.head())
print("***********************************")



# Fiber intake per sample
diet_fiber = pd.read_csv(base / "diet/diet.fiber.txt", sep="\t", index_col=0)
print("=== diet fiber ===")
print(diet_fiber.shape)
print(diet_fiber.head())
print("***********************************")

# Fiber intake per food item averaged per subject
fiber_avg = pd.read_csv(base / "diet/fiber/fiber.smry.otu.txt", sep="\t", index_col=0)
print("=== fiber avg ===")
print(fiber_avg.shape)
print(fiber_avg.head())
print("***********************************")

# Average count microbiome per subject across all days
micro_avg_counts = pd.read_csv(base / "microbiome/processed_average/UN_taxonomy_counts_s.txt", sep="\t", index_col=0)
print("=== micro avg counts ===")
print(micro_avg_counts.shape)
print(micro_avg_counts.head())
print("***********************************")

# Average clr microbiome per subject across all days
micro_avg_clr = pd.read_csv(base / "microbiome/processed_average/UN_taxonomy_clr_s.txt", sep="\t", index_col=0)
print("=== micro avg clr ===")
print(micro_avg_clr.shape)
print(micro_avg_clr.head())
print("***********************************")

# Average norm microbiome per subject across all days
micro_avg_norm = pd.read_csv(base / "microbiome/processed_average/UN_taxonomy_norm_s.txt", sep="\t", index_col=0)
print("=== micro avg norm ===")
print(micro_avg_norm.shape)
print(micro_avg_norm.head())
print("***********************************")




# Save all the datasets to csv
microbiome.to_csv(DATA_DIR / "microbiome.csv")
sample_meta.to_csv(DATA_DIR / "sample_meta.csv", index=False)
diet_raw.to_csv(DATA_DIR / "diet_raw.csv")
diet_decay.to_csv(DATA_DIR / "diet_decay.csv")
nutrition_totals.to_csv(DATA_DIR / "nutrition_totals.csv")
microbiome_clr.to_csv(DATA_DIR / "microbiome_clr.csv")
diet_fiber.to_csv(DATA_DIR / "diet_fiber.csv")
fiber_avg.to_csv(DATA_DIR / "fiber_avg.csv")
micro_avg_counts.to_csv(DATA_DIR / "micro_avg_counts.csv")
micro_avg_clr.to_csv(DATA_DIR / "micro_avg_clr.csv")
micro_avg_norm.to_csv(DATA_DIR / "micro_avg_norm.csv")
