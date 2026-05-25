import pandas as pd
import os

base = "dietstudy_analyses/data"

# ── מיקרוביום ──
microbiome = pd.read_csv(
    f"{base}/microbiome/processed_sample/taxonomy_counts_s.txt",
    sep="\t", index_col=0
)

# ── מטא-דאטה (נבדק + יום לכל sample) ──
sample_meta = pd.read_csv(
    f"{base}/maps/food_map.txt",
    sep="\t"
).rename(columns={"#SampleID": "SampleID"})

# ── תזונה: כמות גרם לכל מזון לכל יום (dehydrated weight) ──
diet_raw = pd.read_csv(
    f"{base}/diet/processed_food/dhydrt.txt",
    sep="\t", index_col=0
)

# ── תזונה: ממוצע משוקלל עם decay (כפי שנבנה במאמר) ──
diet_decay = pd.read_csv(
    f"{base}/diet/processed_food/masterdecaydiet.txt",
    sep="\t", index_col=0
)

# הדפסת מבנה
print("=== DIET RAW ===")
print(diet_raw.shape)
print(diet_raw.iloc[:3, :4])

print("\n=== DIET DECAY ===")
print(diet_decay.shape)
print(diet_decay.iloc[:3, :4])

print("\n=== SAMPLE META עמודות מפתח ===")
print(sample_meta[['SampleID','UserName','StudyDayNo','DietDayNo']].head(5))