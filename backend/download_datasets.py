import pandas as pd
import os
import json

os.makedirs("app/data", exist_ok=True)

datasets = {
    "titanic": "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv",
    "iris":    "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv",
    "wine":    "https://raw.githubusercontent.com/dsrscientist/dataset1/master/winequality-red.csv",
    "heart":   "https://raw.githubusercontent.com/dsrscientist/dataset1/master/heart_disease.csv",
    "house":   "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv",
}

meta = {
    "titanic": {"name": "Titanic Survival",  "target": "Survived", "task": "classification", "rows": 0, "cols": 0},
    "iris":    {"name": "Iris Species",       "target": "species",  "task": "classification", "rows": 0, "cols": 0},
    "wine":    {"name": "Wine Quality",       "target": "quality",  "task": "regression",     "rows": 0, "cols": 0},
    "heart":   {"name": "Heart Disease", "target": "target", "task": "classification", "rows": 0, "cols": 0},
    "house":   {"name": "House Prices",       "target": "medv",     "task": "regression",     "rows": 0, "cols": 0},
}

for key, url in datasets.items():
    try:
        df = pd.read_csv(url)
        df.to_parquet(f"app/data/{key}.parquet", index=False)
        meta[key]["rows"] = len(df)
        meta[key]["cols"] = len(df.columns)
        print(f"✓ {key} — {df.shape[0]} rows, {df.shape[1]} cols")
    except Exception as e:
        print(f"✗ {key} — {e}")

with open("app/data/meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print("\nDone. Check app/data/")