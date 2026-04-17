import os
import pandas as pd
from datasets import load_dataset
import yaml

def load_config(config_path="configs/train.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def main():
    config = load_config()
    data_cfg = config["data"]
    
    os.makedirs("sample_data", exist_ok=True)
    
    dataset = load_dataset(data_cfg["dataset_name"], revision="refs/convert/parquet")
    labels = dataset["train"].features["label"].names
    
    train_full = dataset["train"].shuffle(seed=data_cfg["seed"]).select(range(data_cfg["train_sample_size"]))
    split = train_full.train_test_split(test_size=data_cfg["val_size"], seed=data_cfg["seed"])
    
    train_df = pd.DataFrame([{
        "text": x["text"], 
        "label": labels[x["label"]]
    } for x in split["train"]])
    
    val_df = pd.DataFrame([{
        "text": x["text"], 
        "label": labels[x["label"]]
    } for x in split["test"]])
    
    test_full = dataset["test"].shuffle(seed=data_cfg["seed"]).select(range(data_cfg["test_sample_size"]))
    
    test_df = pd.DataFrame([{
        "text": x["text"], 
        "label": labels[x["label"]]
    } for x in test_full])
    
    train_df.to_csv("sample_data/train.csv", index=False)
    val_df.to_csv("sample_data/val.csv", index=False)
    test_df.to_csv("sample_data/test.csv", index=False)

if __name__ == "__main__":
    main()