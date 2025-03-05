import os
import kagglehub
import splitfolders

if os.path.exists("data/train") and os.path.exists("data/test"):
    print("Train and Test sets Ready.")
else:

    data = kagglehub.dataset_download("alessiocorrado99/animals10")
    splitfolders.ratio("data/raw-img", output="data", seed=42, ratio=(0.8, 0.2))
    os.rename("data/val", "data/test")
    print("Dataset downloaded and split into train and test sets")
