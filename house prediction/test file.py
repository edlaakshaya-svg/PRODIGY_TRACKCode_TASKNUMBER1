import pandas as pd

# Load training data (make sure train.csv is in the same folder)
train = pd.read_csv("train.csv")

print("File loaded successfully!")
print(train.head())  # show first 5 rows
