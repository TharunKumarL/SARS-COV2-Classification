import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

# Load Classified Clusters
classified_df = pd.read_csv("../preprocesseddata/classified_sequences.txt", sep="\t")

# Load Actual Labels
labeled_df = pd.read_csv("../labeled_protein_sequences.csv")  # Ensure it has 'Sequence_ID' and 'Label'

# Merge Data on Sequence_ID
merged_df = pd.merge(classified_df, labeled_df, on="Sequence_ID")

# Extract Predictions and Ground Truth
y_pred = merged_df["Cluster_Label"].values
y_true = merged_df["Label"].values  # Ensure 'Label' column exists in CSV

# Compute Accuracy
accuracy = accuracy_score(y_true, y_pred)

print(f" Clustering Accuracy: {accuracy * 100:.2f}%")
