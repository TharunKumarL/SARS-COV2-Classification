import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# Load Feature Vectors
feature_vectors = np.load("../preprocesseddata/protein_features.npy")

# Apply K-Means Clustering
num_clusters = 2  # Binary Classification
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
clusters = kmeans.fit_predict(feature_vectors)  # Output: Cluster labels (0 or 1)

# Load Sequence IDs
df = pd.read_csv("../labeled_protein_sequences.csv")  # Assuming this contains sequence IDs
sequence_ids = df["Sequence_ID"].values  # Ensure this column exists in CSV

# Ensure arrays are the same length
if len(sequence_ids) != len(clusters):
    raise ValueError("Mismatch in sequence IDs and cluster labels!")

# Create DataFrame with Classification Results
output_df = pd.DataFrame({"Sequence_ID": sequence_ids, "Cluster_Label": clusters})

# Save as Text File
output_file = "../preprocesseddata/classified_sequences.txt"
output_df.to_csv(output_file, sep="\t", index=False, header=True)

print(f" Classification completed and saved to {output_file}")
