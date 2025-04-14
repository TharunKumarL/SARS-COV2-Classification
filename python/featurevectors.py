import pandas as pd
import numpy as np
from collections import Counter
from sklearn.feature_extraction import DictVectorizer

# Load protein sequences from CSV
df = pd.read_csv('../labeled_protein_sequences.csv')  # Assumes 'Sequence' column

# Function to generate k-mers
def get_kmers(sequence, k=3):
    return [sequence[i:i+k] for i in range(len(sequence) - k + 1)]

# Choose your k
k = 3

# Convert sequences to k-mer count dictionaries
kmers_dicts = [Counter(get_kmers(seq, k)) for seq in df["Protein_Sequence"]]

# Vectorize
vec = DictVectorizer(sparse=False)
X = vec.fit_transform(kmers_dicts)  # Feature matrix

# Save the feature vectors to .npy file
np.save('../preprocesseddata/protein_kmer_features.npy', X)

print("Feature matrix shape:", X.shape)
print("Saved as 'protein_kmer_features.npy'")
