import numpy as np
import pandas as pd
import pickle

def create_adjacency_matrix(sequence):
    """Creates an adjacency matrix for a given protein sequence."""
    n = len(sequence)
    A = np.zeros((n, n))
    for i in range(n - 1):
        A[i, i + 1] = 1
        A[i + 1, i] = 1  # Undirected graph
    return A

def one_hot_encode(sequence):
    """One-hot encodes an amino acid sequence."""
    AA_dict = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
    UNKNOWN_INDEX = 20
    X = np.zeros((len(sequence), 21))
    for i, aa in enumerate(sequence):
        X[i, AA_dict.get(aa, UNKNOWN_INDEX)] = 1
    return X

def graph_convolution(A, X, W):
    """Performs one-layer graph convolution using NumPy."""
    I = np.eye(A.shape[0])  # Identity matrix
    A_hat = A + I  # Add self-loops
    D = np.diag(np.sum(A_hat, axis=1))  # Degree matrix
    D_inv_sqrt = np.linalg.inv(np.sqrt(D))  # D^(-1/2)
    A_norm = D_inv_sqrt @ A_hat @ D_inv_sqrt  # Normalize adjacency matrix
    H = np.maximum(A_norm @ X @ W, 0)  # ReLU activation
    return H

def extract_features(sequence):
    """Creates a feature vector for a given sequence graph."""
    A = create_adjacency_matrix(sequence)
    X = one_hot_encode(sequence)
    W = np.random.randn(21, 32)  # Random weight matrix (21 input -> 32 output)
    H = graph_convolution(A, X, W)  # Apply GCN
    feature_vector = np.mean(H, axis=0)  # Mean pooling
    return feature_vector

# Load protein sequences
df = pd.read_csv("../labeled_protein_sequences.csv")

# Generate feature vectors
feature_vectors = []
for i, seq in enumerate(df["Protein_Sequence"]):
    feature_vectors.append(extract_features(seq))
    if (i + 1) % 100 == 0:  # Display progress every 100 sequences
        print(f"{i + 1} sequences completed")

feature_vectors = np.array(feature_vectors)
feature_vectors = np.array([extract_features(seq) for seq in df["Protein_Sequence"]])

# Save feature vectors
np.save("../preprocesseddata/protein_features.npy", feature_vectors)
print("Feature vectors saved! Shape:", feature_vectors.shape)
