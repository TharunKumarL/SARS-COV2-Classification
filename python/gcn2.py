import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import os

# SMILES strings for 20 standard amino acids
amino_acid_smiles = {
    'A': 'CC(C(=O)O)N',             # Alanine
    'C': 'C(C(=O)O)N',              # Cysteine
    'D': 'CC(C(=O)O)N',             # Aspartic Acid
    'E': 'CCC(C(=O)O)N',            # Glutamic Acid
    'F': 'C1=CC=C(C=C1)CC(C(=O)O)N',# Phenylalanine
    'G': 'C(C(=O)O)N',              # Glycine
    'H': 'C1=CN=CN1CC(C(=O)O)N',    # Histidine
    'I': 'CC(C)CC(C(=O)O)N',        # Isoleucine
    'K': 'CCCC(C(=O)O)N',           # Lysine
    'L': 'CC(C)CC(C(=O)O)N',        # Leucine
    'M': 'CSCC(C(=O)O)N',           # Methionine
    'N': 'CC(C(=O)O)N',             # Asparagine
    'P': 'C1CC(NC1)C(=O)O',         # Proline
    'Q': 'CCC(C(=O)O)N',            # Glutamine
    'R': 'C(CCN)C(C(=O)O)N',        # Arginine
    'S': 'C(C(C(=O)O)N)O',          # Serine
    'T': 'C(C(C(=O)O)N)(C)O',       # Threonine
    'V': 'CC(C)C(C(=O)O)N',         # Valine
    'W': 'C1=CC=C2C(=C1)C=CN2CC(C(=O)O)N',  # Tryptophan
    'Y': 'C1=CC(=CC=C1CC(C(=O)O)N)O' # Tyrosine
}

# Graph construction
def create_adjacency_matrix(sequence):
    n = len(sequence)
    A = np.zeros((n, n))
    for i in range(n - 1):
        A[i, i + 1] = 1
        A[i + 1, i] = 1
    return A

from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

# Initialize the generator once
morgan_gen = GetMorganGenerator(radius=2, fpSize=64)

def get_rdkit_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(morgan_gen.GetNumBits())
    fp = morgan_gen.GetFingerprint(mol)
    return np.array(fp)  # Converts RDKit bit vector to numpy


def get_rdkit_feature_matrix(sequence):
    features = []
    for aa in sequence:
        smiles = amino_acid_smiles.get(aa)
        if smiles:
            f = get_rdkit_features(smiles)
        else:
            f = np.zeros(64)  # Unknown or invalid AA
        features.append(f)
    return np.array(features)

# Graph Convolution
def graph_convolution(A, X, W):
    I = np.eye(A.shape[0])
    A_hat = A + I
    D = np.diag(np.sum(A_hat, axis=1))
    D_inv_sqrt = np.linalg.inv(np.sqrt(D))
    A_norm = D_inv_sqrt @ A_hat @ D_inv_sqrt
    H = np.maximum(A_norm @ X @ W, 0)
    return H

# Full feature extraction pipeline
def extract_features(sequence):
    A = create_adjacency_matrix(sequence)
    X = get_rdkit_feature_matrix(sequence)
    W = np.random.randn(X.shape[1], 32)  # 64 input → 32 output
    H = graph_convolution(A, X, W)
    return np.mean(H, axis=0)

# Load sequences and process
df = pd.read_csv("../labeled_protein_sequences.csv")  # Assumes 'Protein_Sequence' column
feature_vectors = []

for i, seq in enumerate(df["Protein_Sequence"]):
    feature = extract_features(seq)
    feature_vectors.append(feature)
    if (i + 1) % 100 == 0:
        print(f"{i + 1} sequences processed")

feature_vectors = np.array(feature_vectors)

# Save features
os.makedirs("../preprocesseddata", exist_ok=True)
np.save("../preprocesseddata/protein_features.npy", feature_vectors)
print("Feature vectors saved! Shape:", feature_vectors.shape)
