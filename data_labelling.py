import os
import pandas as pd
from Bio import SeqIO

# Define file paths and labels
file_labels = {
    "hcov-oc43.fasta": 0,
    "humancoronavirus.fasta": 0,
    "mers-cov.fasta": 0,
    "sars-cov.fasta": 0,
    "sars-cov2.fasta": 1
}

data = []

# Directory containing FASTA files
dataset_dir = "dataset"  # Change this if needed

# Process each file
for file_name, label in file_labels.items():
    file_path = os.path.join(dataset_dir, file_name)
    if os.path.exists(file_path):
        for record in SeqIO.parse(file_path, "fasta"):
            data.append([record.id, str(record.seq), label])
    else:
        print(f"Warning: {file_name} not found in {dataset_dir}")

# Convert to DataFrame
df = pd.DataFrame(data, columns=["Sequence_ID", "Protein_Sequence", "Label"])

# Save to CSV
df.to_csv("labeled_protein_sequences.csv", index=False)
print("CSV file saved as labeled_protein_sequences.csv")