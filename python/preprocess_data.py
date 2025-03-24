import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Load labeled protein sequences from CSV
csv_file = "../labeled_protein_sequences.csv"
df = pd.read_csv(csv_file)

def sequence_to_graph(sequence):
    G = nx.Graph()
    for i, amino_acid in enumerate(sequence):
        G.add_node(i, label=amino_acid)
        if i > 0:
            G.add_edge(i - 1, i)  # Connect consecutive amino acids
    return G

# Convert sequences to graphs
df["Graph"] = df["Protein_Sequence"].apply(sequence_to_graph)

# Save and visualize graphs
for idx, row in df.iterrows():
    plt.figure(figsize=(5, 5))
    nx.draw(row["Graph"], with_labels=True, node_size=300, font_size=8)
    plt.title(f"Graph for {row['Sequence_ID']}")
    plt.show()

print("Graphs generated successfully!")