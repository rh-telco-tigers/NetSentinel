# scripts/visualize_faiss_embeddings.py

import faiss
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def load_faiss_index_and_metadata(index_file='faiss_index/index.faiss', metadata_file='faiss_index/metadata.json'):
    if os.path.exists(index_file) and os.path.exists(metadata_file):
        index = faiss.read_index(index_file)
        with open(metadata_file, 'r') as f:
            metadata_store = json.load(f)
        return index, metadata_store
    else:
        print("FAISS index or metadata file not found.")
        return None, None

def visualize_embeddings(index, metadata_store):
    # Extract embeddings from the index
    embeddings = np.zeros((index.ntotal, index.d))
    for i in range(index.ntotal):
        embeddings[i] = index.reconstruct(i)
    print(f"Number of embeddings: {embeddings.shape[0]}")
    print(f"Embedding dimension: {embeddings.shape[1]}")

    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Plot the embeddings
    plt.figure(figsize=(10, 8))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c='blue', alpha=0.5)

    # Annotate with event IDs
    for idx, (x, y) in enumerate(embeddings_2d):
        event_id = metadata_store[idx].get('event_id', 'N/A')
        plt.annotate(event_id, (x, y), fontsize=8, alpha=0.7)

    plt.title('t-SNE Visualization of FAISS Embeddings')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.show()

def main():
    index, metadata_store = load_faiss_index_and_metadata()
    if index is not None and metadata_store is not None:
        visualize_embeddings(index, metadata_store)
    else:
        print("Unable to load FAISS index and metadata.")

if __name__ == "__main__":
    main()
