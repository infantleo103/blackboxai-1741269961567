import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Tuple
import pickle
import os

class VectorStore:
    """Handles document embeddings and similarity search using FAISS."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the vector store.
        
        Args:
            model_name (str): Name of the sentence-transformers model to use
        """
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks = []
        
    def create_embeddings(self, chunks: List[str]) -> np.ndarray:
        """
        Create embeddings for text chunks.
        
        Args:
            chunks (List[str]): List of text chunks to embed
            
        Returns:
            np.ndarray: Matrix of embeddings
        """
        return self.model.encode(chunks, show_progress_bar=True)
    
    def build_index(self, chunks: List[str]):
        """
        Build FAISS index from text chunks.
        
        Args:
            chunks (List[str]): List of text chunks to index
        """
        self.chunks = chunks
        embeddings = self.create_embeddings(chunks)
        
        # Initialize FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        
        # Add vectors to the index
        self.index.add(embeddings.astype('float32'))
    
    def similarity_search(self, query: str, k: int = 3) -> List[Tuple[str, float]]:
        """
        Perform similarity search for a query.
        
        Args:
            query (str): Query text
            k (int): Number of results to return
            
        Returns:
            List[Tuple[str, float]]: List of (chunk, distance) tuples
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index first.")
        
        # Create query embedding
        query_embedding = self.model.encode([query])
        
        # Search the index
        distances, indices = self.index.search(
            query_embedding.astype('float32'), k
        )
        
        # Return results with distances
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.chunks):  # Ensure valid index
                results.append((self.chunks[idx], float(distance)))
        
        return results
    
    def save(self, directory: str):
        """
        Save the vector store to disk.
        
        Args:
            directory (str): Directory to save the vector store
        """
        os.makedirs(directory, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, os.path.join(directory, 'index.faiss'))
        
        # Save chunks
        with open(os.path.join(directory, 'chunks.pkl'), 'wb') as f:
            pickle.dump(self.chunks, f)
    
    def load(self, directory: str):
        """
        Load the vector store from disk.
        
        Args:
            directory (str): Directory containing the saved vector store
        """
        # Load FAISS index
        self.index = faiss.read_index(os.path.join(directory, 'index.faiss'))
        
        # Load chunks
        with open(os.path.join(directory, 'chunks.pkl'), 'rb') as f:
            self.chunks = pickle.load(f)
