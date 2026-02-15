import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Load our embedding model. 
# 'all-MiniLM-L6-v2' is a lightweight, fast, and free open-source model perfect for this.
print("Loading Embedding Model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

def create_and_save_faiss_index(chunks, index_path, chunks_path):
    """
    Converts text chunks into embeddings and saves them into a FAISS index.
    Also saves the raw chunks so we can retrieve the text later.
    """
    print(f"Generating embeddings for {len(chunks)} chunks. This might take a moment...")
    
    # 1. Convert text chunks to vector embeddings
    # We use model.encode(). It returns a numpy array of vectors.
    embeddings = model.encode(chunks, show_progress_bar=True)
    
    # FAISS requires the vectors to be float32
    embeddings = np.array(embeddings).astype('float32')
    
    # 2. Create the FAISS Index
    # We need to tell FAISS the dimensionality of our vectors. 
    # 'all-MiniLM-L6-v2' outputs 384-dimensional vectors.
    dimension = embeddings.shape[1] 
    
    # IndexFlatL2 measures the Euclidean distance (straight-line distance) between vectors.
    index = faiss.IndexFlatL2(dimension)
    
    # Add our embeddings into the index
    index.add(embeddings)
    print(f"-> Successfully added {index.ntotal} vectors to FAISS index.")
    
    # 3. Save the index and the chunks to disk
    # Ensure the directory exists
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    
    faiss.write_index(index, index_path)
    
    # We use the 'pickle' library to save our Python list of text chunks
    with open(chunks_path, 'wb') as f:
        pickle.dump(chunks, f)
        
    print(f"Saved FAISS index to {index_path} and chunks to {chunks_path}")

def get_query_embedding(query):
    """
    Converts a user's string question into a vector using the exact same model.
    """
    # FAISS expects a 2D array, so we wrap the query in a list and reshape it
    query_vector = model.encode([query])
    return np.array(query_vector).astype('float32')

def load_faiss_index_and_chunks(index_path, chunks_path):
    """
    Loads the saved FAISS index and the raw chunks from disk.
    """
    if not os.path.exists(index_path) or not os.path.exists(chunks_path):
        raise FileNotFoundError("Index or chunks file not found. Please process a PDF first.")
        
    index = faiss.read_index(index_path)
    
    with open(chunks_path, 'rb') as f:
        chunks = pickle.load(f)
        
    return index, chunks