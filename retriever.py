import os
import numpy as np
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Import our helper functions from embeddings.py
from embeddings import get_query_embedding, load_faiss_index_and_chunks

# Load environment variables
load_dotenv()

# Initialize the Gemini Client
# It will automatically look for the GEMINI_API_KEY in your environment variables
client = genai.Client()

def retrieve_context(query, index, chunks, top_k=5):
    """
    Searches the FAISS index for the top_k most relevant chunks to the query.
    """
    print(f"\nSearching for top {top_k} chunks for query: '{query}'")
    
    query_vector = get_query_embedding(query)
    distances, indices = index.search(query_vector, top_k)
    
    retrieved_chunks = []
    
    print("--- Retrieval Statistics ---")
    for i, idx in enumerate(indices[0]):
        dist = distances[0][i]
        chunk_text = chunks[idx]
        retrieved_chunks.append(chunk_text)
        print(f"Rank {i+1} | Chunk ID: {idx} | L2 Distance: {dist:.4f}")
        
    print("----------------------------\n")
    return retrieved_chunks

def generate_answer(query, context_chunks):
    """
    Sends the user's question and the retrieved context to Gemini Pro to generate an answer.
    """
    print("Generating answer via Gemini Pro...")
    
    # Combine the top 5 chunks into a single string
    context_text = "\n\n".join(context_chunks)
    
    # Construct the System Instruction
    system_instruction = (
        "You are a highly intelligent and helpful teaching assistant. "
        "You will be provided with excerpts from a textbook. "
        "Answer the user's question using ONLY the provided context. "
        "If the answer is not contained in the context, say 'I cannot find the answer in the textbook.' "
        "Do not use outside knowledge."
    )
    
    # The actual prompt sent to the model
    user_prompt = f"Context:\n{context_text}\n\nQuestion: {query}"
    
    try:
        # Call the Gemini API
        # Note: You can also use 'gemini-1.5-pro' or 'gemini-2.5-flash' depending on your needs
        response = client.models.generate_content(
            model="gemini-2.5-flash", 
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.2 # Keep it low for factual textbook answers
            )
        )
        
        return response.text
        
    except Exception as e:
        print(f"Error communicating with Gemini: {e}")
        return "Sorry, I encountered an error while generating the answer."

def answer_question(query, index_path, chunks_path):
    """
    Orchestrates the full pipeline: Load -> Retrieve -> Generate.
    """
    index, chunks = load_faiss_index_and_chunks(index_path, chunks_path)
    context_chunks = retrieve_context(query, index, chunks, top_k=5)
    answer = generate_answer(query, context_chunks)
    
    return answer