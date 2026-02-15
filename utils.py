import PyPDF2

def extract_text_from_pdf(pdf_path):
    """
    Reads a PDF file and extracts all text into a single large string.
    """
    print(f"Starting text extraction from: {pdf_path}")
    text = ""
    
    try:
        # Open the file in 'rb' (read binary) mode, which PyPDF2 requires
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            
            # Iterate through every page and append the text
            for page_num, page in enumerate(reader.pages):
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
                    
        print(f"Successfully extracted {len(text)} characters.")
        return text
        
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return None

def chunk_text(text, chunk_size=800, overlap=100):
    """
    Splits a large string into chunks of `chunk_size` words, 
    with a sliding window `overlap` of words.
    """
    print(f"Chunking text (Size: {chunk_size}, Overlap: {overlap})...")
    
    # Split the massive text string into a flat list of individual words
    words = text.split()
    chunks = []
    
    # Calculate how far to jump forward for each new chunk
    # e.g., 800 - 100 = 700. We jump forward 700 words each time.
    step_size = chunk_size - overlap
    
    # Iterate over the list of words using our step_size
    for i in range(0, len(words), step_size):
        # Slice the word list from our current index up to the chunk_size
        chunk_words = words[i : i + chunk_size]
        
        # Join the list of words back into a normal string with spaces
        chunk_text = " ".join(chunk_words)
        chunks.append(chunk_text)
        
    print(f"-> Total chunks created: {len(chunks)}")
    return chunks

# --- Quick Test Block ---
# If you run this file directly, it will test the functions.
if __name__ == "__main__":
    # Create a dummy text string of 2000 words
    dummy_text = " ".join([f"word{i}" for i in range(2000)])

    # Should create chunks of 800 words, overlapping by 100
    test_chunks = chunk_text(dummy_text, chunk_size=800, overlap=100)