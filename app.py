import os
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename

# Import the custom modules we just built
from utils import extract_text_from_pdf, chunk_text
from embeddings import create_and_save_faiss_index
from retriever import answer_question

app = Flask(__name__)

# --- Configuration ---
UPLOAD_FOLDER = "data"
FAISS_INDEX_PATH = "faiss_index/index.faiss"
CHUNKS_PATH = "faiss_index/chunks.pkl"

# Ensure our directories exist before the app starts
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("faiss_index", exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# --- Routes ---

@app.route("/")
def index():
    """Serves the main HTML user interface."""
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_pdf():
    """Handles PDF uploads and triggers the RAG ingestion pipeline."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
        
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
        
    if file and file.filename.endswith(".pdf"):
        # Security best practice: sanitize the filename
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)
        
        # --- Run our Ingestion Pipeline ---
        print(f"Starting processing pipeline for {filename}...")
        
        raw_text = extract_text_from_pdf(filepath)
        if not raw_text:
            return jsonify({"error": "Failed to extract text from PDF"}), 500
            
        chunks = chunk_text(raw_text)
        create_and_save_faiss_index(chunks, FAISS_INDEX_PATH, CHUNKS_PATH)
        
        return jsonify({"message": f"Successfully processed {filename} into {len(chunks)} chunks!"})
        
    return jsonify({"error": "Invalid file format. Please upload a PDF."}), 400

@app.route("/ask", methods=["POST"])
def ask_question():
    """Receives a question, retrieves context, and returns the AI's answer."""
    data = request.get_json()
    query = data.get("question")
    
    if not query:
        return jsonify({"error": "No question provided"}), 400
        
    try:
        # Call our retriever brain!
        answer = answer_question(query, FAISS_INDEX_PATH, CHUNKS_PATH)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Run the Flask development server
    app.run(debug=True)