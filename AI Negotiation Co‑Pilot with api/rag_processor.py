import os
import pickle
import fitz  # PyMuPDF
import faiss
import numpy as np
import google.generativeai as genai

# Configure the Gemini API key
from dotenv import load_dotenv
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Use a specific directory for the index and chunks
INDEX_DIR = "faiss_index"
INDEX_PATH = os.path.join(INDEX_DIR, "pdf_index.faiss")
CHUNKS_PATH = os.path.join(INDEX_DIR, "pdf_chunks.pkl")

def process_pdf(pdf_file_path):
    """
    Extracts text from a PDF, splits it into chunks, and creates embeddings.
    """
    print("Starting PDF processing...")
    # 1. Extract Text
    doc = fitz.open(pdf_file_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()
    print(f"Extracted {len(full_text)} characters from PDF.")

    # 2. Chunk the Text (simple paragraph splitting)
    chunks = [chunk for chunk in full_text.split('\n\n') if chunk.strip()]
    print(f"Split text into {len(chunks)} chunks.")

    # 3. Create Embeddings
    print("Creating embeddings for chunks...")
    try:
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=chunks,
            task_type="RETRIEVAL_DOCUMENT"
        )
        embeddings = result['embedding']
        print("Embeddings created successfully.")
        return chunks, np.array(embeddings)
    except Exception as e:
        print(f"An error occurred during embedding: {e}")
        return None, None

def create_and_save_faiss_index(chunks, embeddings):
    """
    Creates a FAISS index from embeddings and saves it along with the chunks.
    """
    if embeddings is None or len(embeddings) == 0:
        print("No embeddings to process.")
        return

    print("Creating and saving FAISS index...")
    # Create the directory if it doesn't exist
    os.makedirs(INDEX_DIR, exist_ok=True)
    
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # Save the index and the chunks
    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)
    print(f"FAISS index and chunks saved to {INDEX_DIR}.")

def retrieve_relevant_chunks(question, k=5):
    """
    Retrieves the top k relevant chunks from the FAISS index for a given question.
    """
    if not os.path.exists(INDEX_PATH) or not os.path.exists(CHUNKS_PATH):
        return []

    print(f"Retrieving relevant chunks for question: '{question}'")
    # Load the index and chunks
    index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)

    # 1. Get embedding for the question
    try:
        question_embedding_result = genai.embed_content(
            model="models/text-embedding-004",
            content=question,
            task_type="RETRIEVAL_QUERY"
        )
        question_embedding = np.array([question_embedding_result['embedding']])
    except Exception as e:
        print(f"An error occurred during query embedding: {e}")
        return []

    # 2. Search the FAISS index
    distances, indices = index.search(question_embedding, k)

    # 3. Return the actual text chunks
    relevant_chunks = [chunks[i] for i in indices[0]]
    print(f"Found {len(relevant_chunks)} relevant chunks.")
    return relevant_chunks

    