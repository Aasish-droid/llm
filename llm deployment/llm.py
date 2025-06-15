import faiss
import numpy as np
import mysql.connector
from flask import Flask, request, render_template_string
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer

# Initialize Flask app
app = Flask(__name__)

# Database Configuration
db_conn = mysql.connector.connect(
    host="localhost",
    user="root",  # Your MySQL username
    password="yourpassword",  # Your MySQL password
    database="ai_code_db"
)
cursor = db_conn.cursor()

# AI Model Configuration
MODEL_NAME = "EleutherAI/gpt-neo-1.3B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Create FAISS Index
def update_faiss_index():
    """Fetch documents, encode embeddings, and update FAISS index."""
    cursor.execute("SELECT text FROM documents")
    documents = [row[0] for row in cursor.fetchall()]
    if documents:
        doc_embeddings = embed_model.encode(documents)
        doc_embeddings = np.array(doc_embeddings, dtype=np.float32)
        faiss_index = faiss.IndexFlatL2(doc_embeddings.shape[1])
        faiss_index.add(doc_embeddings)
        return faiss_index, documents
    return None, []

faiss_index, documents = update_faiss_index()

# Store a New Document in MySQL
def store_document(text):
    cursor.execute("INSERT INTO documents (text) VALUES (%s)", (text,))
    db_conn.commit()

# Retrieve Relevant Context for Query
def retrieve_context(query, top_k=2):
    """Retrieve top-k relevant documents using FAISS."""
    global faiss_index, documents
    faiss_index, documents = update_faiss_index()  # Update index dynamically
    if not documents:
        return "No relevant documents found."
    
    query_embedding = embed_model.encode([query])
    query_embedding = np.array(query_embedding, dtype=np.float32)
    distances, indices = faiss_index.search(query_embedding, top_k)
    retrieved_docs = [documents[i] for i in indices[0]]
    return " ".join(retrieved_docs)

# Generate Code Using GPT-Neo
def generate_code(prompt):
    """Retrieve context + generate code using GPT-Neo."""
    context = retrieve_context(prompt)
    full_prompt = f"Context: {context}\nUser Query: {prompt}"
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    output = generator(full_prompt, max_length=200)[0]["generated_text"]
    return output

# HTML Frontend
HTML_PAGE = """
<!DOCTYPE html>
<head>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/bootstrap/5.3.0/css/bootstrap.min.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        body { background: linear-gradient(135deg, #1E88E5, #D32F2F); color: #fff; text-align: center; padding: 20px; }
        .container { max-width: 700px; margin: auto; padding: 20px; background: rgba(255, 255, 255, 0.1); border-radius: 15px; box-shadow: 0px 4px 10px rgba(0,0,0,0.3); }
        textarea { width: 100%; border-radius: 8px; padding: 10px; }
        .btn { background-color: #FFC107; border: none; color: black; padding: 10px 20px; font-size: 18px; cursor: pointer; border-radius: 10px; }
        .btn:hover { background-color: #FF9800; }
        .generated-code { background: #333; padding: 10px; border-radius: 10px; color: #fff; text-align: left; font-family: "Courier New", monospace; }
    </style>
</head>
<body>
    <div class="container">
        <h1><i class="fa-solid fa-code"></i> AI Code Generator with Hybrid RAG 🚀</h1>
        <p>Generate code snippets using GPT-Neo with FAISS-powered retrieval.</p>

        <form method="POST" action="/generate">
            <textarea name="user_input" rows="5" placeholder="Enter your coding prompt..."></textarea><br><br>
            <button type="submit" class="btn"><i class="fa-solid fa-robot"></i> Generate Code</button>
        </form>

        {% if generated %}
        <h2><i class="fa-solid fa-terminal"></i> Generated Code:</h2>
        <pre class="generated-code">{{ generated }}</pre>
        {% endif %}
    </div>
</body>
"""

# Home Route
@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML_PAGE)

# Generate Code Route
@app.route("/generate", methods=["POST"])
def generate():
    user_input = request.form.get("user_input", "")
    generated = ""
    if user_input:
        generated = generate_code(user_input)
    return render_template_string(HTML_PAGE, generated=generated)

# Store Documents Route (Manually Add Documents)
@app.route("/add_document", methods=["POST"])
def add_document():
    """API Route to manually store documents."""
    new_text = request.form.get("document_text", "")
    if new_text:
        store_document(new_text)
        return {"message": "Document added successfully!"}
    return {"error": "Document cannot be empty."}

# Run Flask App
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
