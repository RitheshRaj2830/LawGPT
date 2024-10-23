from flask import Flask, request, jsonify, render_template
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

app = Flask(__name__)

# Initialize embeddings and vector store (IPC Database)
embeddings = HuggingFaceEmbeddings(
    model_name="nomic-ai/nomic-embed-text-v1",
    model_kwargs={"trust_remote_code": True},
)

# Load FAISS vector store
db = FAISS.load_local("ipc_vector_db", embeddings, allow_dangerous_deserialization=True)
db_retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})

@app.route('/')
def home():
    return render_template('main.html')

@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')

# Endpoint for handling user queries and fetching results from FAISS
@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('input')

    try:
        # Perform a similarity search on the FAISS vector store
        results = db_retriever.get_relevant_documents(user_input)

        # Combine the top result into the response
        if results:
            response = results[0].page_content  # Get the content of the first result
        else:
            response = "I couldn't find anything relevant. Can you rephrase your question?"

        # Return the response to the frontend
        return jsonify({"response": response})

    except Exception as e:
        # Handle any errors during the FAISS retrieval
        return jsonify({"response": f"Error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
