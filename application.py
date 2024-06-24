from typing import Optional, Generator
import pandas as pd
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
import json
from flask import Flask, jsonify, request
from config import DEVICE, EMBEDDINGS_MODEL, LLM_API_KEY, MODEL

app = Flask(__name__)
DB_PATH = "./db"

# Initialize the chat model at the module level
chat = ChatGroq(
    temperature=0, 
    groq_api_key=LLM_API_KEY, 
    model_name=MODEL
)

embedding = HuggingFaceBgeEmbeddings(
    model_name=EMBEDDINGS_MODEL,
    model_kwargs={'device': DEVICE},
    encode_kwargs={'normalize_embeddings': True}
)

def generate_vector_db(documents):

    vectordb = Chroma.from_documents(
        documents=documents, 
        embedding=embedding,
        persist_directory=DB_PATH
    )
    return True

def create_vector_db(df: pd.DataFrame) -> str:
    """Create a vector database from a DataFrame."""
    if 'QUESTIONS' not in df.columns or 'ANSWERS' not in df.columns:
        raise KeyError("DataFrame must contain 'question' and 'answer' columns")

    documents = df['QUESTIONS'] + " -> " + df['ANSWERS']
    chunk_size = documents.str.len().max() #chuck size must be relative to the maximium question, answer pair in the dataset
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
    texts = text_splitter.split_text('\n'.join(documents.values))
    documents = text_splitter.create_documents(texts)

    db_path = generate_vector_db(documents)

    return db_path

@app.route('/chat', methods=['POST'])
def chat_bot() -> str:
    """Generate a response to a question using a vector database."""
    question = request.get_json()['question']

    # Create prompt template with detailed instructions and context
    prompt_template = """
    You are a highly knowledgeable and capable AI assistant called Octopus, you help work at Octopus which is an Online prominent e-commerce platform based in Uganda.
    It offers a diverse range of products, catering to various customer needs, including electronics, fashion, home appliances,
   beauty products, and groceries. The store is designed to provide a convenient shopping experience, featuring user-friendly
    navigation, secure payment options, and efficient delivery services across Uganda. With a commitment to customer satisfaction,
    Octopus Online Store frequently offers promotions and discounts, 
    aiming to make quality products accessible and affordable for its growing customer base. 
    Your role is to provide accurate and informative responses to questions by leveraging the provided 
    context.
    Context: {context}

    To answer the question effectively, please consider the following points:
    - Carefully analyze the context and identify relevant information.
    - Combine the context with your own knowledge and understanding to formulate a comprehensive response.
    - If the context is insufficient or lacks relevant information, use your reasoning abilities to fill in the gaps and provide additional insights or explanations.
    - Ensure your response is clear, concise, and easy to understand.
    - If the question cannot be satisfactorily answered based on the given context, politely acknowledge the limitations and provide a best-effort response.

    Question: {question}

    Your Response:
    """
    
    # Initialize prompt with context and question variables
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    # Initialize vector database for context retrieval
    vectordb = Chroma(persist_directory=DB_PATH, embedding_function=embedding)

    # Setup QA chain with retrieval and prompt
    qa_chain = RetrievalQA.from_chain_type(
        chat,
        retriever=vectordb.as_retriever(),
        chain_type_kwargs={"prompt": prompt}
    )

    response = qa_chain.invoke({"query": question})

    # Ensure the response is in the expected format
    if isinstance(response, str):
        try:
            response = json.loads(response)
        except json.JSONDecodeError:
            return response  # Return the raw response if it's not a valid JSON string
    
    print(response)

    if "result" not in response:
        raise ValueError("Response does not contain 'result' key")

    return jsonify({"response": response["result"]})


app.route("/", methods=["GET", "POST"])
def index():
    return jsonify({
        "message" : "Hello World"
    })

@app.route('/new_chat', methods=['GET'])
def new_chat():
    df = pd.read_excel("dataset.xlsx")
    try:
        db = create_vector_db(df)
        if db:
            return jsonify({
                "message" : "DB created successfully"
            })
        
    except Exception as e:
        return jsonify({
            "message" : str(e)
        })
    

app.run(host="0.0.0.0", port=5000, debug=True)