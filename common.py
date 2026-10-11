from pathlib import Path
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

data_dir = Path("football_data")
persist_directory = "./chroma_db"
manifest_path = Path(".ingestion_manifest.json")
collection_name = "football_knowledge"

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_store = Chroma(
    collection_name=collection_name,
    persist_directory=persist_directory,
    embedding_function=embeddings,
)
