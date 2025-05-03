from evaluate_retriever import evaluate_retriever, print_evaluation_results
from vector import retriever as food_retriever
from general_data import retriever as general_retriever
import pandas as pd

def get_database_documents(db_path, collection_name):
    """
    Get all documents from a Chroma database
    """
    from langchain_chroma import Chroma
    from langchain_ollama import OllamaEmbeddings
    
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_store = Chroma(
        collection_name=collection_name,
        persist_directory=db_path,
        embedding_function=embeddings
    )
    
    # Get all documents from the database
    all_docs = vector_store.get()
    return all_docs

def check_data():
    food_docs = get_database_documents("./chroma_langchain_db", "food_data")
    general_docs = get_database_documents("./general_data_db", "general_data")

    print(f"Total documents in food_db: {len(food_docs)}")
    print(f"Total documents in general_db: {len(general_docs)}")
    
    print(food_docs)
    # print(general_docs)

if __name__ == "__main__":
    check_data()

