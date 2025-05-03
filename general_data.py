from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import shutil 
import os

embeddings = OllamaEmbeddings(model="nomic-embed-text")

try:
    # Read the txt file
    with open('Dataset/general_infomration.txt', 'r', encoding='utf-8') as file:
        data = file.read()
except FileNotFoundError:
    print("Error: File not found!")
    exit(1)
except Exception as e:
    print(f"Error reading file: {str(e)}")
    exit(1)

# Clean up existing database if it exists
db_location = "./general_data_db"
if os.path.exists(db_location):
    shutil.rmtree(db_location)

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,  # Ukuran chunk dalam karakter
    chunk_overlap=100,  # Overlap antar chunk untuk menjaga konteks
    length_function=len,
    separators=["\n\n", "\n", " ", ""]  # Prioritas pemisah
)

# Split the text into chunks
chunks = text_splitter.split_text(data)

# Convert chunks to documents
documents = []
for i, chunk in enumerate(chunks):
    documents.append(Document(
        page_content=chunk,
        metadata={"id": i, "source": "general_information"}
    ))

# Create and store documents in vector database
vector_store = Chroma(
    collection_name="general_data",
    persist_directory=db_location,
    embedding_function=embeddings
)

# Add documents to the vector store
vector_store.add_documents(documents)

retriever = vector_store.as_retriever(
    search_kwargs = {"k" : 5}
)

# print("\n=== Mencoba Retriever ===")
# query = "Faktor-Faktor yang Memengaruhi Berat Badan"
# results = retriever.invoke(query)

# print(f"\nHasil pencarian untuk query: '{query}'")
# for doc in results:
#     print(f"\nKonten: {doc.page_content}")
#     print(f"Metadata: {doc.metadata}")

all_documents = vector_store.get()

print("\nIsi dokumen yang tersimpan:")
for i in range(len(all_documents['ids'])):
    print(f"\nID: {all_documents['ids'][i]}")
    print(f"Konten: {all_documents['documents'][i]}")
    print(f"Metadata: {all_documents['metadatas'][i]}")

