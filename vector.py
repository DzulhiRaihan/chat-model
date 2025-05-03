from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import pandas as pd
import os
import shutil  # untuk menghapus direktori
import atexit  # untuk menangani cleanup saat program berakhir

# Baca data CSV
df = pd.read_csv('Dataset/food_data.csv')

# Inisialisasi embeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Hapus direktori database lama jika ada
db_location = "./chroma_langchain_db"
if os.path.exists(db_location):
    try:
        shutil.rmtree(db_location)
    except PermissionError:
        print("Warning: Could not remove old database directory. It may be in use by another process.")

documents = []

for i, row in df.iterrows():
    document = Document(
        page_content=row["Nama Bahan"] + " " + row["description"],
        metadata={"kode": row["Kode"], "id": str(i)}
    )
    documents.append(document)

vector_store = Chroma(
    collection_name="food_data",
    persist_directory=db_location,
    embedding_function=embeddings
)

vector_store.add_documents(documents)

retriever = vector_store.as_retriever(
    search_kwargs = {"k" : 5}
)

# Fungsi untuk cleanup
def cleanup():
    try:
        if hasattr(vector_store, '_client'):
            vector_store._client.close()
    except Exception as e:
        print(f"Error during cleanup: {e}")

# Register cleanup function
atexit.register(cleanup)

# print("\n=== Mencoba Retriever ===")
# query = "Tumis bandeng, masakan"
# results = retriever.invoke(query)

# print(f"\nHasil pencarian untuk query: '{query}'")
# for doc in results:
#     print(f"\nKonten: {doc.page_content}")
#     print(f"Metadata: {doc.metadata}")




# print("Mengambil data yang tersimpan...")
all_documents = vector_store.get()
# # print(f"\nJumlah dokumen yang tersimpan: {len(all_documents['ids'])}")
# print(all_documents)

print("\nIsi dokumen yang tersimpan:")
for i in range(len(all_documents['ids'])):
    print(f"\nID: {all_documents['ids'][i]}")
    print(f"Konten: {all_documents['documents'][i]}")
    print(f"Metadata: {all_documents['metadatas'][i]}")

# print("\n=== Contoh Pencarian Similarity ===")
# query = "Beras"
# similar_docs = vector_store.similarity_search(query)

# print(f"\nHasil pencarian untuk query: '{query}'")
# for doc in similar_docs:
#     print(f"\nKonten: {doc.page_content}")
#     print(f"Metadata: {doc.metadata}")

