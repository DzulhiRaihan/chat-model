from langchain_ollama import OllamaEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_core.documents import Document
import pandas as pd
from pymongo import MongoClient
import atexit
from get_food_data import df
# Baca data CSV
# df = pd.read_csv('Dataset/food_data.csv')

# Inisialisasi embeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Inisialisasi koneksi MongoDB
mongo_uri = "mongodb+srv://FrieskyAst:DzulhiRaihan@cluster0.isccjre.mongodb.net/"
client = MongoClient(mongo_uri)
db_name = "langchain_food_db"
collection_name = "food_data_vectors"
collection = client[db_name][collection_name]

documents = []
for i, row in df.iterrows():
    # Cek apakah dokumen dengan ID ini sudah ada
    existing_doc = collection.find_one({"metadata.id": row["_id"]})
    if not existing_doc:
        document = Document(
            page_content=row["Nama Bahan"] + " " + (row["description"]),
            metadata={"kode": row["Kode"]}
        )
        documents.append(document)

# Inisialisasi vector store
vector_store = MongoDBAtlasVectorSearch(
    collection=collection,
    embedding=embeddings,
)

# Tambahkan hanya dokumen baru
if documents:
    vector_store.add_documents(documents)
    print(f"\n{len(documents)} dokumen baru berhasil ditambahkan.")
else:
    print("\nTidak ada dokumen baru yang perlu ditambahkan.")

# Buat retriever
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 6}
)


# Cleanup koneksi MongoDB saat keluar
def cleanup():
    try:
        client.close()
    except Exception as e:
        print(f"Error during MongoDB cleanup: {e}")

atexit.register(cleanup)

# Tampilkan semua dokumen dari MongoDB
print("\nIsi dokumen yang tersimpan di MongoDB:")
all_docs = client[db_name][collection_name].find().limit(5)
for doc in all_docs:
    print(f"\nID: {doc.get('_id')}")
    print(f"Konten: {doc.get('text')}")
    print(f"Metadata: {doc.get('kode')}")

# print("Mengambil data yang tersimpan...")
all_documents = vector_store.get()
# # print(f"\nJumlah dokumen yang tersimpan: {len(all_documents['ids'])}")
# print(all_documents)


# print("\n=== Contoh Pencarian Similarity ===")
# query = "Beras"
# similar_docs = vector_store.similarity_search(query)

# print(f"\nHasil pencarian untuk query: '{query}'")
# for doc in similar_docs:
#     print(f"\nKonten: {doc.page_content}")
#     print(f"Metadata: {doc.metadata}")

