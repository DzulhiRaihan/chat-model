from pymongo import MongoClient
import pandas as pd

uri = "mongodb+srv://dzulhiraihan:vkmX4POHPXXznHjc@cluster0.erxol.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(uri)
db = client["diet_bot"]  # Ganti dengan nama database kamu
collection = db["food_data"]

# Mengambil semua data dari MongoDB
data = collection.find({})
df = pd.DataFrame(list(data))

kategori = {
    "AR": "Serealia", "AP": "Olahan Serealia", "BR": "Umbi Berpati", "BP": "Olahan Umbi Berpati",
    "CR": "Kacang-Kacangan", "CP": "Olahan Kacang-Kacangan", "DR": "Sayuran", "DP": "Olahan Sayuran",
    "ER": "Buah", "EP": "Olahan Buah", "FR": "Daging", "FP": "Olahan Daging", "GR": "Ikan, Kerang, Udang",
    "GP": "Olahan Ikan, Kerang, Udang", "JR": "Olahan Susu", "JP": "Olahan Susu", "HR": "Telur", "HR": "Olahan Telur",
    "MP": "Gula, Sirup, Makanan Manis","KP": "Olahan Lemak dan Minyak", "KR" : "Lemak dan Minyak",
    "NR": "Bumbu", "NP": "Bumbu Olahan", "QR": "Minuman"
}

# Fungsi untuk mendapatkan kategori dari kode
def get_category(code):
    prefix = ''.join([char for char in code if not char.isdigit()])  # Ambil huruf dari kode
    return kategori.get(prefix, "Tidak Diketahui") 

descriptions = []

for _, row in df.iterrows():
    kategori_bahan = get_category(row['Kode'])  # Ambil kategori berdasarkan kode
    text = (
        f" termasuk dalam kategori {kategori_bahan}. "
        f"Bahan ini per 100 gram-nya memiliki {row['Kalori']} kalori, {row['Protein']} gram protein, "
        f"{row['Lemak']} gram lemak, {row['Karbohidrat']} gram karbohidrat, dan {row['Serat']} gram serat."
    )
    descriptions.append(text)
    
df['description'] = descriptions
df.drop(columns=['_id','Kalori', 'Protein', 'Lemak', 'Karbohidrat', 'Serat'], inplace=True)

df.to_csv('Dataset/food_data.csv', index=False, encoding='utf-8')
print('DataFrame telah disimpan ke food_data.csv')