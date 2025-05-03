from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

model = OllamaLLM(model="llama3.1:8b")

template = """

Anda adalah asisten yang ahli dalam mengonversi teks menjadi format JSON. Berikut adalah sebuah input berisi daftar makanan beserta jadwal makannya.

Tugas Anda:
Ekstrak hanya informasi makanan dan jadwal makan dari input.
Gabungkan data tersebut menjadi sebuah JSON yang terstruktur, dengan masing-masing makanan terhubung pada jadwal makannya dan kode makanan yang sesuai dengan nama makanan yang ada di database.
Buat kode makanan terpisah dari nama makanan.
Pastikan JSON valid dan hanya memuat informasi yang relevan dari input.
Berikan output berupa JSON saja, tanpa penjelasan tambahan.

contoh output:
[
  {{
    "kode": "A001",
    "nama makanan": "Sayur Bayam",
    "waktu": "07:00-09:00",
    "keterangan" : "Sarapan"
  }},
  {{
    "kode": "A002",
    "nama makanan": "Kacang Belimbing",
    "waktu": "12:00-13:00",
    "keterangan" : "Makan Siang"
  }},
  {{
    "kode": "A003",
    "nama makanan": "Bekicot Dendeng Mentah",
    "waktu": "18:00-19:00",
    "keterangan" : "Makan Malam"
  }}
]

catatan : cukup berikan ouput seperti contoh output diatas, tidak perlu menambahkan keterangan lainnya.

Input:
{input}

Data Makanan:
{food_data}

"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

def convert_to_json(input):
    food_data = retriever.invoke(input)
    output = chain.invoke({"input": input, "food_data": food_data})
    return output


# input = """
# Berikut adalah rekomendasi makanan untuk Anda:

# **Sarapan (07:00-09:00)**

# * Bayam rebus (100g) - 30 kalori, 1.3 gram protein
# * Kacang belimbing rebus (100g) - 204 kalori, 16.9 gram protein

# **Makan Siang (12:00-13:00)**

# * Bekicot dendeng mentah (100g) - 441 kalori, 48.7 gram protein
# * Beras tapai (100g) - 99 kalori, 1.7 gram protein

# **Makan Malam (18:00-19:00)**

# * Bayam kukus (100g) - 30 kalori, 1.3 gram protein
# * Kacang belimbing rebus (100g) - 204 kalori, 16.9 gram protein

# **Camilan (10:00 atau 15:00)**

# * (Tidak diperlukan)

# Perlu diingat bahwa Anda memiliki BMI 28.282828282828287 dan kebutuhan kalori harian adalah 2752.645. Oleh karena itu, disarankan untuk mengurangi kebutuhan kalori secara bertahap dan memilih makanan yang rendah kalori, tinggi serat, dan kaya nutrisi.

# """

# print(convert_to_json(input))

# while True:
#     user_input = input("Masukkan input: ")
#     if user_input == "q":
#         break
#     print(convert_to_json(user_input))







