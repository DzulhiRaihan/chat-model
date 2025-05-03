from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

model = OllamaLLM(model="llama3.1:8b")

template = """
Tugas kamu adalah mengklasifikasikan input dari user ke dalam salah satu kategori berikut:

1.greetings: jika user menyapa seperti "halo", "hai", "selamat pagi", dll.
2.recommendation: jika user meminta saran atau rekomendasi, seperti makanan, aktivitas, atau jadwal.
3.general: jika user bertanya atau memberikan pernyataan umum yang tidak masuk kategori lain.
4.consultation: jika user meminta saran spesifik terkait kondisi pribadi, seperti "saya punya diabetes, makanan apa yang cocok?" atau "berat badan saya berlebih, apa yang harus saya lakukan?, apakah berat badan saya ideal/normal?, apakah berat badan saya normal?".

Berikan hanya nama kategorinya yaitu greetings, recommendation, general, atau consultation saja tanpa penjelasan tambahan.

Input: "{user_input}"
Output:

"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

def classify_input(user_input):
    output = chain.invoke({"user_input": user_input})
    return output

# while True:
#     user_input = input("Masukkan input: ")
#     if user_input == "q":
#         break
#     print(classify_input(user_input))
