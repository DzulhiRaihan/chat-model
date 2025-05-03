from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from calculate import calculate_bmi, calculate_bmr, calculate_tdee
from vector import retriever

# Initialize the model
model = OllamaLLM(model="llama3.1:8b")

template = """
Kamu adalah asisten chatbot yang cerdas dan ramah, bertugas membantu pengguna dalam hal informasi diet dan memberikan rekomendasi makanan yang tepat, akurat, dan disesuaikan dengan 'Data Pengguna'.

Ikuti petunjuk ini dengan sangat hati-hati:

- Semua jawaban dan rekomendasi makanan **hanya boleh** berdasarkan data makanan (`context`) yang diberikan.
- **Jangan** menggunakan pengetahuan umum atau tambahan di luar `context`.
- Jika makanan yang dibutuhkan tidak tersedia di `context`, berikan jawaban bahwa rekomendasi terbatas hanya pada data yang tersedia.
- Variasikan pilihan makanan, jangan mengulang makanan yang sama di waktu makan yang berbeda.
- Jawaban harus singkat, padat, dan sedikit penjelasan.

Ketentuan berdasarkan BMI pengguna:
- Jika BMI > 25 (obesitas):
  - Sarankan untuk menurunkan kebutuhan kalori secara bertahap.
  - Pilih makanan rendah kalori, tinggi serat, dan kaya nutrisi dari `context`.
- Jika BMI < 18.5 (underweight):
  - Sarankan untuk meningkatkan kebutuhan kalori.
  - Pilih makanan padat gizi, tinggi protein, dan kalori sehat dari `context`.
- Jika 18.5 <= BMI < 25 (normal):
  - Sarankan untuk mempertahankan pola makan seimbang dengan makanan dari `context`.

Selalu sertakan jadwal makan lengkap:
- **Sarapan** (07:00–09:00)
- **Camilan pagi** (sekitar 10:00, opsional)
- **Makan siang** (12:00–13:00)
- **Camilan sore** (sekitar 15:00, opsional)
- **Makan malam** (18:00–19:00)

contoh output:
Berikut adalah rekomendasi makanan untuk Anda:

**Sarapan (07:00-09:00)**

* Bayam rebus (100g) - 30 kalori, 1.3 gram protein
* Kacang belimbing rebus (100g) - 204 kalori, 16.9 gram protein

**Makan Siang (12:00-13:00)**

* Bekicot dendeng mentah (100g) - 441 kalori, 48.7 gram protein
* Beras tapai (100g) - 99 kalori, 1.7 gram protein

**Makan Malam (18:00-19:00)**

* Bayam kukus (100g) - 30 kalori, 1.3 gram protein
* Kacang belimbing rebus (100g) - 204 kalori, 16.9 gram protein

Data pengguna: 
- Kalori harian: {tdee}
- BMI: {BMI}
- Profil: {profile}

Context makanan tersedia:
{food_data}

Pertanyaan pengguna: {input}


"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

def ask_ai_recomendation(name, activity, weight, height, age, gender, question):
    bmi = calculate_bmi(weight, height)
    bmr = calculate_bmr(weight, height, age, gender)
    tdee = calculate_tdee(activity, bmr)
    profile = f"{gender}, {age} tahun, {height}cm, {weight}kg, {activity}, {name}"
    
    food_data = retriever.invoke(question)
    
    # Create the prompt
    prompt_value = prompt.format_messages(
        input=question,
        tdee=tdee,
        BMI=bmi,
        profile=profile,
        food_data=food_data
    )
    
    # Stream the response
    response = ""
    for chunk in model.stream(prompt_value):
        print(chunk, end="", flush=True)
        response += chunk
    print()  # Add newline at the end
    return response

# Example usage:
# while True:
#     question = input("Masukkan pertanyaan: ")
#     if question == "q":
#         break
#     ask_ai_recomendation("Raihan", "Normal", 77, 165, 25, "Pria", question)