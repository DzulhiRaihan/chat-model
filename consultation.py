from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from calculate import calculate_bmi, calculate_bmr, calculate_tdee

model = OllamaLLM(model="llama3.2:3b")

template = """
Anda adalah seorang ahli gizi profesional yang bertugas menangani user yang ingin berkonsultasi mengenai kesehatan dan gizi (terutama diet)
Anda cukup memberikan informasi yang relevan berdasarkan pertanyaan dari user dan data pengguna yang diberikan.
Berikan jawaban yang singkat dan jelas

Data Pengguna: 
kalori_harian : {tdee}
BMI : {BMI}
profil : {profile}

Pertanyaan pengguna: {input}


"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

def ask_ai_consultation(name, activity, weight, height, age, gender, question):
    bmi = calculate_bmi(weight, height)
    bmr = calculate_bmr(weight, height, age, gender)
    tdee = calculate_tdee(activity, bmr)
    profile = f"{gender}, {age} tahun, {height}cm, {weight}kg, {activity}, {name}"
   
    prompt_value = prompt.format_messages(
        input=question,
        tdee=tdee,
        BMI=bmi,
        profile=profile
    )
    
    response = ""
    for chunk in model.stream(prompt_value):
        print(chunk, end="", flush=True)
        response += chunk
    print()
    return response
    
while True:
    question = input("Masukkan pertanyaan: ")
    if question == "q":
        break   
    ask_ai_consultation("Raihan", "Jarang", 43, 170, 25, "Pria", question)
