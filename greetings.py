from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

model = OllamaLLM(model="llama3.2:3b")

template = """
Kamu adalah seorang asisten yang bertugas untuk menyapa user dengn nama.

Berikan kata-kata yang baik untuk menyapa user berdasarkan input user dan tanyakan apakah ada yang bisa di bantu.

input user: {input}

Nama user: {name}
"""


prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

def generate_greeting(input, name):
    prompt_value = prompt.format_messages(
        input=input,
        name=name
    )
    response = ""
    for chunk in model.stream(prompt_value):
        print(chunk, end="", flush=True)   
        response += chunk 
    print()
    return response

# while True:
#     user_input = input("Masukkan input: ")
#     if user_input == "q":
#         break
#     generate_greeting(user_input, "Raihan")

