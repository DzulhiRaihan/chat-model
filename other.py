from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from retriever.vector import retriever
model = OllamaLLM(model="llama3.2:3b")

template = """
Anda adalah seorang asisten dalam membantu mencari data dalam database yang tersedia 

Tugas Anda:
Mencari data berdasarkan permintaan user

data yang tersedia: {food_data}

input user: {user_input}

"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

def search_data(user_input):
    food_data = retriever.invoke(user_input)
    output = chain.invoke({"food_data": food_data, "user_input": user_input})
    return output

while True:
    user_input = input("Masukkan input: ")
    if user_input == "q":
        break
    print(search_data(user_input))





