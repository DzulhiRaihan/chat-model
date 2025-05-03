from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from general_data import retriever
model = OllamaLLM(model="llama3.2:3b")

template = """
Anda adalah seorang ahli kesehatan yang bertugas untuk memberikan informasi tentang kesehatan dan gizi berdasarkan informasi yang tersedia pada database.

Pertanyaan: "{question}"
informasi: "{information}"

sesuaikan jawaban berdasarkan pertanyaan dari user
cukup berikan jawaban yang singkat dan padat
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

def ask_ai_general(question):
    information = retriever.invoke(question)
    prompt_value = prompt.format_messages(
        question=question,
        information=information
    )
    response = ""
    for chunk in model.stream(prompt_value):
        print(chunk, end="", flush=True)
        response += chunk
    print()
    return response

# while True:
#     question = input("Masukkan pertanyaan: ")
#     if question == "q":
#         break
#     ask_ai_general(question)