from decision import classify_input
from greetings import generate_greeting
from recomendation import ask_ai_recomendation
from consultation import ask_ai_consultation
from general_Information import ask_ai_general
from convert_to_json import convert_to_json

def main():
    while True:
        user_input = input("Masukkan input: ")
        if user_input == "q":
            break
        clasify_result = classify_input(user_input)
        if clasify_result == "greetings":
            generate_greeting(user_input, "Raihan")
        elif clasify_result == "recommendation" or clasify_result == "rekomendasi":
            question = ask_ai_recomendation("Raihan", "Jarang", 43, 170, 25, "Pria", user_input)
            question
            convert_to_json(question)
            print("\n\n")
            print(convert_to_json(question))
        elif clasify_result == "consultation" or clasify_result == "konsultasi":
            ask_ai_consultation("Raihan", "Jarang", 43, 170, 25, "Pria", user_input)
        elif clasify_result == "general":
            ask_ai_general(user_input)
            
if __name__ == "__main__":
    main()


