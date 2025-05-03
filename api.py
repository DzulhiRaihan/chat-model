from flask import Flask, request, jsonify
from flask_cors import CORS
from decision import classify_input
from greetings import generate_greeting
from recomendation import ask_ai_recomendation
from consultation import ask_ai_consultation
from general_Information import ask_ai_general

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_input = data.get('message')
        user_name = data.get('name', 'User')
        bmi_category = data.get('bmi_category', 'Normal')
        weight = data.get('weight', 70)
        height = data.get('height', 170)
        age = data.get('age', 25)
        gender = data.get('gender', 'Pria')

        if not user_input:
            return jsonify({'error': 'Message is required'}), 400

        classify_result = classify_input(user_input)
        
        if classify_result == "greetings":
            response = generate_greeting(user_input, user_name)
        elif classify_result in ["recommendation", "rekomendasi"]:
            response = ask_ai_recomendation(user_name, bmi_category, weight, height, age, gender, user_input)
        elif classify_result in ["consultation", "konsultasi"]:
            response = ask_ai_consultation(user_name, bmi_category, weight, height, age, gender, user_input)
        elif classify_result == "general":
            response = ask_ai_general(user_input)
        else:
            response = "Maaf, saya tidak mengerti pertanyaan Anda."

        return jsonify({
            'status': 'success',
            'response': response,
            'type': classify_result
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 