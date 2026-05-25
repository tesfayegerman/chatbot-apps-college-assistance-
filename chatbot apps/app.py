from flask import Flask, render_template, request, jsonify
import nltk, json, numpy as np, random, pickle
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')

app = Flask(__name__)
lemmatizer = WordNetLemmatizer()

# Load all files
model = load_model('chatbot_model.h5')
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
with open('intents.json') as f:
    intents = json.load(f)

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    return [lemmatizer.lemmatize(w.lower()) for w in sentence_words]

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    return np.array([1 if w in sentence_words else 0 for w in words])

def predict_class(sentence):
    bow = bag_of_words(sentence)
    result = model.predict(np.array([bow]), verbose=0)[0]
    results = [[i, r] for i, r in enumerate(result) if r > 0.25]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{'intent': classes[r[0]], 'probability': str(r[1])} for r in results]

def get_response(intents_list):
    if not intents_list:
        return "I'm sorry, I didn't understand that."
    tag = intents_list[0]['intent']
    for intent in intents['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    intents_list = predict_class(user_message)
    response = get_response(intents_list)
    return jsonify({'response': response})

if __name__ == '__main__':
    # app.run(debug=True)
    app.run(host='0.0.0.0', port=5000, debug=True)