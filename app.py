from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

with open("gradient_boosting_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

@app.route('/')
def home():
    return render_template('index.html')  

@app.route('/predict', methods=['POST'])
def predict():
    news_content = request.form['news']
    
    transformed_input = vectorizer.transform([news_content])
    
    prediction = model.predict(transformed_input)[0]
    
    result = "Fake News" if prediction == 1 else "Real News"
    return jsonify({'prediction': result})

if __name__ == "__main__":
    app.run(debug=True)
