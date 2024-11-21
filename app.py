from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

# Load the model and vectorizer
with open("gradient_boosting_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

@app.route('/')
def home():
    return render_template('index.html')  # Renders an HTML form

@app.route('/predict', methods=['POST'])
def predict():
    # Get the news content from the form
    news_content = request.form['news']
    
    # Transform the input text using the vectorizer
    transformed_input = vectorizer.transform([news_content])
    
    # Make a prediction
    prediction = model.predict(transformed_input)[0]
    
    # Return the result
    result = "Fake News" if prediction == 1 else "Real News"
    return jsonify({'prediction': result})

if __name__ == "__main__":
    app.run(debug=True)
