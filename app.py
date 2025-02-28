from flask import Flask, render_template, request, jsonify
import onnxruntime
import numpy as np
from sentence_transformers import SentenceTransformer
import json
import os
from dotenv import load_dotenv
from groq import Groq
import re

app = Flask(__name__)

def extract_json_from_text(text):
    pattern = r'```json\s*(\{[\s\S]*?\})\s*```'
    match = re.search(pattern, text, re.DOTALL)
    
    if match:
        json_str = match.group(1)
    else:
        # Fallback: try to extract any JSON object from the text
        match = re.search(r'(\{[\s\S]*\})', text)
        if match:
            json_str = match.group(1)
        else:
            return None

    try:
        data = json.loads(json_str)
        # Optionally verify the expected structure:
        if "response" in data and isinstance(data["response"], list):
            return data
        else:
            print("JSON does not match the expected format (missing 'response' key or it is not a list).")
            return None
    except json.JSONDecodeError as e:
        print("Error decoding JSON:", e)
        return None

# Load ONNX model and embedding model
session = onnxruntime.InferenceSession("model/embedding-NN-model.onnx")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load environment variables
load_dotenv("utils/.env", override=True)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize LLM model
class groq_model:
    def __init__(self, model="llama-3.1-8b-instant"):
        self.client = Groq(api_key=GROQ_API_KEY)
        self.model_name = model

    def get_response(self, query):
        chat_completion = self.client.chat.completions.create(
            messages=[
                {"role": "user", "content": query},
            ],
            model=self.model_name,
        )
        return chat_completion.choices[0].message.content

llm_model = groq_model()

# Sentiment classification using ONNX
class_mapping = {0: "Negative", 1: "Neutral", 2: "Positive"}

def classify_text(text):
    text_embedding = embedding_model.encode([text])
    input_data = np.array(text_embedding, dtype=np.float32)
    inputs = {session.get_inputs()[0].name: input_data}
    outputs = session.run(None, inputs)
    predicted_index = np.argmax(outputs[0], axis=1)[0]
    return class_mapping[predicted_index]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.json
    text = data.get("text", "")
    method = data.get("method", "nn")

    if method == "llm":
        with open("utils/prompts/detect-classification.txt", "r") as f:
            prompt = f.read()
        with open("utils/sentiments.json", "r") as f:
            sentiment_classes = json.load(f)
        sentiments_str = "\n".join([f"{i+1}. {s}" for i, s in enumerate(sentiment_classes)])
        query = prompt.replace("$#emailbody#$", text).replace("$#sentiment-categories#$", sentiments_str)
        response = llm_model.get_response(query)
        result_class = extract_json_from_text(response)['response']
        highest_category = max(result_class, key=lambda x: x["probability"])

        result = highest_category["category_name"]

        return jsonify({"prediction": result})
    else:
        prediction = classify_text(text)
        return jsonify({"prediction": prediction})

if __name__ == "__main__":
    app.run(debug=True)