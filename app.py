from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
import fitz  # PyMuPDF
from dotenv import load_dotenv
import os

# 🔹 Load environment variables
load_dotenv()

# 🔹 Get API key
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("❌ OPENAI_API_KEY not found. Check your .env file")

# 🔹 Initialize OpenAI client
client = OpenAI(api_key=api_key)

# 🔹 Flask app setup
app = Flask(__name__)
CORS(app, origins="http://localhost:3000")


# 🔹 Common summary function
def get_summary(prompt: str) -> str:
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",   # ✅ fast + cheap + stable
            messages=[
                {"role": "system", "content": "Summarize clearly in simple language."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.5
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"❌ OpenAI Error: {str(e)}"


# 🔹 TEXT SUMMARIZATION
@app.route('/summarize/text', methods=['POST'])
def summarize_text():
    try:
        data = request.get_json()

        if not data or 'text' not in data:
            return jsonify({"error": "No text provided"}), 400

        text = data['text'].strip()

        if not text:
            return jsonify({"error": "Empty text"}), 400

        prompt = f"Summarize this text:\n{text}"
        summary = get_summary(prompt)

        return jsonify({"summary": summary})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# 🔹 PDF SUMMARIZATION
@app.route('/summarize/pdf', methods=['POST'])
def summarize_pdf():
    try:
        file = request.files.get('file')

        if not file:
            return jsonify({"error": "No file uploaded"}), 400

        doc = fitz.open(stream=file.read(), filetype="pdf")

        text = ""
        for page in doc:
            text += page.get_text()

        if not text.strip():
            return jsonify({"error": "PDF has no readable text"}), 400

        prompt = f"Summarize this PDF content:\n{text[:4000]}"  # 🔥 limit input
        summary = get_summary(prompt)

        return jsonify({"summary": summary})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# 🔹 HEALTH CHECK
@app.route('/')
def home():
    return jsonify({"status": "✅ API is running"})


# 🔹 RUN SERVER
if __name__ == '__main__':
    app.run(debug=True)
