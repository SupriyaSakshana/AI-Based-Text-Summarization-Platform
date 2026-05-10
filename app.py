from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os
import fitz  # PyMuPDF

from google import genai

# 🔹 Load environment variables
load_dotenv()

# 🔹 Get API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("❌ GEMINI_API_KEY not found. Check your .env file")

# 🔹 Initialize Gemini client
client = genai.Client(api_key=GEMINI_API_KEY)


# 🔹 Flask app
app = Flask(__name__)
CORS(app)  # Allow all origins

# 🔹 COMMON SUMMARY FUNCTION
def get_summary(text: str) -> str:
    print("inside get summary")
    
    try:
        prompt = f"Summarize the following text in simple and clear language:\n\n{text}"
        print("after prompt")
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        # ✅ safer response handling
        if hasattr(response, "text") and response.text:
            return response.text.strip()
        else:
            return "❌ No response from Gemini"

    except Exception as e:
        print("❌ Gemini Error:", e)
        return f"❌ Gemini Error: {str(e)}"


# 🔹 TEXT SUMMARIZATION
@app.route("/summarize/text", methods=["POST"])
def summarize_text():
    
    try:
        data = request.get_json()

        if not data or "text" not in data:
            return jsonify({"error": "No text provided"}), 400

        text = data["text"].strip()

        if not text:
            return jsonify({"error": "Empty text"}), 400

        print("📥 Text request received")

        summary = get_summary(text)

        return jsonify({"summary": summary})

    except Exception as e:
        print("❌ Error:", e)
        return jsonify({"error": str(e)}), 500


# 🔹 PDF SUMMARIZATION
@app.route("/summarize/pdf", methods=["POST"])
def summarize_pdf():
    try:
        file = request.files.get("file")

        if not file:
            return jsonify({"error": "No file uploaded"}), 400

        print("📥 PDF request received")

        # Read PDF
        doc = fitz.open(stream=file.read(), filetype="pdf")

        text = ""
        for page in doc:
            text += page.get_text()

        if not text.strip():
            return jsonify({"error": "PDF has no readable text"}), 400

        # 🔥 Limit text (important for API)
        text = text[:8000]

        summary = get_summary(text)

        return jsonify({"summary": summary})

    except Exception as e:
        print("❌ PDF Error:", e)
        return jsonify({"error": str(e)}), 500


# 🔹 HEALTH CHECK
@app.route("/")
def home():
    return jsonify({
        "status": "✅ API running",
        "message": "Gemini backend working"
    })


# 🔹 RUN SERVER
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
