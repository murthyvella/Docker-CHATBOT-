from flask import Flask, render_template, request, jsonify
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

app = Flask(__name__)

# Load model & tokenizer
MODEL_NAME = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.eval()

chat_history_ids = None # conversation memory

# Custom fixed responses for important keywords
special_responses = {
    "python": "Python 🐍 is an amazing programming language — and it’s the one I’m built with!",
    "java": "Java ☕ is powerful for backend and enterprise applications.",
    "ai": "AI 🤖 is the future — I’m living proof of it!",
    "hru": "I’m doing great, thanks for asking 😃 How about you?",
    "loose": "Did you mean *lose*? 😉",
    "bye": "Goodbye! 👋 Come back soon."
}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def chatbot_response():
    global chat_history_ids
    user_text = request.json["message"].lower().strip()

    # 1) Check for special fixed replies first
    if user_text in special_responses:
        return jsonify({"reply": special_responses[user_text]})

    # 2) Otherwise, use AI model
    new_user_input_ids = tokenizer.encode(user_text + tokenizer.eos_token, return_tensors="pt")

    if chat_history_ids is None:
        input_ids = new_user_input_ids
    else:
        input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1)

    output_ids = model.generate(
        input_ids,
        max_new_tokens=120,
        do_sample=True,
        top_p=0.92,
        top_k=50,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    reply_ids = output_ids[:, input_ids.shape[-1]:]
    reply = tokenizer.decode(reply_ids[0], skip_special_tokens=True).strip()

    chat_history_ids = output_ids

    return jsonify({"reply": reply})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
