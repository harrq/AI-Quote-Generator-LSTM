from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from tensorflow.keras.models import load_model
import pickle
import numpy as np
import uvicorn

app = FastAPI()

# ==========================
# Load Model
# ==========================

model = load_model("lstm_model.h5")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("max_len.pkl", "rb") as f:
    max_len = pickle.load(f)


# ==========================
# Predict Next Word
# ==========================

def predict_next_word(seed_text):

    token_list = tokenizer.texts_to_sequences([seed_text])[0]

    if not token_list:
        return ""

    token_list = token_list[-max_len:]

    token_list = np.pad(
        token_list,
        (max_len - len(token_list), 0),
        mode="constant"
    )

    token_list = np.reshape(token_list, (1, max_len))

    predicted = model.predict(token_list, verbose=0)

    predicted_id = np.argmax(predicted, axis=-1)[0]

    for word, index in tokenizer.word_index.items():

        if index == predicted_id:
            return word

    return ""


# ==========================
# Generate Quote
# ==========================

def generate_quote(seed_text, num_words=30):

    generated_text = seed_text

    for _ in range(num_words):

        next_word = predict_next_word(generated_text)

        if not next_word:
            break

        generated_text += " " + next_word

    return generated_text


# ==========================
# HOME PAGE
# ==========================

@app.get("/", response_class=HTMLResponse)
async def home():

    return """
    <!DOCTYPE html>

    <html>

    <head>

        <title>AI Quote Generator</title>

        <style>

            body {
                font-family: Arial;
                background: #f5f5f5;

                display: flex;
                justify-content: center;
                align-items: center;

                min-height: 100vh;
            }

            .container {
                background: white;

                width: 500px;

                padding: 35px;

                border-radius: 12px;

                box-shadow: 0 5px 20px rgba(0,0,0,0.1);
            }

            h1 {
                text-align: center;
            }

            textarea {
                width: 100%;
                height: 100px;

                padding: 10px;

                font-size: 16px;

                border: 1px solid #ccc;

                border-radius: 8px;

                resize: none;
            }

            button {
                width: 100%;

                margin-top: 15px;

                padding: 12px;

                background: #4f46e5;

                color: white;

                border: none;

                border-radius: 8px;

                font-size: 16px;

                cursor: pointer;
            }

            button:hover {
                background: #4338ca;
            }

        </style>

    </head>


    <body>

        <div class="container">

            <h1>AI Quote Generator</h1>

            <form method="POST" action="/generate">

                <textarea
                    name="seed_text"
                    placeholder="Enter your starting text..."
                    required
                ></textarea>

                <button type="submit">
                    Generate Quote
                </button>

            </form>

        </div>

    </body>

    </html>
    """


# ==========================
# GENERATE
# ==========================

@app.post("/generate", response_class=HTMLResponse)
async def generate(seed_text: str = Form(...)):

    quote = generate_quote(seed_text)

    return f"""
    <!DOCTYPE html>

    <html>

    <head>

        <title>AI Quote Generator</title>

        <style>

            body {{
                font-family: Arial;
                background: #f5f5f5;

                display: flex;
                justify-content: center;
                align-items: center;

                min-height: 100vh;
            }}

            .container {{
                background: white;

                width: 500px;

                padding: 35px;

                border-radius: 12px;

                box-shadow: 0 5px 20px rgba(0,0,0,0.1);
            }}

            h1 {{
                text-align: center;
            }}

            textarea {{
                width: 100%;
                height: 100px;

                padding: 10px;

                font-size: 16px;

                border: 1px solid #ccc;

                border-radius: 8px;

                resize: none;
            }}

            button {{
                width: 100%;

                margin-top: 15px;

                padding: 12px;

                background: #4f46e5;

                color: white;

                border: none;

                border-radius: 8px;

                font-size: 16px;

                cursor: pointer;
            }}

            .result {{
                margin-top: 25px;

                padding: 20px;

                background: #f3f4f6;

                border-radius: 8px;

                line-height: 1.6;
            }}

        </style>

    </head>


    <body>

        <div class="container">

            <h1>AI Quote Generator</h1>

            <form method="POST" action="/generate">

                <textarea
                    name="seed_text"
                    required
                >{seed_text}</textarea>

                <button type="submit">
                    Generate Quote
                </button>

            </form>


            <div class="result">

                <strong>Generated Quote:</strong>

                <p>
                    {quote}
                </p>

            </div>

        </div>

    </body>

    </html>
    """


# ==========================
# START SERVER
# ==========================

if __name__ == "__main__":

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000
    )

