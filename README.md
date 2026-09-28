# QuoteGen AI — FastAPI LSTM UI

A polished FastAPI web interface for the supplied quote-generation LSTM.

## Files
- `app.py` — FastAPI API + LSTM generation
- `templates/index.html` — UI
- `static/style.css` — styling
- `lstm_model.h5` — trained model
- `tokenizer.pkl` — fitted tokenizer
- `max_len.pkl` — sequence length (347)
- `qoute_dataset.csv` — quote dataset

## Run on Windows

Use Python 3.11 or 3.12 for the smoothest TensorFlow setup.

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn app:app --reload
```

Open: http://127.0.0.1:8000

## API
POST `/api/generate`

Example JSON:
```json
{
  "seed": "Success begins when",
  "words": 30,
  "temperature": 0.8
}
```

GET `/api/health`

## Important
The tokenizer and model must be loaded with a TensorFlow/Keras environment compatible with the environment that created the `.h5` file. The supplied model reports Keras 3.13.2 and TensorFlow backend.
