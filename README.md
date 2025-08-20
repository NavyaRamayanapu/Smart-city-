# Sustainable Smart City Chatbot 🌱🤖

This is a FastAPI-based chatbot project designed to assist with smart city concepts. It uses a transformer-based model under the hood and serves a local HTML UI to interact with.

---

## 🚀 Quick Start

### 1. Clone or Download the Project

```bash
git clone https://github.com/your-username/smart-city-chatbot.git
cd smart-city-chatbot
```

Or, if you're receiving this as a ZIP:

- Extract the folder
- Open the folder in a terminal (CMD or PowerShell)

---

### 2. Set Up Python Virtual Environment (Recommended)

Create a virtual environment:

```bash
python -m venv venv
```

Activate it:

- On **Windows**:

  ```bash
  .\venv\Scripts\activate
  ```

- On **Mac/Linux**:

  ```bash
  source venv/bin/activate
  ```

---

### 3. Install Required Packages

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

If you face errors with `torch`, manually install:

```bash
pip install torch==2.7.1
```

---

### 4. Run the FastAPI Server

Start the backend API:

```bash
python -m uvicorn main:app --reload
```

By default, the server runs at:
📍 [http://127.0.0.1:8000](http://127.0.0.1:8000)

---

### 5. Access the Chat UI

Open `index.html` from the `static/` folder in your browser:
📂 `static/index.html`

(You can double-click it or drag it into Chrome)

Make sure the server is running before chatting.

---

## 💡 Notes

- The chatbot model may take time to load the first time (downloading).
- Avoid closing the terminal running the FastAPI server.
- If the chatbot doesn’t respond, check terminal logs for model loading or memory issues.

---

## 🧪 Example API Test (Optional)

You can test the API directly:

```bash
curl -X POST http://127.0.0.1:8000/chat \
-H "Content-Type: application/json" \
-d '{"message": "Hello", "max_tokens": 50, "temperature": 0.7}'
```

---

## 📦 Files

- `main.py` – FastAPI backend
- `static/index.html` – Frontend UI
- `requirements.txt` – Python dependencies
- `README.md` – Setup guide (this file)

---

## ✨ Credits

Built with ❤️ using FastAPI and Hugging Face Transformers.
