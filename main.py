from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM 
import uvicorn
import os
from typing import Optional

# Initialize FastAPI app
app = FastAPI(title="Sustainable Smart City Assistant", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global variables for model and tokenizer
model = None
tokenizer = None

class ChatRequest(BaseModel):
    message: str
    max_tokens: Optional[int] = 150
    temperature: Optional[float] = 0.7

class ChatResponse(BaseModel):
    response: str
    status: str

# Initialize the IBM Granite model
def load_model():
    global model, tokenizer
    try:
        model_name = "ibm-granite/granite-3.3-2b-instruct"
        print(f"Loading model: {model_name}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        
        print("Model loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

# Generate response using the model
def generate_response(prompt: str, max_tokens: int = 150, temperature: float = 0.7):
    try:
        # Create a sustainable smart city context
        system_prompt = """You are a Sustainable Smart City Assistant. You help users with:
- Green energy solutions and renewable energy planning
- Waste management and recycling strategies
- Smart transportation and mobility solutions
- Urban planning for sustainability
- Environmental monitoring and air quality
- Water conservation and management
- Smart building and energy efficiency
- Sustainable agriculture and urban farming
- Climate change adaptation strategies
- Circular economy principles

Provide practical, actionable advice for creating more sustainable urban environments."""

        full_prompt = f"{system_prompt}\n\nUser: {prompt}\nAssistant:"
        
        # Tokenize input
        inputs = tokenizer.encode(full_prompt, return_tensors="pt")
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        # Decode response
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the assistant's response
        response = full_response.split("Assistant:")[-1].strip()
        
        return response
    except Exception as e:
        return f"Error generating response: {str(e)}"

@app.on_event("startup")
async def startup_event():
    """Load the model when the app starts"""
    success = load_model()
    if not success:
        print("Warning: Model failed to load. The app will start but responses may be limited.")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main HTML page"""
    try:
        with open("static/index.html", "r") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Please create static/index.html file</h1>")

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Handle chat requests"""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        response = generate_response(
            request.message, 
            request.max_tokens, 
            request.temperature
        )
        return ChatResponse(response=response, status="success")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "cuda_available": torch.cuda.is_available()
    }

if __name__ == "__main__":
    # Create static directory if it doesn't exist
    os.makedirs("static", exist_ok=True)
    
    # Run the app
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)