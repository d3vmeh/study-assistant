from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List
import os
from dotenv import load_dotenv
from openai import OpenAI  # for OpenAI
import uvicorn  # Add this import

load_dotenv()  # Load environment variables from .env file

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize your LLM client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Data models
class Question(BaseModel):
    question: str
    options: List[str]
    correct: int

class ChatMessage(BaseModel):
    question: str
    user_message: str
    chat_history: List[dict]  # Add this to track conversation history

# Sample question bank
questions = [
    Question(
        question="What is the capital of France?",
        options=["London", "Paris", "Berlin", "Madrid"],
        correct=1
    ),
    Question(
        question="Which planet is known as the Red Planet?",
        options=["Venus", "Jupiter", "Mars", "Saturn"],
        correct=2
    )
]

@app.get("/")
async def read_root():
    return FileResponse("static/questions.html")

@app.get("/api/questions")
async def get_questions():
    return questions

async def get_llm_response(question: str, user_message: str, chat_history: List[dict]) -> str:
    """
    Get response from LLM with chat history context
    """
    try:
        system_prompt = f"""You are a helpful tutor assisting with the following question:
        {question}
        
        Provide a clear, concise explanation that helps the student understand the topic better.
        If they're asking for the direct answer, encourage them to think through it instead.
        
        Remember to maintain context from the previous conversation in this thread."""

        # Convert chat history to OpenAI message format
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add chat history
        for msg in chat_history:
            messages.append({
                "role": "user" if msg["isUser"] else "assistant",
                "content": msg["message"]
            })
            
        # Add current message
        messages.append({"role": "user", "content": user_message})

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        return response.choices[0].message.content

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM API Error: {str(e)}")

@app.post("/api/chat")
async def chat(message: ChatMessage):
    try:
        response = await get_llm_response(
            message.question, 
            message.user_message,
            message.chat_history
        )
        return {"response": response}
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Add this at the bottom of the file
if __name__ == "__main__":
    print("Starting server...")
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True  # This enables auto-reload when you make code changes
    )