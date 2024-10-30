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
class ChatMessage(BaseModel):
    question: str
    user_message: str
    chat_history: List[dict]
    passage_text: str = None
    underlined_text: str = None

class Question(BaseModel):
    id: int
    text: str
    options: List[str]
    correct: int
    passage_ref: str = None
    underlined_text: str = None

class Passage(BaseModel):
    id: str
    text: str
    questions: List[Question]

# Sample data structure
passages = [
    Passage(
        id="passage1",
        text="""The city council's decision to implement a new recycling program [1]was met with 
        significant resistance from local business owners, who argued that the [2]additional costs 
        would be prohibitively expensive. However, environmental advocates [3]pointed towards 
        successful similar programs in neighboring cities as evidence of the potential benefits.""",
        questions=[
            Question(
                id=1,
                text="Which choice best maintains the formal tone of the passage?",
                options=[
                    "was met with",
                    "ran into",
                    "encountered",
                    "faced"
                ],
                correct=0,
                passage_ref="passage1",
                underlined_text="was met with"
            ),
            Question(
                id=2,
                text="Which choice most effectively emphasizes the business owners' concerns?",
                options=[
                    "additional costs",
                    "financial burden",
                    "monetary requirements",
                    "economic implications"
                ],
                correct=1,
                passage_ref="passage1",
                underlined_text="additional costs"
            )
        ]
    )
]

@app.get("/")
async def read_root():
    return FileResponse("static/questions.html")

@app.get("/api/questions")
async def get_questions():
    return questions

@app.get("/api/passages")
async def get_passages():
    return passages

async def get_llm_response(question: str, user_message: str, chat_history: List[dict], passage_text: str = None, underlined_text: str = None) -> str:
    """
    Enhanced LLM response function that includes passage context
    """
    try:
        context = f"""Passage: {passage_text}

Question about the underlined text "{underlined_text}":
{question}"""

        system_prompt = f"""You are a helpful tutor assisting with a reading comprehension question.
        
        {context if passage_text else f"Question: {question}"}
        
        Provide clear explanations that help the student understand the reasoning behind the answer.
        Focus on the specific context and how it relates to the question.
        If they ask for the direct answer, guide them to think through it instead."""

        messages = [{"role": "system", "content": system_prompt}]
        
        for msg in chat_history:
            messages.append({
                "role": "user" if msg["isUser"] else "assistant",
                "content": msg["message"]
            })
            
        messages.append({"role": "user", "content": user_message})

        response = client.chat.completions.create(
            model="gpt-4o-mini",
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