
import shutil
import asyncio
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from commentator import Commentator
# Import tools from other modules
from cricket_server import get_match_context_at_time
from llm import call_llm

class ChatRequest(BaseModel):
    question: str
    timestamp: float

app = FastAPI(title="Cricket Commentary AI")

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(".").resolve()
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# Global State (Single user simplified)
processing_state = {
    "status": "idle", # idle, processing, completed, error
    "message": "Ready to start",
    "step": 0,
    "total_steps": 7,
    "result_path": None,
    "logs": []
}

commentator = Commentator(BASE_DIR)

def update_progress(msg):
    global processing_state
    processing_state["message"] = msg
    processing_state["logs"].append(msg)
    # Simple step increment heuristic
    if processing_state["step"] < processing_state["total_steps"]:
        processing_state["step"] += 1

def run_pipeline_task(video_path: Path):
    global processing_state
    processing_state["status"] = "processing"
    processing_state["step"] = 0
    processing_state["logs"] = []
    
    try:
        final_path = commentator.process_video(
            video_path, 
            has_scorecard=processing_state.get("has_scorecard", True),
            update_callback=update_progress
        )
        if final_path:
            processing_state["status"] = "completed"
            processing_state["result_path"] = final_path
            processing_state["message"] = "Processing Complete!"
            processing_state["step"] = processing_state["total_steps"]
        else:
            processing_state["status"] = "error"
            processing_state["message"] = "Pipeline failed at TTS stage."
    except Exception as e:
        processing_state["status"] = "error"
        processing_state["message"] = f"Error: {str(e)}"
        print(f"Pipeline Error: {e}")

@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    global processing_state
    
    # We will get 'has_scorecard' from a separate call or wait, the user uploads first then processes.
    # The /process call should set the flags.
    
    # Reset state
    processing_state = {
        "status": "idle", 
        "message": "File uploaded", 
        "step": 0, 
        "total_steps": 7,
        "result_path": None, 
        "logs": []
    }
    
    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    return {"filename": file.filename, "path": str(file_path)}

@app.post("/process")
async def process_video(filename: str, background_tasks: BackgroundTasks, has_scorecard: bool = True):
    video_path = UPLOAD_DIR / filename
    if not video_path.exists():
        return JSONResponse(status_code=404, content={"message": "File not found"})
    
    # Store options in state for the worker to pick up
    processing_state["has_scorecard"] = has_scorecard
    
    background_tasks.add_task(run_pipeline_task, video_path)
    return {"message": "Processing started"}

@app.get("/status")
async def get_status():
    return processing_state

@app.get("/result")
async def get_result():
    # Robust check: If file exists on disk, serve it (handles server restarts)
    output_path = BASE_DIR / "final_output.mp4"
    if output_path.exists():
        return FileResponse(output_path, media_type="video/mp4", filename="cricket_commentary.mp4")
    
    if processing_state["status"] == "completed" and processing_state["result_path"]:
        return FileResponse(processing_state["result_path"])
        
    return JSONResponse(status_code=404, content={"message": "Result not ready"})

@app.post("/chat")
async def chat_with_analyst(req: ChatRequest):
    # 1. Get Context from the MCP tool
    context_json = get_match_context_at_time(req.timestamp)
    
    # 2. Build Prompt
    prompt = (
        f"You are an expert Cricket Analyst explaining the match to a viewer.\n"
        f"CONTEXT (Events happening around {req.timestamp}s):\n{context_json}\n\n"
        f"USER QUESTION: {req.question}\n\n"
        f"TASK: Answer the user's question based strictly on the provided context. "
        f"Keep it short (max 2 sentences) and helpful. If the context doesn't have the answer, say you are analyzing live data."
    )
    
    # 3. Call LLM (We reuse the Jina client from pipeline10)
    answer = call_llm(prompt)
    
    # Cleanup any error tags if they appear
    if "[LLM ERROR]" in answer:
        answer = "I'm having trouble connecting to the analysis server. Please try again."
        
    return {"answer": answer}

# Serve Static
app.mount("/", StaticFiles(directory="static", html=True), name="static")
