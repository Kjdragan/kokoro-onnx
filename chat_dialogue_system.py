import threading
from queue import Queue
from typing import Optional, Dict, List
import time
import keyboard
import json
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from unknown_dialogue_updated import DynamicDialogueSystem

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatResponse(BaseModel):
    message: ChatMessage
    audio_id: str

class VoiceChatSystem:
    def __init__(self):
        self.dialogue_system = DynamicDialogueSystem()
        self.message_queue = Queue()
        self.active_connections: List[WebSocket] = []
        self.interrupt_event = threading.Event()
        self.current_audio_id: Optional[str] = None
        
        # Start the dialogue system
        self.dialogue_system.start_playback()
        
        # Start keyboard listener for Ctrl+Space
        keyboard.on_press_key('space', self._handle_interrupt, suppress=True)
        
    def _handle_interrupt(self, event):
        if keyboard.is_pressed('ctrl'):
            print("Interrupt triggered!")
            self.interrupt_current_speech()

    def interrupt_current_speech(self):
        if self.current_audio_id:
            self.interrupt_event.set()
            self.dialogue_system.audio_queue.clear()
            self.dialogue_system.currently_playing.clear()
            # Notify frontend of interruption
            self._broadcast_to_clients({
                "type": "interrupt",
                "audio_id": self.current_audio_id
            })
            self.current_audio_id = None
            self.interrupt_event.clear()

    async def connect_client(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect_client(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def _broadcast_to_clients(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                continue

    async def process_message(self, message: ChatMessage) -> ChatResponse:
        if message.role == "user":
            # Generate unique ID for this audio response
            audio_id = f"audio_{int(time.time()*1000)}"
            self.current_audio_id = audio_id
            
            # Add to dialogue system with assistant voice
            self.dialogue_system.add_dialogue(
                "assistant",
                message.content,
                "af_bella"  # Using bella voice for assistant
            )
            
            # Create response object
            response = ChatResponse(
                message=ChatMessage(
                    role="assistant",
                    content=message.content
                ),
                audio_id=audio_id
            )
            
            # Broadcast audio status to all clients
            await self._broadcast_to_clients({
                "type": "audio_start",
                "audio_id": audio_id
            })
            
            return response

    def cleanup(self):
        self.dialogue_system.stop_playback()

# FastAPI application
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize chat system
chat_system = VoiceChatSystem()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await chat_system.connect_client(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            message = ChatMessage.parse_raw(data)
            response = await chat_system.process_message(message)
            await websocket.send_json(response.dict())
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        chat_system.disconnect_client(websocket)

@app.post("/chat")
async def chat_endpoint(message: ChatMessage) -> ChatResponse:
    return await chat_system.process_message(message)

@app.post("/interrupt")
async def interrupt_endpoint():
    chat_system.interrupt_current_speech()
    return {"status": "success"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
