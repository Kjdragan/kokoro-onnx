# Chat Application with Text-to-Speech

This is a real-time chat application that integrates text-to-speech capabilities using the ONNX-based TTS system.

## Features

- Real-time chat interface
- Low-latency text-to-speech for assistant responses
- Ctrl+Space to interrupt speech
- WebSocket-based real-time updates
- Modern web interface

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Start the server:
```bash
python chat_dialogue_system.py
```

3. Open your browser to `http://localhost:8000`

## System Architecture

The application consists of:

1. **Backend**
   - FastAPI server with WebSocket support
   - Integration with ONNX TTS system
   - Real-time audio generation and playback
   - Speech interruption handling

2. **Frontend**
   - Modern web interface using Tailwind CSS
   - Real-time WebSocket communication
   - Speech status indicators
   - Interrupt controls

## Usage

1. Type your message in the chat input
2. Press Enter or click Send
3. The assistant's response will be:
   - Displayed in the chat
   - Spoken through text-to-speech
4. Press Ctrl+Space to interrupt the assistant's speech

## Development

The frontend is built with vanilla JavaScript and Tailwind CSS for easy customization. You can modify the UI by editing the files in the `frontend` directory:

- `index.html`: Main layout
- `chat.js`: WebSocket and chat logic
- `styles.css`: Custom styles

The backend is modular and can be extended by modifying `chat_dialogue_system.py`.
