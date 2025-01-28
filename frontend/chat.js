// WebSocket connection
let ws = null;
let currentAudioId = null;

// Connect to WebSocket
function connectWebSocket() {
    ws = new WebSocket('ws://localhost:8000/ws');
    
    ws.onopen = () => {
        updateStatus('Connected');
    };
    
    ws.onclose = () => {
        updateStatus('Disconnected');
        // Attempt to reconnect after 2 seconds
        setTimeout(connectWebSocket, 2000);
    };
    
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleWebSocketMessage(data);
    };
}

// Handle incoming WebSocket messages
function handleWebSocketMessage(data) {
    if (data.type === 'audio_start') {
        currentAudioId = data.audio_id;
        updateStatus('Assistant is speaking...');
    } else if (data.type === 'interrupt') {
        if (data.audio_id === currentAudioId) {
            currentAudioId = null;
            updateStatus('Speech interrupted');
        }
    } else if (data.message) {
        addMessage(data.message);
    }
}

// Add message to chat
function addMessage(message) {
    const chatMessages = document.getElementById('chat-messages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `p-2 mb-2 rounded ${
        message.role === 'user' 
            ? 'bg-blue-100 ml-12' 
            : 'bg-gray-100 mr-12'
    }`;
    
    messageDiv.innerHTML = `
        <div class="font-semibold">${message.role === 'user' ? 'You' : 'Assistant'}</div>
        <div>${message.content}</div>
    `;
    
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Update status display
function updateStatus(status) {
    const statusElement = document.getElementById('status');
    statusElement.textContent = status;
}

// Handle form submission
document.getElementById('chat-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const input = document.getElementById('message-input');
    const message = input.value.trim();
    
    if (message) {
        // Add user message to chat
        addMessage({
            role: 'user',
            content: message
        });
        
        // Send message through WebSocket
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({
                role: 'user',
                content: message
            }));
        }
        
        input.value = '';
    }
});

// Handle Ctrl+Space for interruption
document.addEventListener('keydown', async (e) => {
    if (e.code === 'Space' && e.ctrlKey) {
        e.preventDefault();
        
        if (currentAudioId) {
            try {
                const response = await fetch('http://localhost:8000/interrupt', {
                    method: 'POST'
                });
                
                if (response.ok) {
                    updateStatus('Interrupting...');
                }
            } catch (error) {
                console.error('Failed to interrupt:', error);
            }
        }
    }
});

// Initialize WebSocket connection
connectWebSocket();
