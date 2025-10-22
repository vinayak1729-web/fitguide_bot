window.CarAudioBot = (function() {
    let API_URL = '';
    let isOpen = false;
    
    function createWidget() {
        const widgetHTML = `
            <!-- Floating Chat Button -->
            <div id="cab-chat-button" class="cab-chat-button cab-pulse">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="white" width="28" height="28">
                    <path d="M20 2H4c-1.1 0-2 .9-2 2v18l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zm0 14H6l-2 2V4h16v12z"/>
                </svg>
            </div>

            <!-- Chat Window -->
            <div id="cab-chat-window" class="cab-chat-window cab-hidden">
                <div class="cab-chat-header">
                    <h3>🎵 Car Audio Bot</h3>
                    <button id="cab-close-chat" class="cab-close-btn">×</button>
                </div>
                <div id="cab-chat-messages" class="cab-chat-messages">
                    <div class="cab-bot-message">
                        <p>👋 Hello! I'm your Car Audio Assistant. Ask me about speakers, vehicle compatibility, and installation details!</p>
                    </div>
                </div>
                <div class="cab-chat-input-container">
                    <input type="text" id="cab-chat-input" placeholder="Ask about car speakers..." />
                    <button id="cab-send-btn">Send</button>
                </div>
                <div id="cab-loading" class="cab-loading cab-hidden">
                    <span></span><span></span><span></span>
                </div>
            </div>
        `;
        
        const container = document.getElementById('car-audio-bot-widget');
        container.innerHTML = widgetHTML;
        
        attachEventListeners();
    }
    
    function attachEventListeners() {
        const chatButton = document.getElementById('cab-chat-button');
        const chatWindow = document.getElementById('cab-chat-window');
        const closeChat = document.getElementById('cab-close-chat');
        const chatInput = document.getElementById('cab-chat-input');
        const sendBtn = document.getElementById('cab-send-btn');
        
        // Remove pulse animation after first interaction
        chatButton.addEventListener('click', () => {
            chatButton.classList.remove('cab-pulse');
            toggleChat();
        }, { once: true });
        
        // Subsequent clicks
        chatButton.addEventListener('click', toggleChat);
        
        closeChat.addEventListener('click', () => {
            closeWindow();
        });
        
        // Send message
        sendBtn.addEventListener('click', sendMessage);
        chatInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });
    }
    
    function toggleChat() {
        const chatButton = document.getElementById('cab-chat-button');
        const chatWindow = document.getElementById('cab-chat-window');
        const chatInput = document.getElementById('cab-chat-input');
        
        isOpen = !isOpen;
        chatWindow.classList.toggle('cab-hidden');
        chatButton.classList.toggle('cab-active');
        
        if (isOpen) {
            setTimeout(() => {
                chatInput.focus();
            }, 400);
        }
    }
    
    function closeWindow() {
        const chatButton = document.getElementById('cab-chat-button');
        const chatWindow = document.getElementById('cab-chat-window');
        
        isOpen = false;
        chatWindow.classList.add('cab-hidden');
        chatButton.classList.remove('cab-active');
    }
    
    async function sendMessage() {
        const chatInput = document.getElementById('cab-chat-input');
        const sendBtn = document.getElementById('cab-send-btn');
        const loading = document.getElementById('cab-loading');
        
        const message = chatInput.value.trim();
        
        if (!message) return;
        
        // Add user message to chat
        addMessage(message, 'user');
        chatInput.value = '';
        
        // Disable input while processing
        chatInput.disabled = true;
        sendBtn.disabled = true;
        loading.classList.remove('cab-hidden');
        
        try {
            const response = await fetch(API_URL + '/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message: message })
            });
            
            const data = await response.json();
            
            if (data.success) {
                addMessage(data.message, 'bot');
            } else {
                addMessage('Sorry, I encountered an error. Please try again.', 'bot');
            }
        } catch (error) {
            console.error('Error:', error);
            addMessage('Sorry, I could not connect to the server. Please try again.', 'bot');
        } finally {
            chatInput.disabled = false;
            sendBtn.disabled = false;
            loading.classList.add('cab-hidden');
            chatInput.focus();
        }
    }
    
    function addMessage(text, sender) {
        const chatMessages = document.getElementById('cab-chat-messages');
        const messageDiv = document.createElement('div');
        messageDiv.className = sender === 'user' ? 'cab-user-message' : 'cab-bot-message';
        
        const messageParagraph = document.createElement('p');
        messageParagraph.textContent = text;
        
        messageDiv.appendChild(messageParagraph);
        chatMessages.appendChild(messageDiv);
        
        // Scroll to bottom smoothly
        chatMessages.scrollTo({
            top: chatMessages.scrollHeight,
            behavior: 'smooth'
        });
    }
    
    return {
        init: function(apiUrl) {
            API_URL = apiUrl;
            createWidget();
        }
    };
})();
