(function() {
    // Configuration
    const API_URL = 'http://localhost:5000'; // Change this to your production URL
    
    // Create and inject the widget
    function initCarAudioBot() {
        // Prevent multiple initializations
        if (document.getElementById('car-audio-bot-widget')) {
            return;
        }
        
        // Load CSS
        const link = document.createElement('link');
        link.rel = 'stylesheet';
        link.href = API_URL + '/widget.css';
        document.head.appendChild(link);
        
        // Create widget container
        const widgetContainer = document.createElement('div');
        widgetContainer.id = 'car-audio-bot-widget';
        document.body.appendChild(widgetContainer);
        
        // Load widget script
        const script = document.createElement('script');
        script.src = API_URL + '/widget.js';
        script.onload = function() {
            if (window.CarAudioBot) {
                window.CarAudioBot.init(API_URL);
            }
        };
        document.body.appendChild(script);
    }
    
    // Initialize when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initCarAudioBot);
    } else {
        initCarAudioBot();
    }
})();
