document.addEventListener('DOMContentLoaded', () => {
    const messageInput = document.getElementById('message-input');
    const charCount = document.getElementById('char-count');
    const clearBtn = document.getElementById('clear-btn');
    const analyzeBtn = document.getElementById('analyze-btn');
    const resetBtn = document.getElementById('reset-btn');
    
    const detectionCard = document.getElementById('detection-card');
    const loadingCard = document.getElementById('loading-card');
    const resultCard = document.getElementById('result-card');
    const errorDiv = document.getElementById('error-message');

    // Result UI elements
    const resultIcon = document.getElementById('result-icon');
    const resultTitle = document.getElementById('result-title');
    const resultSubtitle = document.getElementById('result-subtitle');
    const resultOriginal = document.getElementById('result-original');
    const resultPrediction = document.getElementById('result-prediction');
    const resultStatusTag = document.getElementById('result-status-tag');

    // Update character counter
    messageInput.addEventListener('input', () => {
        charCount.textContent = messageInput.value.length;
    });

    // Clear input
    clearBtn.addEventListener('click', () => {
        messageInput.value = '';
        charCount.textContent = '0';
        hideError();
    });

    // Reset UI to start over
    resetBtn.addEventListener('click', () => {
        resultCard.classList.add('hidden');
        resultCard.classList.remove('theme-safe', 'theme-danger');
        
        messageInput.value = '';
        charCount.textContent = '0';
        hideError();
        
        detectionCard.classList.remove('hidden');
    });

    // Handle analysis
    analyzeBtn.addEventListener('click', async () => {
        const text = messageInput.value.trim();
        
        if (!text) {
            showError("Please enter a message to analyze.");
            return;
        }

        hideError();
        
        // Show loading state
        detectionCard.classList.add('hidden');
        loadingCard.classList.remove('hidden');

        try {
            const response = await fetch('/validate-comment', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ text: text })
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => null);
                throw new Error(errorData?.error || `HTTP Error: ${response.status}`);
            }

            const data = await response.json();
            
            // Artificial delay to show loading animation (looks more professional for ML)
            setTimeout(() => {
                displayResult(text, data);
            }, 800);

        } catch (error) {
            console.error("Analysis failed:", error);
            loadingCard.classList.add('hidden');
            detectionCard.classList.remove('hidden');
            showError(`Failed to connect to the analysis engine. Please try again. (${error.message})`);
        }
    });

    function showError(msg) {
        errorDiv.textContent = msg;
        errorDiv.classList.remove('hidden');
    }

    function hideError() {
        errorDiv.classList.add('hidden');
        errorDiv.textContent = '';
    }

    function displayResult(originalText, data) {
        loadingCard.classList.add('hidden');
        
        // Truncate original message if it's too long for display
        const truncatedMsg = originalText.length > 200 ? originalText.substring(0, 197) + '...' : originalText;
        resultOriginal.textContent = `"${truncatedMsg}"`;
        
        resultPrediction.textContent = data.prediction === 1 ? '1 (Cyberbullying)' : '0 (Safe)';
        document.getElementById('result-toxic').textContent = data.toxic_word_detected ? 'Yes' : 'No';
        document.getElementById('result-targeted').textContent = data.targeted_abuse ? 'Yes' : 'No';
        
        resultStatusTag.textContent = data.label.toUpperCase();
        document.getElementById('explanation-text').textContent = data.message;

        // Reset themes
        resultCard.classList.remove('theme-safe', 'theme-danger', 'theme-warning');

        // Safe message
        if (data.label === 'safe') {
            resultCard.classList.add('theme-safe');
            resultIcon.textContent = '✅';
            resultTitle.textContent = 'SAFE MESSAGE';
            resultSubtitle.textContent = 'Message appears to be non-cyberbullying.';
        } 
        // Cyberbullying message
        else if (data.label === 'cyberbullying') {
            resultCard.classList.add('theme-danger');
            resultIcon.textContent = '⚠️';
            resultTitle.textContent = 'CYBERBULLYING DETECTED';
            resultSubtitle.textContent = 'This message has been identified as potentially harmful.';
        }
        // Review / Ambiguous
        else if (data.label === 'review') {
            resultCard.classList.add('theme-warning');
            resultIcon.textContent = '👀';
            resultTitle.textContent = 'POTENTIALLY OFFENSIVE';
            resultSubtitle.textContent = 'Offensive language detected, but context is ambiguous.';
        }

        resultCard.classList.remove('hidden');
    }
});
