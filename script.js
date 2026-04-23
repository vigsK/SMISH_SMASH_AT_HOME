document.addEventListener('DOMContentLoaded', () => {
    const checkBtn = document.getElementById('check-btn');
    const resetBtn = document.getElementById('reset-btn');
    const smsInput = document.getElementById('sms-input');
    const resultContainer = document.getElementById('result-container');
    const btnText = checkBtn.querySelector('.btn-text');
    const loader = checkBtn.querySelector('.loader-dots');

    const resultBadge = document.getElementById('result-badge');
    const resultTitle = document.getElementById('result-title');
    const resultDescription = document.getElementById('result-description');
    const confidenceFill = document.getElementById('confidence-fill');
    const confidenceValue = document.getElementById('confidence-value');

    const sampleItems = document.querySelectorAll('.sample-item');

    const API_URL = 'http://localhost:5000/predict';

    // Handle sample clicks
    sampleItems.forEach(item => {
        item.addEventListener('click', () => {
            smsInput.value = item.getAttribute('data-text');
            resultContainer.style.display = 'none';
            smsInput.focus();
        });
    });

    checkBtn.addEventListener('click', async () => {
        const text = smsInput.value.trim();
        if (!text) {
            alert('Please enter some text to analyze.');
            return;
        }

        // UI State: Loading
        setLoading(true);
        resultContainer.style.display = 'none';

        try {
            const response = await fetch(API_URL, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ text })
            });

            if (!response.ok) throw new Error('API request failed');

            const data = await response.json();
            displayResult(data);
        } catch (error) {
            console.error('Error:', error);
            alert('Could not connect to the backend server. Make sure it is running on port 5000.');
        } finally {
            setLoading(false);
        }
    });

    resetBtn.addEventListener('click', () => {
        smsInput.value = '';
        resultContainer.style.display = 'none';
        smsInput.focus();
    });

    function setLoading(isLoading) {
        if (isLoading) {
            checkBtn.disabled = true;
            btnText.style.display = 'none';
            loader.style.display = 'flex';
        } else {
            checkBtn.disabled = false;
            btnText.style.display = 'block';
            loader.style.display = 'none';
        }
    }

    function displayResult(data) {
        resultContainer.style.display = 'block';
        
        const isSmishing = data.is_smishing;
        const confidence = data.confidence ? Math.round(data.confidence * 100) : 0;

        // Reset colors to black for B&W theme
        resultTitle.style.color = '#000000';
        confidenceFill.style.backgroundColor = '#000000';

        if (isSmishing) {
            resultBadge.textContent = 'HIGH RISK';
            resultBadge.style.backgroundColor = '#000000';
            resultBadge.style.color = '#ffffff';
            resultTitle.textContent = 'Smishing Detected';
            resultDescription.textContent = 'This message contains patterns commonly used in phishing attacks, such as suspicious URLs, urgent language, or financial requests.';
        } else {
            resultBadge.textContent = 'SAFE';
            resultBadge.style.backgroundColor = '#ffffff';
            resultBadge.style.color = '#000000';
            resultTitle.textContent = 'Legitimate Message';
            resultDescription.textContent = 'Our AI did not find any strong indicators of social engineering. However, always remain cautious with unknown senders.';
        }

        // Update confidence bar without animation
        confidenceFill.style.width = `${confidence}%`;
        confidenceValue.textContent = `${confidence}%`;

        // Scroll to results
        resultContainer.scrollIntoView({ behavior: 'auto', block: 'nearest' });
    }
});
