// frontend/src/app.js

document.getElementById('classifyBtn').addEventListener('click', async () => {
    const smsInput = document.getElementById('smsInput').value;
    const resultDiv = document.getElementById('result');
    
    // Clear previous results and show loading state
    resultDiv.textContent = 'Classifying...';
    resultDiv.className = 'result'; // Reset class for styling
    resultDiv.style.display = 'block'; // Make sure it's visible

    if (!smsInput.trim()) {
        resultDiv.textContent = 'Please enter a message to classify.';
        resultDiv.classList.add('error-result');
        return;
    }

    try {
        // Assume backend is running on http://127.0.0.1:8000
        const response = await fetch('http://127.0.0.1:8000/predict', { 
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message: smsInput }),
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Failed to get prediction from server.');
        }

        const data = await response.json();
        const prediction = data.prediction; // 'ham' or 'spam'
        const recordId = data.record_id;

        resultDiv.textContent = `Prediction: This message is ${prediction.toUpperCase()}! (Record ID: ${recordId})`;
        resultDiv.classList.add(prediction === 'spam' ? 'spam-result' : 'ham-result');

    } catch (error) {
        resultDiv.textContent = `Error: ${error.message}`;
        resultDiv.classList.add('error-result');
        console.error('Prediction failed:', error);
    }
});