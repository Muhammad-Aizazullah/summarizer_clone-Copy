// This script handles the client-side logic for the AI Essay Generator Tool.

document.addEventListener('DOMContentLoaded', () => {
    // Get references to HTML elements
    const essayTopicInput = document.getElementById('essayTopicInput');
    const essayKeywordsInput = document.getElementById('essayKeywordsInput');
    const essayToneInput = document.getElementById('essayToneInput');
    const essayLengthInput = document.getElementById('essayLengthInput');
    const essayLengthValue = document.getElementById('essayLengthValue');
    const generateEssayBtn = document.getElementById('generateEssayBtn');
    const essayOutput = document.getElementById('essayOutput');
    const loader = document.getElementById('loader');
    const errorMessage = document.getElementById('errorMessage');
    const outputCounter = document.getElementById('outputCounter');

    // Function to update word count for a given textarea
    function updateWordCount(textArea, counterElement) {
        if (!textArea || !counterElement) return;
        const text = textArea.value;
        if (text.trim() === '') {
            counterElement.textContent = '0 words';
        } else {
            const wordCount = text.trim().split(/\s+/).filter(word => word.length > 0).length;
            counterElement.textContent = `${wordCount} words`;
        }
    }

    // Function to reset UI elements
    function resetUI() {
        if (loader) loader.style.display = 'none';
        if (errorMessage) errorMessage.style.display = 'none';
        if (generateEssayBtn) generateEssayBtn.disabled = false;
        // Do not clear essayOutput here, only when input changes
        if (essayOutput) updateWordCount(essayOutput, outputCounter); // Update output word count
    }

    // Initial update on page load for output text area (it starts empty)
    if (essayOutput && outputCounter) {
        updateWordCount(essayOutput, outputCounter);
    }

    // Update desired length display as slider moves
    if (essayLengthInput && essayLengthValue) {
        essayLengthInput.addEventListener('input', () => {
            essayLengthValue.textContent = essayLengthInput.value;
            resetUI(); // Reset UI whenever slider value changes
        });
    }

    // Add event listeners to input fields to reset UI when user types
    if (essayTopicInput) {
        essayTopicInput.addEventListener('input', () => {
            // Clear output and hide error/loader when input changes
            essayOutput.value = '';
            updateWordCount(essayOutput, outputCounter);
            resetUI();
        });
    }
    if (essayKeywordsInput) {
        essayKeywordsInput.addEventListener('input', () => {
            essayOutput.value = '';
            updateWordCount(essayOutput, outputCounter);
            resetUI();
        });
    }
    if (essayToneInput) {
        essayToneInput.addEventListener('change', () => { // For select element, 'change' is more appropriate
            essayOutput.value = '';
            updateWordCount(essayOutput, outputCounter);
            resetUI();
        });
    }

    // Generate Essay button click handler
    if (generateEssayBtn && essayTopicInput && essayOutput && loader && errorMessage) {
        generateEssayBtn.addEventListener('click', async () => {
            const essayTopic = essayTopicInput.value.trim();
            const essayKeywords = essayKeywordsInput.value.trim();
            const essayTone = essayToneInput.value.trim();
            const essayLength = essayLengthInput.value.trim(); // Desired length in words

            // Basic validation: ensure essay topic is provided
            if (!essayTopic) {
                errorMessage.textContent = 'Please enter an essay topic to generate an essay.';
                errorMessage.style.display = 'block';
                essayOutput.value = ''; // Clear any previous output
                if (outputCounter) updateWordCount(essayOutput, outputCounter);
                return;
            }

            // Clear previous output and error, show loader
            essayOutput.value = '';
            if (outputCounter) updateWordCount(essayOutput, outputCounter);
            errorMessage.style.display = 'none';
            loader.style.display = 'block';
            generateEssayBtn.disabled = true; // Disable button during processing

            try {
                // Prepare the prompt for the Gemini API
                let prompt = `Generate an essay on the topic "${essayTopic}".`;
                if (essayKeywords) {
                    prompt += ` Include the following keywords or concepts: ${essayKeywords}.`;
                }
                if (essayTone) {
                    prompt += ` The tone of the essay should be ${essayTone}.`;
                }
                if (essayLength) {
                    prompt += ` The essay should be approximately ${essayLength} words long.`;
                }
                prompt += ` Ensure the essay has a clear introduction, body paragraphs, and conclusion.`;


                // Fetch call to the Gemini API (adjust endpoint if using a backend proxy)
                // Using gemini-2.0-flash model for text generation
                let chatHistory = [];
                chatHistory.push({ role: "user", parts: [{ text: prompt }] });
                const payload = { contents: chatHistory };
                const apiKey = ""; // Canvas will automatically provide the API key
                const apiUrl = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=${apiKey}`;

                const response = await fetch(apiUrl, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });

                const result = await response.json();

                if (result.candidates && result.candidates.length > 0 &&
                    result.candidates[0].content && result.candidates[0].content.parts &&
                    result.candidates[0].content.parts.length > 0) {
                    const generatedText = result.candidates[0].content.parts[0].text;
                    essayOutput.value = generatedText;
                    updateWordCount(essayOutput, outputCounter);
                } else {
                    errorMessage.textContent = 'Error: Could not generate essay. Invalid response from AI.';
                    errorMessage.style.display = 'block';
                    console.error('Gemini API response structure unexpected:', result);
                }

            } catch (error) {
                console.error('Error during essay generation:', error);
                errorMessage.textContent = `Error: ${error.message}. Please try again.`;
                errorMessage.style.display = 'block';
                essayOutput.value = 'Could not generate essay. Please try again.';
                if (outputCounter) updateWordCount(essayOutput, outputCounter);
            } finally {
                // Hide loader and re-enable button
                loader.style.display = 'none';
                generateEssayBtn.disabled = false;
            }
        });
    }

    // --- FAQ Accordion Functionality (reused from main script for consistency) ---
    const faqItems = document.querySelectorAll('.faq-item');
    faqItems.forEach(item => {
        const question = item.querySelector('.faq-question');
        const answer = item.querySelector('.faq-answer');
        const icon = item.querySelector('.faq-question i');

        if (question) {
            question.addEventListener('click', () => {
                // Close other active items
                const currentlyActive = document.querySelector('.faq-item.active');
                if (currentlyActive && currentlyActive !== item) {
                    currentlyActive.classList.remove('active');
                    const otherAnswer = currentlyActive.querySelector('.faq-answer');
                    const otherIcon = currentlyActive.querySelector('.faq-question i');
                    if (otherAnswer) otherAnswer.style.maxHeight = "0";
                    if (otherIcon) {
                        otherIcon.classList.remove('fa-minus');
                        otherIcon.classList.add('fa-plus');
                    }
                }
                // Toggle current item
                item.classList.toggle('active');

                if (item.classList.contains('active')) {
                    if (answer) {
                        answer.style.maxHeight = answer.scrollHeight + "px";
                    }
                    if (icon) {
                        icon.classList.remove('fa-plus');
                        icon.classList.add('fa-minus');
                    }
                } else {
                    if (answer) {
                        answer.style.maxHeight = "0";
                    }
                    if (icon) {
                        icon.classList.remove('fa-minus');
                        icon.classList.add('fa-plus');
                    }
                }
            });
        }
    });
    // --- End FAQ Accordion Functionality ---
});
