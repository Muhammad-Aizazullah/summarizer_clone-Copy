// This script handles the client-side logic for the AI Trending News Generator Tool.

document.addEventListener('DOMContentLoaded', () => {
    // Get references to HTML elements
    const newsTopicInput = document.getElementById('newsTopicInput');
    const newsCategoryInput = document.getElementById('newsCategoryInput');
    const numArticlesInput = document.getElementById('numArticlesInput');
    const generateNewsBtn = document.getElementById('generateNewsBtn');
    const newsOutput = document.getElementById('newsOutput');
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
        if (generateNewsBtn) generateNewsBtn.disabled = false;
        // Do not clear newsOutput here, only when input changes explicitly
        if (newsOutput) updateWordCount(newsOutput, outputCounter); // Update output word count
    }

    // Initial update on page load for output text area (it starts empty)
    if (newsOutput && outputCounter) {
        updateWordCount(newsOutput, outputCounter);
    }

    // Add event listeners to input fields to reset UI when user types/changes
    if (newsTopicInput) {
        newsTopicInput.addEventListener('input', () => {
            // Clear output and hide error/loader when input changes
            newsOutput.value = '';
            updateWordCount(newsOutput, outputCounter);
            resetUI();
        });
    }
    if (newsCategoryInput) {
        newsCategoryInput.addEventListener('change', () => { // For select element, 'change' is more appropriate
            newsOutput.value = '';
            updateWordCount(newsOutput, outputCounter);
            resetUI();
        });
    }
    if (numArticlesInput) {
        numArticlesInput.addEventListener('input', () => {
            newsOutput.value = '';
            updateWordCount(newsOutput, outputCounter);
            resetUI();
        });
    }

    // Generate News button click handler
    if (generateNewsBtn && newsOutput && loader && errorMessage) {
        generateNewsBtn.addEventListener('click', async () => {
            const topic = newsTopicInput.value.trim();
            const category = newsCategoryInput.value.trim();
            const numArticles = numArticlesInput.value.trim();

            // Basic validation: at least one input field should have a value
            if (!topic && !category) {
                errorMessage.textContent = 'Please enter a news topic/keywords or select a category.';
                errorMessage.style.display = 'block';
                newsOutput.value = ''; // Clear any previous output
                if (outputCounter) updateWordCount(newsOutput, outputCounter);
                return;
            }

            // Clear previous output and error, show loader
            newsOutput.value = '';
            if (outputCounter) updateWordCount(newsOutput, outputCounter);
            errorMessage.style.display = 'none';
            loader.style.display = 'block';
            generateNewsBtn.disabled = true; // Disable button during processing

            try {
                // Prepare the prompt for the Gemini API
                let prompt = `Generate ${numArticles} trending news headlines and a short summary for each.`;
                if (topic) {
                    prompt += ` Focus on topics related to "${topic}".`;
                }
                if (category) {
                    prompt += ` From the category "${category}".`;
                }
                prompt += ` Ensure the news is diverse and sounds realistic.`;

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
                    newsOutput.value = generatedText;
                    updateWordCount(newsOutput, outputCounter);
                } else {
                    errorMessage.textContent = 'Error: Could not generate trending news. Invalid response from AI.';
                    errorMessage.style.display = 'block';
                    console.error('Gemini API response structure unexpected:', result);
                }

            } catch (error) {
                console.error('Error during news generation:', error);
                errorMessage.textContent = `Error: ${error.message}. Please try again.`;
                errorMessage.style.display = 'block';
                newsOutput.value = 'Could not generate trending news. Please try again.';
                if (outputCounter) updateWordCount(newsOutput, outputCounter);
            } finally {
                // Hide loader and re-enable button
                loader.style.display = 'none';
                generateNewsBtn.disabled = false;
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
