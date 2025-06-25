// This script handles the client-side logic for the AI Grammar Checker Tool.

document.addEventListener('DOMContentLoaded', () => {
    // Get references to HTML elements
    const inputText = document.getElementById('inputText');
    const outputText = document.getElementById('outputText');
    const checkGrammarBtn = document.getElementById('checkGrammarBtn');
    const loader = document.getElementById('loader');
    const errorMessage = document.getElementById('errorMessage');
    const inputCounter = document.getElementById('inputCounter');
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
        if (checkGrammarBtn) checkGrammarBtn.disabled = false;
        if (outputText) outputText.value = ''; // Clear output
        if (outputCounter) updateWordCount(outputText, outputCounter); // Reset output word count
    }

    // Initial update on page load for input text area
    if (inputText && inputCounter) {
        updateWordCount(inputText, inputCounter);
        // Update word count for input text area on input
        inputText.addEventListener('input', () => {
            updateWordCount(inputText, inputCounter);
            resetUI(); // Reset UI whenever input changes
        });
    }

    // Check Grammar button click handler
    if (checkGrammarBtn && inputText && outputText && loader && errorMessage) {
        checkGrammarBtn.addEventListener('click', async () => {
            const text = inputText.value.trim();

            // Basic validation
            if (!text) {
                errorMessage.textContent = 'Please enter text to check grammar.';
                errorMessage.style.display = 'block';
                outputText.value = ''; // Clear any previous output
                if (outputCounter) updateWordCount(outputText, outputCounter);
                return;
            }

            // Clear previous output and error, show loader
            outputText.value = '';
            if (outputCounter) updateWordCount(outputText, outputCounter);
            errorMessage.style.display = 'none';
            loader.style.display = 'block';
            checkGrammarBtn.disabled = true; // Disable button during processing

            try {
                // Call your Flask backend API for grammar checking
                const response = await fetch('/api/check_grammar', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ text: text }),
                });

                const data = await response.json();

                if (!response.ok) {
                    // Handle API errors
                    throw new Error(data.error || 'Failed to check grammar.');
                }

                // Display the corrected text
                if (outputText) {
                    outputText.value = data.corrected_text;
                    updateWordCount(outputText, outputCounter);
                }

            } catch (error) {
                console.error('Error during grammar check:', error);
                errorMessage.textContent = `Error: ${error.message}`;
                errorMessage.style.display = 'block';
                outputText.value = 'Could not check grammar. Please try again.';
                if (outputCounter) updateWordCount(outputText, outputCounter);
            } finally {
                // Hide loader and re-enable button
                loader.style.display = 'none';
                checkGrammarBtn.disabled = false;
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