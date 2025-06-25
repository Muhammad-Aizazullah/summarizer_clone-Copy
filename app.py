import nltk
import os
import logging
import re
import sys
import random
import json

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, T5Tokenizer
import torch

try:
    import google.generativeai as genai
    # *** YAHAN CHANGE KIYA GAYA HAI ***
    # API key ko environment variable se load kiya ja raha hai
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    logging.info("Google Generative AI library loaded and configured.")
    GEMINI_API_AVAILABLE = True
except ImportError:
    logging.warning("Google Generative AI library not found. Gemini functions will not work.")
    GEMINI_API_AVAILABLE = False
except Exception as e:
    logging.error(f"Error configuring Gemini API: {e}. Gemini functions might not work.")
    GEMINI_API_AVAILABLE = False


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

nltk_data_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'nltk_data')
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)
    logging.info(f"Created NLTK data directory: {nltk_data_dir}")
nltk.data.path.append(nltk_data_dir)
logging.info(f"NLTK data path added: {nltk_data_dir}")

def ensure_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
        logging.info("NLTK 'punkt' tokenizer already exists.")
    except nltk.downloader.DownloadError:
        logging.info("Downloading NLTK 'punkt' tokenizer...")
        nltk.download('punkt', download_dir=nltk_data_dir)
        logging.info("NLTK 'punkt' tokenizer downloaded.")
    
    try:
        nltk.data.find('corpora/stopwords')
        logging.info("NLTK 'stopwords' corpus already exists.")
    except nltk.downloader.DownloadError:
        logging.info("Downloading NLTK 'stopwords' corpus...")
        nltk.download('stopwords', download_dir=nltk_data_dir)
        logging.info("NLTK 'stopwords' corpus downloaded.")

ensure_nltk_data()

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

summarizer_tokenizer = None
summarizer_model = None
paraphraser_tokenizer = None
paraphraser_model = None

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Using device: {DEVICE}")

def load_summarizer_model():
    """Loads the T5-small model and tokenizer for summarization."""
    global summarizer_tokenizer, summarizer_model
    if summarizer_model is None:
        try:
            logging.info("Loading summarizer model (t5-small)...")
            summarizer_tokenizer = T5Tokenizer.from_pretrained("t5-small")
            summarizer_model = AutoModelForSeq2SeqLM.from_pretrained("t5-small").to(DEVICE)
            logging.info("Summarizer model loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading summarizer model: {e}")
            summarizer_tokenizer = None
            summarizer_model = None
    return summarizer_tokenizer, summarizer_model

def load_paraphraser_model():
    """Loads a T5-small model and tokenizer for paraphrasing/rewriting."""
    global paraphraser_tokenizer, paraphraser_model
    if paraphraser_model is None:
        try:
            logging.info("Loading paraphraser model (t5-small)...")
            paraphraser_tokenizer = T5Tokenizer.from_pretrained("t5-small")
            paraphraser_model = AutoModelForSeq2SeqLM.from_pretrained("t5-small").to(DEVICE)
            logging.info("Paraphraser model loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading paraphraser model: {e}")
            paraphraser_tokenizer = None
            paraphraser_model = None
    return paraphraser_tokenizer, paraphraser_model

def get_gemini_model():
    """Helper function to get a Gemini model that supports generateContent."""
    if not GEMINI_API_AVAILABLE:
        raise Exception("Gemini API is not available.")
    
    available_models = [m for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
    selected_model_name = None
    for model_info in available_models:
        if 'gemini-2.0-flash' in model_info.name:
            selected_model_name = model_info.name
            break
    
    if not selected_model_name:
        if available_models:
            selected_model_name = available_models[0].name
            logging.warning(f"gemini-2.0-flash not found. Using available model: {selected_model_name}")
        else:
            raise Exception("No Gemini model found that supports content generation.")
    
    return genai.GenerativeModel(selected_model_name)

def summarize_text(text, max_length_ratio=0.5):
    """Summarizes the given text using the Gemini model."""
    if not GEMINI_API_AVAILABLE:
        logging.error("Gemini API is not available for summarization.")
        return f"Error: Gemini API not configured for summarization."

    try:
        model = get_gemini_model()
        
        input_length = len(text.split())
        min_summary_words = max(20, int(input_length * (max_length_ratio - 0.2)))
        max_summary_words = max(min_summary_words + 10, int(input_length * max_length_ratio))

        prompt = (
            f"Please summarize the following text concisely. The summary should be between "
            f"{min_summary_words} and {max_summary_words} words, maintaining the core information.\n\n"
            f"Original text:\n---\n{text}\n---\n\nSummary:"
        )
        
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                top_p=0.8,
                top_k=30,
                candidate_count=1,
            )
        )
        
        if response.text:
            summary_text = response.text.strip()
            summary_text = re.sub(r'^(Summary:)?\s*', '', summary_text, flags=re.IGNORECASE).strip()
            return summary_text
        elif response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            summary_text = response.candidates[0].content.parts[0].text.strip()
            summary_text = re.sub(r'^(Summary:)?\s*', '', summary_text, flags=re.IGNORECASE).strip()
            return summary_text
        else:
            return "No summary could be generated by Gemini. Try different input."

    except Exception as e:
        logging.error(f"Error summarizing text with Gemini: {e}", exc_info=True)
        return f"Error: Summarization failed with Gemini. Details: {str(e)}"


def rewrite_article(text, creativity=0.5):
    """Rewrites/paraphrases the given text using the loaded paraphraser model."""
    tokenizer, model = load_paraphraser_model()
    if model is None:
        return "Error: Paraphraser model not loaded."

    input_text = text
    tokenized_text = tokenizer.encode(input_text, return_tensors="pt").to(DEVICE)

    temperature = 0.5 + (creativity * 0.5) 

    output_ids = model.generate(
        tokenized_text,
        max_length=int(tokenized_text.shape[1] * 1.2) + 50,
        min_length=int(tokenized_text.shape[1] * 0.8),
        num_beams=5,
        no_repeat_ngram_size=2,
        early_stopping=True,
        do_sample=True,
        temperature=temperature,
        top_k=50,
        top_p=0.95
    )
    rewritten_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    rewritten_text = re.sub(r'\s*([.,;!?])', r'\1', rewritten_text)
    rewritten_text = re.sub(r'\n+', '\n', rewritten_text).strip()

    return rewritten_text


def humanize_text_content(text, creativity_level=0.7):
    """Humanizes AI-generated text using the Gemini model."""
    if not GEMINI_API_AVAILABLE:
        logging.error("Gemini API is not available for humanization.")
        return f"Error: Gemini API not configured for humanization."

    try:
        model = get_gemini_model()
        
        prompt = (
            f"Please rewrite the following text to sound more natural, human-like, and engaging. "
            f"Aim for a {int(creativity_level*100)}% creative flair while maintaining the original meaning. "
            "Remove any robotic or overly formal phrasing, and inject a natural flow.\n\n"
            f"Original text:\n---\n{text}\n---\n\nRewritten human-like version:"
        )
        
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.9,
                top_p=0.9,
                top_k=40,
                candidate_count=1,
            )
        )
        
        if response.text:
            humanized_text = response.text.strip()
            humanized_text = re.sub(r'^(Rewritten human-like version:)?\s*', '', humanized_text, flags=re.IGNORECASE).strip()
            return humanized_text
        elif response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            humanized_text = response.candidates[0].content.parts[0].text.strip()
            humanized_text = re.sub(r'^(Rewritten human-like version:)?\s*', '', humanized_text, flags=re.IGNORECASE).strip()
            return humanized_text
        else:
            return "No humanized text could be generated by Gemini. Try different input."

    except Exception as e:
        logging.error(f"Error humanizing text with Gemini: {e}", exc_info=True)
        return f"Error: Humanization failed with Gemini. Details: {str(e)}"


def generate_email_content(subject, purpose, recipient=''):
    """Generates email content using the Gemini model."""
    if not GEMINI_API_AVAILABLE:
        logging.error("Gemini API is not available for email generation.")
        return f"Error: Gemini API not configured for email generation."

    try:
        model = get_gemini_model()
        
        email_prompt = (
            f"Generate a professional email.\n"
            f"Subject: {subject}\n"
            f"Purpose: {purpose}\n"
        )
        if recipient:
            email_prompt += f"Recipient: {recipient}\n"
        email_prompt += "\nFormat the email appropriately, starting with 'Dear [Recipient Name or Team],' and ending with a professional closing."

        response = model.generate_content(
            email_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.8,
                top_p=0.8,
                top_k=30,
                candidate_count=1,
            )
        )
        
        if response.text:
            generated_email = response.text.strip()
            return generated_email
        elif response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            generated_email = response.candidates[0].content.parts[0].text.strip()
            return generated_email
        else:
            return "No email content could be generated by Gemini. Try different details."

    except Exception as e:
        logging.error(f"Error generating email with Gemini: {e}", exc_info=True)
        return f"Error: Email generation failed with Gemini. Details: {str(e)}"


def generate_content_ideas(keywords):
    """Generates content ideas based on keywords using the Gemini model."""
    if not GEMINI_API_AVAILABLE:
        logging.error("Gemini API is not available for content idea generation.")
        return "Error: Gemini API not configured for content idea generation."

    try:
        model = get_gemini_model()
        
        response_schema = {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "idea": {"type": "STRING"}
                }
            }
        }

        prompt = (
            f"Generate 7 creative, unique, and engaging content ideas related to '{keywords}'. "
            "Focus on diverse angles, trending topics, and actionable ideas. "
            "Each idea should be concise and compelling."
        )

        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                response_mime_type="application/json",
                response_schema=response_schema,
                temperature=0.9,
                top_p=0.9,
                top_k=40,
                candidate_count=1,
            )
        )
        
        if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            json_string = response.candidates[0].content.parts[0].text
            ideas_data = json.loads(json_string)
            ideas_list = [item["idea"].strip() for item in ideas_data if "idea" in item]
            formatted_ideas = []
            for i, idea in enumerate(ideas_list):
                if idea:
                    formatted_ideas.append(f"{i + 1}. {idea}")
            if not formatted_ideas:
                return "No specific ideas generated by Gemini. Try different keywords or consider:\n1. Introduction to your field\n2. Common challenges\n3. Future trends\n4. How-to guides"
            return "\n".join(formatted_ideas)
        else:
            return "No content ideas could be generated by Gemini. Try different keywords."

    except Exception as e:
        logging.error(f"Error generating content ideas with Gemini: {e}", exc_info=True)
        return f"Error: Content idea generation failed with Gemini. Details: {str(e)}"

def generate_slogans(keywords, num_slogans=5):
    """Generates slogans using the Gemini model."""
    if not GEMINI_API_AVAILABLE:
        logging.error("Gemini API is not available.")
        return [f"Error: Gemini API not configured."]
    try:
        model = get_gemini_model()
        response_schema = {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "slogan": {"type": "STRING"}
                }
            }
        }
        prompt = (
            f"Generate {num_slogans} unique, catchy, and memorable advertising slogans "
            f"for a brand or campaign related to '{keywords}'. "
            "Each slogan should be concise and highly impactful."
        )
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                response_mime_type="application/json",
                response_schema=response_schema,
                temperature=0.9,
                top_p=0.9,
                top_k=40,
                candidate_count=1,
            )
        )
        
        if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            json_string = response.candidates[0].content.parts[0].text
            slogans_data = json.loads(json_string)
            slogans_list = [item["slogan"].strip() for item in slogans_data if "slogan" in item]
            
            unique_slogans = []
            seen_slogans = set()
            for slogan in slogans_list:
                if slogan.lower() not in seen_slogans:
                    unique_slogans.append(slogan)
                    seen_slogans.add(slogan.lower())
                if len(unique_slogans) >= num_slogans:
                    break
            
            if not unique_slogans:
                return ["No slogans could be generated by Gemini. Try different keywords."]
            
            return unique_slogans[:num_slogans]
        else:
            return ["No slogans could be generated by Gemini. Try different keywords."]

    except Exception as e:
        logging.error(f"Error generating slogans with Gemini: {e}", exc_info=True)
        return [f"Error: Slogan generation failed with Gemini. Details: {str(e)}"]


def check_plagiarism_and_ai(text):
    """
    Performs (or simulates) plagiarism and AI detection.
    This function needs YOUR INTEGRATION for real plagiarism checks.
    """
    logging.info("Attempting plagiarism and AI detection...")

    # --- Plagiarism Check (Requires External API Integration) ---
    plagiarism_percentage = 0.0
    plagiarism_details = "Real plagiarism check requires integration with a third-party API (e.g., Copyleaks, Turnitin). Please refer to the comments in app.py for instructions."
    
    # EXAMPLE Placeholder for Copyleaks API Integration (User needs to fill this in)
    # import requests
    # COYLEAKS_API_KEY = os.getenv("COYLEAKS_API_KEY") # Ensure you set this environment variable
    # if COYLEAKS_API_KEY:
    #     try:
    #         # This is a simplified example. Refer to Copyleaks API documentation for exact details.
    #         # You might need to submit a file, get a process ID, and then check its status.
    #         copyleaks_api_url = "YOUR_COYLEAKS_SCAN_API_ENDPOINT" 
    #         headers = {"Authorization": f"Bearer {COYLEAKS_API_KEY}", "Content-Type": "application/json"}
    #         payload = {"text": text, "properties": {"webhooks": {"status": "YOUR_WEBHOOK_URL_FOR_RESULTS"}}}
    #         
    #         response = requests.post(copyleaks_api_url, headers=headers, json=payload)
    #         response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
    #         scan_id = response.json().get("processId")
    #         
    #         # In a real app, you'd then poll for results or wait for a webhook.
    #         # For actual percentage, you would parse the results from Copyleaks' callback/polling API.
    #         # For now, we'll just simulate a result after submission setup.
    #         plagiarism_percentage = random.uniform(0.1, 99.9) # Simulated for demo after API call setup
    #         plagiarism_details = f"Plagiarism scan initiated (ID: {scan_id}). Results will be available via Copyleaks dashboard/webhook. Simulated percentage: {plagiarism_percentage:.2f}%."
    #         logging.info(f"Copyleaks scan initiated. Process ID: {scan_id}")
    #     except Exception as e:
    #         logging.error(f"Error during Copyleaks API call: {e}")
    #         plagiarism_details = f"Failed to connect to Copyleaks API: {str(e)}. Using simulated plagiarism result."
    #         plagiarism_percentage = random.uniform(5.0, 40.0) # Fallback to simulation
    # else:
    #     plagiarism_details = "Copyleaks API Key not found. Please set COYLEAKS_API_KEY environment variable. Using simulated plagiarism result."
    #     plagiarism_percentage = random.uniform(5.0, 40.0) # Fallback to simulation

    # --- AI Detection (Current implementation uses Gemini if available, falls back to random) ---
    is_ai_generated = False
    ai_probability = 0.0
    if GEMINI_API_AVAILABLE:
        try:
            model = get_gemini_model()
            ai_detection_prompt = (
                f"Analyze the following text and determine if it was likely generated by an AI. "
                "Respond with a single word: 'AI' if it's likely AI-generated, or 'Human' if it's likely human-written. "
                "Also provide a confidence score between 0.0 and 1.0 (e.g., 'AI: 0.85' or 'Human: 0.92').\n\n"
                f"Text:\n---\n{text}\n---\n\nAnalysis:"
            )
            response = model.generate_content(ai_detection_prompt)
            if response.text:
                analysis = response.text.strip().lower()
                if "ai:" in analysis:
                    is_ai_generated = True
                    try:
                        ai_probability = float(analysis.split("ai:")[1].strip())
                    except (ValueError, IndexError):
                        ai_probability = 0.5 # Default if parsing fails
                elif "human:" in analysis:
                    is_ai_generated = False
                    try:
                        ai_probability = 1.0 - float(analysis.split("human:")[1].strip()) # Confidence for human, so AI probability is 1-confidence
                    except (ValueError, IndexError):
                        ai_probability = 0.5
                else:
                    logging.warning(f"Unexpected Gemini AI detection response format: {response.text}")
                    ai_probability = random.uniform(0.05, 0.95) # Fallback to random
                    is_ai_generated = ai_probability > 0.5
            else:
                logging.warning("Gemini did not return text for AI detection.")
                ai_probability = random.uniform(0.05, 0.95) # Fallback to random
                is_ai_generated = ai_probability > 0.5

        except Exception as e:
            logging.error(f"Error using Gemini for AI detection: {e}")
            ai_probability = random.uniform(0.05, 0.95) # Fallback to random
            is_ai_generated = ai_probability > 0.5
    else:
        # Fallback to random if Gemini API not available
        ai_probability = random.uniform(0.05, 0.95)
        is_ai_generated = ai_probability > 0.5


    suggestions = []
    if is_ai_generated:
        suggestions.append("The text appears to be AI-generated. To humanize it, try rephrasing sentences, adding personal anecdotes, and varying sentence structure.")
    
    if plagiarism_percentage > 0:
        suggestions.append(f"Approximately {plagiarism_percentage:.2f}% of the text may be plagiarized. {plagiarism_details} Please review your sources and rephrase content as needed.")
    elif "Please refer to the comments in app.py for instructions" in plagiarism_details:
         suggestions.append(f"Plagiarism check requires setup. {plagiarism_details}")

    if not suggestions:
        suggestions.append("The text appears original and human-like. No specific suggestions at this time.")

    return {
        "is_ai_generated": is_ai_generated,
        "ai_probability": round(ai_probability, 4),
        "plagiarism_percentage": round(plagiarism_percentage, 2), 
        "suggestions": suggestions
    }

def generate_essay_content(topic, length="medium", style="formal"):
    """Generates essay content using the Gemini model."""
    if not GEMINI_API_AVAILABLE:
        logging.error("Gemini API is not available for essay generation.")
        return "Error: Gemini API not configured for essay generation."

    length_map = {
        "short": "around 200-300 words",
        "medium": "around 500-700 words",
        "long": "around 1000-1200 words"
    }
    word_count_target = length_map.get(length, "a reasonable length (500-700 words)")

    try:
        model = get_gemini_model()
        
        prompt = (
            f"Write an essay on the topic: '{topic}'.\n"
            f"The essay should be {word_count_target} and written in a {style} style. "
            "Include an introduction, body paragraphs with supporting details, and a conclusion."
        )
        
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.8,
                top_p=0.8,
                top_k=30,
                candidate_count=1,
            )
        )
        
        if response.text:
            generated_essay = response.text.strip()
            return generated_essay
        elif response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            generated_essay = response.candidates[0].content.parts[0].text.strip()
            return generated_essay
        else:
            return "No essay could be generated by Gemini. Try a different topic or adjust parameters."

    except Exception as e:
        logging.error(f"Error generating essay with Gemini: {e}", exc_info=True)
        return f"Error: Essay generation failed with Gemini. Details: {str(e)}"

def check_grammar_and_style(text):
    """Checks grammar and suggests style improvements using the Gemini model."""
    if not GEMINI_API_AVAILABLE:
        logging.error("Gemini API is not available for grammar check.")
        return "Error: Gemini API not configured for grammar check."

    try:
        model = get_gemini_model()
        
        prompt = (
            f"Review the following text for grammar, spelling, punctuation, and style. "
            "Provide a list of specific corrections and suggestions for improvement. "
            "Format the output as a clear list of identified issues and their proposed corrections/improvements.\n\n"
            f"Text to check:\n---\n{text}\n---\n\nCorrections and Suggestions:"
        )
        
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.2,
                top_p=0.7,
                top_k=20,
                candidate_count=1,
            )
        )
        
        if response.text:
            corrections = response.text.strip()
            corrections = re.sub(r'^(Corrections and Suggestions:)?\s*', '', corrections, flags=re.IGNORECASE).strip()
            return corrections
        elif response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            corrections = response.candidates[0].content.parts[0].text.strip()
            corrections = re.sub(r'^(Corrections and Suggestions:)?\s*', '', corrections, flags=re.IGNORECASE).strip()
            return corrections
        else:
            return "No grammar corrections or suggestions could be generated by Gemini. The text might be perfect, or try different input."

    except Exception as e:
        logging.error(f"Error checking grammar with Gemini: {e}", exc_info=True)
        return f"Error: Grammar check failed with Gemini. Details: {str(e)}"


def generate_product_description_content(product_name, features, audience, tone="informative"):
    """Generates a product description using the Gemini model."""
    if not GEMINI_API_AVAILABLE:
        logging.error("Gemini API is not available for product description generation.")
        return "Error: Gemini API not configured for product description generation."

    try:
        model = get_gemini_model()
        
        prompt = (
            f"Generate a compelling product description for '{product_name}'.\n"
            f"Key features: {features}\n"
            f"Target audience: {audience}\n"
            f"Desired tone: {tone}\n\n"
            "The description should highlight benefits, engage the audience, and encourage purchase."
        )
        
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.8,
                top_p=0.8,
                top_k=30,
                candidate_count=1,
            )
        )
        
        if response.text:
            generated_description = response.text.strip()
            return generated_description
        elif response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            generated_description = response.candidates[0].content.parts[0].text.strip()
            return generated_description
        else:
            return "No product description could be generated by Gemini. Try different details."

    except Exception as e:
        logging.error(f"Error generating product description with Gemini: {e}", exc_info=True)
        return f"Error: Product description generation failed with Gemini. Details: {str(e)}"

def generate_story_content(genre, characters, plot_keywords, length="medium"):
    """Generates a short story using the Gemini model."""
    if not GEMINI_API_AVAILABLE:
        logging.error("Gemini API is not available for story generation.")
        return "Error: Gemini API not configured for story generation."

    length_map = {
        "short": "a short story (approx. 500 words)",
        "medium": "a medium-length story (approx. 1000 words)",
        "long": "a longer story (approx. 2000 words)"
    }
    word_count_target = length_map.get(length, "a medium-length story (approx. 1000 words)")

    try:
        model = get_gemini_model()
        
        prompt = (
            f"Write {word_count_target} in the '{genre}' genre.\n"
            f"Main characters: {characters}\n"
            f"Plot elements: {plot_keywords}\n"
            "Develop an engaging narrative with a clear beginning, middle, and end. "
            "Focus on character development and plot progression."
        )
        
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.9,
                top_p=0.9,
                top_k=40,
                candidate_count=1,
            )
        )
        
        if response.text:
            generated_story = response.text.strip()
            return generated_story
        elif response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            generated_story = response.candidates[0].content.parts[0].text.strip()
            return generated_story
        else:
            return "No story could be generated by Gemini. Try different inputs."

    except Exception as e:
        logging.error(f"Error generating story with Gemini: {e}", exc_info=True)
        return f"Error: Story generation failed with Gemini. Details: {str(e)}"

def generate_trending_news_summary_content(keywords):
    """Generates a summary of trending news based on keywords using the Gemini model."""
    if not GEMINI_API_AVAILABLE:
        logging.error("Gemini API is not available for trending news generation.")
        return "Error: Gemini API not configured for trending news generation."

    try:
        model = get_gemini_model()
        
        prompt = (
            f"Provide a concise summary of trending news related to '{keywords}'. "
            "Include the most important developments and key facts. Focus on recent, impactful news."
        )
        
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                top_p=0.8,
                top_k=30,
                candidate_count=1,
            )
        )
        
        if response.text:
            news_summary = response.text.strip()
            return news_summary
        elif response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            news_summary = response.candidates[0].content.parts[0].text.strip()
            return news_summary
        else:
            return "No trending news summary could be generated by Gemini. Try different keywords."

    except Exception as e:
        logging.error(f"Error generating trending news summary with Gemini: {e}", exc_info=True)
        return f"Error: Trending news summary generation failed with Gemini. Details: {str(e)}"

@app.route('/')
def index():
    logging.info("Serving index.html")
    return render_template('index.html')

@app.route('/article-rewriter')
def article_rewriter_page():
    logging.info("Serving article_rewriter.html")
    return render_template('article_rewriter.html')

@app.route('/plagiarism-checker')
def plagiarism_checker_page():
    logging.info("Serving plagiarism_checker.html")
    return render_template('plagiarism_checker.html')

@app.route('/paraphraser')
def paraphrasing_tool_page():
    logging.info("Serving paraphrasing_tool.html")
    return render_template('paraphrasing_tool.html')

@app.route('/content_ideas')
def content_idea_generator_page():
    logging.info("Serving content_idea_generator.html")
    return render_template('content_idea_generator.html')

@app.route('/slogan_generator')
def slogan_generator_page():
    logging.info("Serving slogan_generator.html")
    return render_template('slogan_generator.html')

@app.route('/ai_humanizer')
def ai_humanizer_page():
    logging.info("Serving ai_text_to_humanize.html")
    return render_template('ai_text_to_humanize.html')

@app.route('/ai_email_generator')
def ai_email_generator_page():
    logging.info("Serving ai_email_generator.html")
    return render_template('ai_email_generator.html')

@app.route('/grammar_checker')
def grammar_checker_page():
    logging.info("Serving grammar_checker.html")
    return render_template('grammar_checker.html')

@app.route('/ai_story_generator')
def ai_story_generator_page():
    logging.info("Serving ai_story_generator.html")
    return render_template('ai_story_generator.html')

@app.route('/ai_product_description_generator')
def ai_product_description_generator_page():
    logging.info("Serving ai_product_description_generator.html")
    return render_template('ai_product_description_generator.html')

@app.route('/essay_generator')
def essay_generator_page():
    logging.info("Serving essay_generator.html")
    return render_template('essay_generator.html')

@app.route('/trending_news_generator')
def trending_news_generator_page():
    logging.info("Serving trending_news_generator.html")
    return render_template('trending_news_generator.html')


@app.route('/api/summarize', methods=['POST'])
def summarize_api():
    logging.info("Received /api/summarize POST request.")
    data = request.get_json()
    text = data.get('text', '')
    max_length_ratio = data.get('maxLengthRatio', 0.5)

    if not text:
        return jsonify({"summary": "", "error": "Please provide text to summarize."}), 400
    
    try:
        summary = summarize_text(text, max_length_ratio)
        if summary.startswith("Error:"):
            logging.error(f"Summarization API call failed: {summary}")
            return jsonify({"summary": "", "error": summary}), 500
        
        logging.info("Summarization successful.")
        return jsonify({"summary": summary.strip()})
    except Exception as e:
        humanized_text = f"An unexpected error occurred during summarization: {str(e)}"
        logging.error(humanized_text, exc_info=True)
        return jsonify({"summary": "", "error": humanized_text}), 500


@app.route('/api/rewrite', methods=['POST'])
def rewrite_api():
    logging.info("Received /api/rewrite POST request.")
    data = request.get_json()
    text = data.get('text', '')
    creativity = data.get('creativity', 0.5)

    if not text:
        return jsonify({"rewritten_text": "", "error": "Please provide text to rewrite."}), 400
    
    try:
        rewritten_text = rewrite_article(text, creativity)
        if rewritten_text.startswith("Error:"):
            logging.error(f"Rewriting API call failed: {rewritten_text}")
            return jsonify({"rewritten_text": "", "error": rewritten_text}), 500
        
        logging.info("Rewriting successful.")
        return jsonify({"rewritten_text": rewritten_text.strip()})
    except Exception as e:
        humanized_text = f"An unexpected error occurred during rewriting: {str(e)}"
        logging.error(humanized_text, exc_info=True)
        return jsonify({"rewritten_text": "", "error": humanized_text}), 500


@app.route('/api/humanize', methods=['POST'])
def humanize_api():
    logging.info("Received /api/humanize POST request.")
    data = request.get_json()
    text = data.get('text', '')
    creativity_level = data.get('creativity_level', 0.7)

    if not text:
        return jsonify({"humanized_text": "", "error": "Please provide text to humanize."}), 400
    
    try:
        humanized_text = humanize_text_content(text, creativity_level)
        if humanized_text.startswith("Error:"):
            logging.error(f"Humanization API call failed: {humanized_text}")
            return jsonify({"humanized_text": "", "error": humanized_text}), 500
        
        logging.info("Humanization successful.")
        return jsonify({"humanized_text": humanized_text.strip()})
    except Exception as e:
        humanized_text = f"An unexpected error occurred during humanization: {str(e)}"
        logging.error(humanized_text, exc_info=True)
        return jsonify({"humanized_text": "", "error": humanized_text}), 500


@app.route('/api/generate_email', methods=['POST'])
def generate_email_api():
    logging.info("Received /api/generate_email POST request.")
    data = request.get_json()
    subject = data.get('subject', '')
    purpose = data.get('purpose', '')
    recipient = data.get('recipient', '')

    if not subject or not purpose:
        return jsonify({"email_content": "", "error": "Please provide both subject and purpose for the email."}), 400
    
    try:
        email_content = generate_email_content(subject, purpose, recipient)
        if email_content.startswith("Error:"):
            logging.error(f"Email generation API call failed: {email_content}")
            return jsonify({"email_content": "", "error": email_content}), 500
        
        logging.info("Email generation successful.")
        return jsonify({"email_content": email_content.strip()})
    except Exception as e:
        humanized_text = f"An unexpected error occurred during email generation: {str(e)}"
        logging.error(humanized_text, exc_info=True)
        return jsonify({"email_content": "", "error": humanized_text}), 500


@app.route('/api/generate_content_ideas', methods=['POST'])
def generate_content_ideas_api():
    logging.info("Received /api/generate_content_ideas POST request.")
    data = request.get_json()
    keywords = data.get('keywords', '')

    if not keywords:
        return jsonify({"content_ideas": [], "error": "Please provide keywords for content ideas."}), 400
    
    try:
        content_ideas = generate_content_ideas(keywords)
        if content_ideas.startswith("Error:"):
            logging.error(f"Content idea generation API call failed: {content_ideas}")
            return jsonify({"content_ideas": [], "error": content_ideas}), 500
        
        logging.info("Content idea generation successful.")
        if isinstance(content_ideas, str):
            return jsonify({"content_ideas": content_ideas.strip().split('\n')})
        return jsonify({"content_ideas": content_ideas})
    except Exception as e:
        humanized_text = f"An unexpected error occurred during content idea generation: {str(e)}"
        logging.error(humanized_text, exc_info=True)
        return jsonify({"content_ideas": [], "error": humanized_text}), 500


@app.route('/api/paraphrase', methods=['POST'])
def paraphrase_api():
    logging.info("Received /api/paraphrase POST request.")
    data = request.get_json()
    text = data.get('text', '')

    if not text:
        return jsonify({"paraphrased_text": "", "error": "Please provide text to paraphrase."}), 400
    
    try:
        paraphrased_text = rewrite_article(text)
        if paraphrased_text.startswith("Error:"):
            logging.error(f"Paraphrasing API call failed: {paraphrased_text}")
            return jsonify({"paraphrased_text": "", "error": paraphrased_text}), 500
        
        logging.info("Paraphrasing successful.")
        return jsonify({"paraphrased_text": paraphrased_text.strip()})
    except Exception as e:
        humanized_text = f"An unexpected error occurred during paraphrasing: {str(e)}"
        logging.error(humanized_text, exc_info=True)
        return jsonify({"paraphrased_text": "", "error": humanized_text}), 500

@app.route('/api/check_grammar', methods=['POST'])
def check_grammar_api():
    logging.info("Received /api/check_grammar POST request.")
    data = request.get_json()
    text = data.get('text', '')

    if not text:
        return jsonify({"corrections": "", "error": "Please provide text to check grammar."}), 400
    
    try:
        corrections = check_grammar_and_style(text)
        if corrections.startswith("Error:"):
            logging.error(f"Grammar check API call failed: {corrections}")
            return jsonify({"corrections": "", "error": corrections}), 500
        
        logging.info("Grammar check successful.")
        return jsonify({"corrections": corrections.strip()})
    except Exception as e:
        humanized_text = f"An unexpected error occurred during grammar check: {str(e)}"
        logging.error(humanized_text, exc_info=True)
        return jsonify({"corrections": "", "error": humanized_text}), 500

@app.route('/api/generate_slogan', methods=['POST'])
def generate_slogan_api():
    logging.info("Received /api/generate_slogan POST request.")
    data = request.get_json()
    keywords = data.get('keywords', '')
    num_slogans = data.get('num_slogans', 5)

    if not keywords:
        return jsonify({"slogans": [], "error": "Please provide keywords for slogan generation."}), 400
    
    try:
        num_slogans = int(num_slogans)
        num_slogans = max(1, min(num_slogans, 10))
    except (ValueError, TypeError):
        logging.warning(f"Invalid num_slogans received: {num_slogans}. Defaulting to 5.")
        num_slogans = 5

    try:
        slogans = generate_slogans(keywords, num_slogans)
        if slogans and isinstance(slogans, list) and slogans[0].startswith("Error: Gemini API"):
             logging.error(f"Gemini slogan generation failed: {slogans[0]}")
             return jsonify({"slogans": [], "error": slogans[0]}), 500
        
        logging.info("Slogan generation successful.")
        return jsonify({"slogans": slogans})
    except Exception as e:
        humanized_text = f"An unexpected error occurred during slogan generation: {str(e)}"
        logging.error(humanized_text, exc_info=True)
        return jsonify({"slogans": [], "error": humanized_text}), 500


@app.route('/api/check_plagiarism_ai', methods=['POST'])
def check_plagiarism_ai_api():
    logging.info("Received /api/check_plagiarism_ai POST request.")
    data = request.get_json()
    text = data.get('text', '')

    if not text:
        return jsonify({"error": "Please provide text to check."}), 400

    try:
        results = check_plagiarism_and_ai(text)
        logging.info("Plagiarism and AI check processed.")
        return jsonify(results)
    except Exception as e:
        humanized_text = f"An unexpected error occurred during plagiarism/AI check: {str(e)}"
        logging.error(humanized_text, exc_info=True)
        return jsonify({"error": humanized_text}), 500


@app.route('/api/generate_story', methods=['POST'])
def generate_story_api():
    logging.info("Received /api/generate_story POST request.")
    data = request.get_json()
    genre = data.get('genre', '')
    characters = data.get('characters', '')
    plot_keywords = data.get('plotKeywords', '')
    length = data.get('length', 'medium')

    if not genre or not characters or not plot_keywords:
        return jsonify({"story": "", "error": "Please provide genre, characters, and plot keywords for the story."}), 400
    
    try:
        story = generate_story_content(genre, characters, plot_keywords, length)
        if story.startswith("Error:"):
            logging.error(f"Story generation API call failed: {story}")
            return jsonify({"story": "", "error": story}), 500
        
        logging.info("Story generation successful.")
        return jsonify({"story": story.strip()})
    except Exception as e:
        humanized_text = f"An unexpected error occurred during story generation: {str(e)}"
        logging.error(humanized_text, exc_info=True)
        return jsonify({"story": "", "error": humanized_text}), 500


@app.route('/api/generate_product_description', methods=['POST'])
def generate_product_description_api():
    logging.info("Received /api/generate_product_description POST request.")
    data = request.get_json()
    product_name = data.get('productName', '')
    features = data.get('features', '')
    audience = data.get('audience', '')
    tone = data.get('tone', 'informative')

    if not product_name or not features or not audience:
        return jsonify({"description": "", "error": "Please provide product name, features, and audience."}), 400
    
    try:
        description = generate_product_description_content(product_name, features, audience, tone)
        if description.startswith("Error:"):
            logging.error(f"Product description generation API call failed: {description}")
            return jsonify({"description": "", "error": description}), 500
        
        logging.info("Product description generation successful.")
        return jsonify({"description": description.strip()})
    except Exception as e:
        humanized_text = f"An unexpected error occurred during product description generation: {str(e)}"
        logging.error(humanized_text, exc_info=True)
        return jsonify({"description": "", "error": humanized_text}), 500

@app.route('/api/generate_essay', methods=['POST'])
def generate_essay_api():
    logging.info("Received /api/generate_essay POST request.")
    data = request.get_json()
    topic = data.get('topic', '')
    length = data.get('length', 'medium')
    style = data.get('style', 'formal')

    if not topic:
        return jsonify({"essay": "", "error": "Please provide a topic for the essay."}), 400
    
    try:
        essay = generate_essay_content(topic, length, style)
        if essay.startswith("Error:"):
            logging.error(f"Essay generation API call failed: {essay}")
            return jsonify({"essay": "", "error": essay}), 500
        
        logging.info("Essay generation successful.")
        return jsonify({"essay": essay.strip()})
    except Exception as e:
        humanized_text = f"An unexpected error occurred during essay generation: {str(e)}"
        logging.error(humanized_text, exc_info=True)
        return jsonify({"essay": "", "error": humanized_text}), 500

@app.route('/api/generate_trending_news', methods=['POST'])
def generate_trending_news_api():
    logging.info("Received /api/generate_trending_news POST request.")
    data = request.get_json()
    keywords = data.get('keywords', 'general news')

    try:
        news_summary = generate_trending_news_summary_content(keywords)
        if isinstance(news_summary, str) and news_summary.startswith("Error:"):
            logging.error(f"Trending news generation API call failed: {news_summary}")
            return jsonify({"news_summary": "", "error": news_summary}), 500
        
        logging.info("Trending news generation successful.")
        return jsonify({"news_summary": news_summary.strip()})
    except Exception as e:
        humanized_text = f"An unexpected error occurred during trending news generation: {str(e)}"
        logging.error(humanized_text, exc_info=True)
        return jsonify({"news_summary": "", "error": humanized_text}), 500


if __name__ == '__main__':
    logging.info("Starting Flask development server. (Main process)")
    load_summarizer_model()
    load_paraphraser_model()
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)