# PDF Q&A Chatbot (Google Gemini)

This project is a Streamlit-based chatbot that allows you to upload a PDF and ask questions about its content using Google Gemini and FAISS for vector search.

## Features
- Upload a PDF and extract its text
- Chunk and embed text using Google Generative AI Embeddings
- Store and search embeddings with FAISS
- Chat interface powered by Streamlit
- Uses Google Gemini LLM for answering questions

## Setup

1. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

2. **Set your Google API Key:**
    - Create a file named `.env` in the project root with the following content:
       ```env
       GOOGLE_API_KEY=your-google-api-key-here
       ```
   - The app uses [python-dotenv](https://pypi.org/project/python-dotenv/) to load this key automatically.

3. **Run the app:**
   ```sh
   streamlit run chatbot.py
   ```

## File Overview
- `chatbot.py`: Main Streamlit app for PDF Q&A chatbot
- `requirements.txt`: Python dependencies

## Notes
- Ensure your API key has access to the required Google Gemini models.
- For GPU acceleration, ensure your environment supports `faiss-gpu`.  
   **Note:** `faiss-gpu` is not supported on macOS; use the CPU version (`faiss-cpu`) instead.
