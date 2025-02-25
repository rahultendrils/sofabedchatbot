# SofaBed.com Chatbot

A chatbot for SofaBed.com using FastAPI, Streamlit, and OpenAI.

## Environment Variables

Required environment variables:
- OPENAI_API_KEY: Your OpenAI API key
- BACKEND_URL: URL of the FastAPI backend (for Streamlit deployment)

## Local Development

1. Clone the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate the virtual environment
4. Install dependencies: `pip install -r requirements.txt`
5. Create a `.env` file with your OPENAI_API_KEY
6. Run the backend: `uvicorn main:app --reload`
7. Run the frontend: `streamlit run streamlit_app.py` 