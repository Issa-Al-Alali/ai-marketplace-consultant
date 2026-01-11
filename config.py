import os
from dotenv import load_dotenv

load_dotenv()

# Get API key from environment variable
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# MODEL_NAME = "gemini-2.5-flash"  # Flash is recommended for bulk analysis
# if not GOOGLE_API_KEY:
#     raise ValueError("GOOGLE_API_KEY not found in .env file. Please create a .env file with your API key.")

# Get Groq API key from environment variable
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Model configuration for Groq
MODEL_NAME = "llama-3.3-70b-versatile" 
# Validate API key
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env file. Please create a .env file with your Groq API key.")
