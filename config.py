import os
from dotenv import load_dotenv

load_dotenv()

# Get API key from environment variable
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Model configuration
MODEL_NAME = "gemini-2.5-flash"  # Flash is recommended for bulk analysis

# Validate API key
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file. Please create a .env file with your API key.")
