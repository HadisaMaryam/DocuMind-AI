
import os
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq
from groq import APIError, AuthenticationError, RateLimitError

# 1. Load .env from the project folder
env_path = Path(__file__).with_name(".env")
load_dotenv(dotenv_path=env_path)

# 2. Get API key
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    print("ERROR: Groq API key not found.")
    print("Check that your .env file is in the project folder.")
    raise SystemExit(1)

# 3. Connect to Groq
client = Groq(api_key=api_key)

try:
    print("Connecting to Groq AI...")

    # 4. Send a test message
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful AI assistant."
            },
            {
                "role": "user",
                "content": "Hello! Introduce yourself in one sentence."
            }
        ],
        max_completion_tokens=200
    )

    # 5. Display response
    print("\nSUCCESS! Groq API is connected.")
    print("\nAI Response:")
    print(response.choices[0].message.content)

except AuthenticationError:
    print("ERROR: Invalid API key. Check your .env file.")

except RateLimitError:
    print("ERROR: Groq API rate limit reached. Try again later.")

except APIError as e:
    print(f"Groq API Error: {e}")

except Exception as e:
    print(f"Unexpected Error: {e}")
