import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

user_input = input("Ask me anything: ")

try:
    response = client.responses.create(
        model="gpt-5.6-luna",
        input=user_input
    )

    print("\nAI:", response.output_text)

except Exception as e:
    print("\nAn error occurred:", e)