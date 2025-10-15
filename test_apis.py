#!/usr/bin/env python3
"""Quick API test"""

import os
from dotenv import load_dotenv

load_dotenv()

print("\n🧪 Testing API Clients\n")

# Test Anthropic
try:
    import anthropic
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    response = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=50,
        messages=[{"role": "user", "content": "Say 'hello' in one word"}]
    )
    print(f"✅ Anthropic works: {response.content[0].text}")
except Exception as e:
    print(f"❌ Anthropic failed: {e}")

# Test OpenAI
try:
    import openai
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4-0125-preview",
        max_tokens=50,
        messages=[{"role": "user", "content": "Say 'hello' in one word"}]
    )
    print(f"✅ OpenAI works: {response.choices[0].message.content}")
except Exception as e:
    print(f"❌ OpenAI failed: {e}")

# Test Gemini
try:
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel('gemini-2.5-flash')
    response = model.generate_content("Say 'hello' in one word")
    print(f"✅ Gemini works: {response.text}")
except Exception as e:
    print(f"❌ Gemini failed: {e}")