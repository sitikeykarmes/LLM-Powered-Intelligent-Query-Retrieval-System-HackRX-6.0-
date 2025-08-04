# Test OpenRouter connection first
from dotenv import load_dotenv
load_dotenv()
import os
from openai import OpenAI

api_key = os.getenv('OPENROUTER_API_KEY')
if api_key:
    print('✅ API Key found:', api_key[:15] + '...')
    client = OpenAI(api_key=api_key, base_url='https://openrouter.ai/api/v1')
    try:
        response = client.chat.completions.create(
            model='moonshotai/kimi-k2:free',
            messages=[{'role': 'user', 'content': 'What is the capital of France?'}],
            max_tokens=10
        )
        print('✅ OpenRouter test successful:', response.choices[0].message.content)
    except Exception as e:
        print('❌ OpenRouter test failed:', e)
else:
    print('❌ API Key not found')
