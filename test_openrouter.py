"""Test script for OpenRouter API with Trinity model."""

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import httpx


async def test_openrouter_api():
    """Test OpenRouter API with Trinity model."""
    
    api_key = os.getenv("OPEN_ROUTER")
    if not api_key:
        print("❌ OPEN_ROUTER not found in .env")
        return
    
    print(f"✅ API Key found: {api_key[:15]}...")
    
    # OpenRouter API endpoint
    url = "https://openrouter.ai/api/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8000",  # Optional
        "X-Title": "Voice RAG Agent"  # Optional
    }
    
    # Test questions
    questions = [
        "What is 2 + 2? Answer in one word.",
        "Name 3 colors. Be brief.",
        "What is the capital of France? One word answer.",
    ]
    
    print("\n🧪 Testing OpenRouter API with Trinity model...\n")
    print("Model: arcee-ai/trinity-large-preview:free")
    print("-" * 60)
    
    success_count = 0
    error_count = 0
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        for i, question in enumerate(questions, 1):
            print(f"\n📝 Question {i}: {question}")
            
            payload = {
                "model": "arcee-ai/trinity-large-preview:free",
                "messages": [
                    {
                        "role": "user",
                        "content": question
                    }
                ],
                "max_tokens": 100,
                "temperature": 0.7
            }
            
            try:
                response = await client.post(url, headers=headers, json=payload)
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data["choices"][0]["message"]["content"]
                    answer = answer[:100] + "..." if len(answer) > 100 else answer
                    print(f"✅ Response: {answer}")
                    success_count += 1
                elif response.status_code == 429:
                    print(f"⚠️  Rate Limited (429)")
                    error_count += 1
                else:
                    print(f"❌ Error {response.status_code}: {response.text[:100]}")
                    error_count += 1
                    
            except Exception as e:
                print(f"❌ Error: {str(e)[:100]}")
                error_count += 1
            
            await asyncio.sleep(0.5)
    
    print("\n" + "-" * 60)
    print(f"\n📊 Results: {success_count} success, {error_count} errors")
    
    if success_count > 0:
        print("\n🎉 OpenRouter API is working! Ready to integrate.")
    else:
        print("\n💡 Check your API key and try again.")


if __name__ == "__main__":
    asyncio.run(test_openrouter_api())
