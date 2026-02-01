"""Simple script to test Gemini API and check rate limits."""

import asyncio
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from google import genai


async def test_gemini_api():
    """Test Gemini API with sample questions."""
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY not found in .env")
        return
    
    print(f"✅ API Key found: {api_key[:10]}...")
    
    # Initialize client
    client = genai.Client(api_key=api_key)
    
    # Test questions
    questions = [
        "What is 2 + 2?",
        "Name 3 colors.",
        "What is the capital of France?",
        "Say hello in Spanish.",
        "What is the speed of light?",
    ]
    
    print("\n🧪 Testing Gemini API with simple questions...\n")
    print("-" * 60)
    
    success_count = 0
    error_count = 0
    
    for i, question in enumerate(questions, 1):
        print(f"\n📝 Question {i}: {question}")
        
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=question
            )
            
            answer = response.text[:100] + "..." if len(response.text) > 100 else response.text
            print(f"✅ Response: {answer}")
            success_count += 1
            
        except Exception as e:
            error_str = str(e)
            if "429" in error_str:
                print(f"⚠️  Rate Limited (429) - Quota exceeded")
            else:
                print(f"❌ Error: {error_str[:100]}")
            error_count += 1
        
        # Small delay between requests
        await asyncio.sleep(0.5)
    
    print("\n" + "-" * 60)
    print(f"\n📊 Results: {success_count} success, {error_count} errors")
    
    if error_count > 0:
        print("\n💡 If you see rate limit errors (429):")
        print("   - Wait a few minutes for the quota to reset")
        print("   - Free tier allows ~15 requests/minute")
        print("   - Consider upgrading to a paid plan for higher limits")
    else:
        print("\n🎉 All requests successful! API is working correctly.")


if __name__ == "__main__":
    asyncio.run(test_gemini_api())
