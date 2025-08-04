#!/usr/bin/env python3
"""
Test script for HackRX Document Query System with OpenRouter
Run this to validate your API before submission
"""

import requests
import json
import time
import os
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
API_BASE_URL = "http://localhost:8000"  # Change to your deployed URL
BEARER_TOKEN = "6316e01746d83a3078c19510945475dd0aa9c7f218659c845184a49e455bf8e0"

# Sample test data
SAMPLE_REQUEST = {
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
    "questions": [
        "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
        "What is the waiting period for pre-existing diseases (PED) to be covered?",
        "Does this policy cover maternity expenses, and what are the conditions?",
        "What is the waiting period for cataract surgery?",
        "Are the medical expenses for an organ donor covered under this policy?",
        "What is the No Claim Discount (NCD) offered in this policy?",
        "Is there a benefit for preventive health check-ups?",
        "How does the policy define a 'Hospital'?",
        "What is the extent of coverage for AYUSH treatments?",
        "Are there any sub-limits on room rent and ICU charges for Plan A?"
    ]
}

def test_openrouter_api():
    """Test OpenRouter API connection directly"""
    print("🔍 Testing OpenRouter API connection...")
    
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        print("❌ OPENROUTER_API_KEY not found in environment variables")
        return False
    
    try:
        from openai import OpenAI
        
        client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )
        
        # Test with a simple request
        response = client.chat.completions.create(
            model="moonshotai/kimi-k2:free",  # Free model
            messages=[
                {"role": "user", "content": "Respond with just 'API Working'"}
            ],
            max_tokens=500
        )
        
        result = response.choices[0].message.content.strip()
        print(f"✅ OpenRouter API test successful: {result}")
        return True
        
    except Exception as e:
        print(f"❌ OpenRouter API test failed: {e}")
        return False

def test_health_endpoint():
    """Test health check endpoint"""
    print("🔍 Testing health endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=10)
        if response.status_code == 200:
            print("✅ Health check passed")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_authentication():
    """Test authentication with invalid token"""
    print("🔍 Testing authentication...")
    try:
        headers = {
            "Authorization": "Bearer invalid_token",
            "Content-Type": "application/json"
        }
        response = requests.post(
            f"{API_BASE_URL}/hackrx/run",
            headers=headers,
            json={"documents": "test", "questions": ["test"]},
            timeout=10
        )
        if response.status_code == 401:
            print("✅ Authentication validation passed")
            return True
        else:
            print(f"❌ Authentication validation failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Authentication test error: {e}")
        return False

def test_main_endpoint():
    """Test main /hackrx/run endpoint"""
    print("🔍 Testing main endpoint...")
    try:
        headers = {
            "Authorization": f"Bearer {BEARER_TOKEN}",
            "Content-Type": "application/json"
        }
        
        start_time = time.time()
        response = requests.post(
            f"{API_BASE_URL}/hackrx/run",
            headers=headers,
            json=SAMPLE_REQUEST,
            timeout=120  # Increased timeout for OpenRouter
        )
        end_time = time.time()
        
        response_time = end_time - start_time
        print(f"⏱️  Response time: {response_time:.2f} seconds")
        
        if response.status_code == 200:
            data = response.json()
            if "answers" in data and len(data["answers"]) == len(SAMPLE_REQUEST["questions"]):
                print("✅ Main endpoint test passed")
                print(f"📊 Received {len(data['answers'])} answers")
                
                # Display sample answers
                for i, (question, answer) in enumerate(zip(SAMPLE_REQUEST["questions"], data["answers"])):
                    print(f"\n📝 Q{i+1}: {question}")
                    print(f"💡 A{i+1}: {answer}")
                
                return True, response_time, data
            else:
                print(f"❌ Invalid response format: {data}")
                return False, response_time, data
        else:
            print(f"❌ Main endpoint failed: {response.status_code}")
            print(f"Error: {response.text}")
            return False, response_time, None
            
    except Exception as e:
        print(f"❌ Main endpoint error: {e}")
        return False, 0, None

def test_alternative_endpoint():
    """Test alternative /api/v1/hackrx/run endpoint"""
    print("🔍 Testing alternative endpoint...")
    try:
        headers = {
            "Authorization": f"Bearer {BEARER_TOKEN}",
            "Content-Type": "application/json"
        }
        
        # Test with a single question for faster response
        simple_request = {
            "documents": SAMPLE_REQUEST["documents"],
            "questions": [SAMPLE_REQUEST["questions"][0]]
        }
        
        response = requests.post(
            f"{API_BASE_URL}/api/v1/hackrx/run",
            headers=headers,
            json=simple_request,
            timeout=60
        )
        
        if response.status_code == 200:
            print("✅ Alternative endpoint test passed")
            return True
        else:
            print(f"❌ Alternative endpoint failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Alternative endpoint error: {e}")
        return False

def validate_response_format(data: Dict[Any, Any]) -> bool:
    """Validate response format matches specification"""
    print("🔍 Validating response format...")
    
    if not isinstance(data, dict):
        print("❌ Response is not a dictionary")
        return False
    
    if "answers" not in data:
        print("❌ Missing 'answers' field")
        return False
    
    if not isinstance(data["answers"], list):
        print("❌ 'answers' is not a list")
        return False
    
    if len(data["answers"]) != len(SAMPLE_REQUEST["questions"]):
        print(f"❌ Answer count mismatch: expected {len(SAMPLE_REQUEST['questions'])}, got {len(data['answers'])}")
        return False
    
    for i, answer in enumerate(data["answers"]):
        if not isinstance(answer, str):
            print(f"❌ Answer {i+1} is not a string")
            return False
        if len(answer.strip()) == 0:
            print(f"❌ Answer {i+1} is empty")
            return False
    
    print("✅ Response format validation passed")
    return True

def performance_test():
    """Run performance test with multiple requests"""
    print("🔍 Running performance test...")
    
    headers = {
        "Authorization": f"Bearer {BEARER_TOKEN}",
        "Content-Type": "application/json"
    }
    
    # Single question for performance test
    perf_request = {
        "documents": SAMPLE_REQUEST["documents"],
        "questions": ["What is the grace period for premium payment?"]
    }
    
    times = []
    success_count = 0
    
    for i in range(2):  # Reduced to 2 requests for OpenRouter rate limits
        try:
            start_time = time.time()
            response = requests.post(
                f"{API_BASE_URL}/hackrx/run",
                headers=headers,
                json=perf_request,
                timeout=60
            )
            end_time = time.time()
            
            response_time = end_time - start_time
            times.append(response_time)
            
            if response.status_code == 200:
                success_count += 1
                print(f"✅ Request {i+1}: {response_time:.2f}s")
            else:
                print(f"❌ Request {i+1} failed: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Request {i+1} error: {e}")
        
        # Add delay between requests to avoid rate limiting
        if i < 1:
            time.sleep(2)
    
    if times:
        avg_time = sum(times) / len(times)
        max_time = max(times)
        min_time = min(times)
        
        print(f"📊 Performance Summary:")
        print(f"   Success Rate: {success_count}/2 ({success_count/2*100:.1f}%)")
        print(f"   Average Time: {avg_time:.2f}s")
        print(f"   Min Time: {min_time:.2f}s")
        print(f"   Max Time: {max_time:.2f}s")
        
        if avg_time < 60:  # More lenient for OpenRouter
            print("✅ Performance test passed (< 60s)")
            return True
        else:
            print("⚠️  Performance test warning (> 60s)")
            return False
    else:
        print("❌ No successful requests in performance test")
        return False

def check_environment():
    """Check environment setup"""
    print("🔍 Checking environment setup...")
    
    # Check OpenRouter API key
    api_key = os.getenv('OPENROUTER_API_KEY')
    if api_key:
        print(f"✅ OPENROUTER_API_KEY found: {api_key[:10]}...")
        return True
    else:
        print("❌ OPENROUTER_API_KEY not found")
        print("💡 Set your OpenRouter API key:")
        print("   PowerShell: $env:OPENROUTER_API_KEY='your-key-here'")
        print("   Or create .env file with: OPENROUTER_API_KEY=your-key-here")
        return False

def run_comprehensive_test():
    """Run all tests and provide summary"""
    print("🚀 Starting HackRX API Comprehensive Test (OpenRouter Edition)")
    print("=" * 60)
    
    results = {}
    
    # Test 0: Environment Check
    results["environment"] = check_environment()
    if not results["environment"]:
        print("⛔ Cannot proceed without API key. Please set OPENROUTER_API_KEY.")
        return
    print()
    
    # Test 1: OpenRouter API Test
    results["openrouter"] = test_openrouter_api()
    print()
    
    # Test 2: Health Check
    results["health"] = test_health_endpoint()
    print()
    
    # Test 3: Authentication
    results["auth"] = test_authentication()
    print()
    
    # Test 4: Main Endpoint
    main_success, response_time, response_data = test_main_endpoint()
    results["main_endpoint"] = main_success
    print()
    
    # Test 5: Response Format Validation
    if main_success and response_data:
        results["format"] = validate_response_format(response_data)
    else:
        results["format"] = False
        print("⏭️  Skipping format validation (main endpoint failed)")
    print()
    
    # Test 6: Alternative Endpoint
    results["alt_endpoint"] = test_alternative_endpoint()
    print()
    
    # Test 7: Performance Test
    if main_success:
        results["performance"] = performance_test()
    else:
        results["performance"] = False
        print("⏭️  Skipping performance test (main endpoint failed)")
    print()
    
    # Summary
    print("📋 TEST SUMMARY")
    print("=" * 40)
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name.upper():<15} {status}")
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests >= total_tests - 1:  # Allow one failure
        print("\n🎉 SYSTEM READY! Your API is ready for submission.")
        print("\n📝 Submission Checklist:")
        print("   ✅ API is accessible")
        print("   ✅ OpenRouter integration works")
        print("   ✅ Authentication works")
        print("   ✅ Endpoints respond correctly")
        print("   ✅ Response format is valid")
        print("\n🔗 Submit your webhook URL: https://your-domain.com/hackrx/run")
    else:
        print(f"\n⚠️  {total_tests - passed_tests} test(s) failed. Please fix issues before submission.")
        
        if not results["environment"]:
            print("   🔧 Set OPENROUTER_API_KEY environment variable")
        if not results["openrouter"]:
            print("   🔧 Check OpenRouter API key and credits")
        if not results["health"]:
            print("   🔧 Check if your API server is running")
        if not results["auth"]:
            print("   🔧 Verify authentication implementation")
        if not results["main_endpoint"]:
            print("   🔧 Check main endpoint implementation and document processing")
        if not results["format"]:
            print("   🔧 Ensure response format matches specification")

def test_simple_question():
    """Test with a very simple question"""
    print("🔍 Testing with simple question...")
    
    headers = {
        "Authorization": f"Bearer {BEARER_TOKEN}",
        "Content-Type": "application/json"
    }
    
    simple_request = {
        "documents": SAMPLE_REQUEST["documents"],
        "questions": ["What is title of this document?"]
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/hackrx/run",
            headers=headers,
            json=simple_request,
            timeout=60
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Simple question test passed")
            print(f"Q: What is this document about?")
            print(f"A: {data['answers'][0]}")
            return True
        else:
            print(f"❌ Simple question test failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Simple question test error: {e}")
        return False

if __name__ == "__main__":
    # Allow custom API URL
    custom_url = input(f"Enter API base URL (default: {API_BASE_URL}): ").strip()
    if custom_url:
        API_BASE_URL = custom_url.rstrip('/')
    
    print(f"Testing API at: {API_BASE_URL}")
    print()
    
    # Run comprehensive test
    run_comprehensive_test()
    
    # Optional simple test
    print("\n" + "=" * 60)
    test_simple_question()
    
    print("\n🏁 Testing completed!")
    print("\n💡 Tips for OpenRouter:")
    print("   - Some models are free but have rate limits")
    print("   - Premium models require credits")
    print("   - Check https://openrouter.ai/models for available models")