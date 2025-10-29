"""Quick test script for PhishGuard API."""
import requests
import time

API_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint."""
    print("Testing /health endpoint...")
    response = requests.get(f"{API_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_predict():
    """Test predict endpoint."""
    print("Testing /predict endpoint...")
    
    # Test with phishing email
    phishing_email = {
        "subject": "URGENT: Account Verification Required!",
        "body": "Your account will be suspended. Click here: http://192.168.1.1/verify"
    }
    
    response = requests.post(
        f"{API_URL}/predict",
        json=phishing_email,
        headers={"X-API-Key": "your-secret-api-key-here"}
    )
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Label: {data['label']}")
        print(f"Confidence: {data['confidence']:.2%}")
        print(f"Phishing Score: {data['phishing_score']:.2%}")
        print(f"Explanation: {data['explanation']}")
        print(f"Features: {data['features']}")
    else:
        print(f"Error: {response.text}")
    print()
    
    # Test with benign email
    benign_email = {
        "subject": "Team Meeting Tomorrow",
        "body": "Hi everyone, let's meet tomorrow at 10 AM to discuss the project."
    }
    
    response = requests.post(
        f"{API_URL}/predict",
        json=benign_email,
        headers={"X-API-Key": "your-secret-api-key-here"}
    )
    
    print(f"Benign Email Test:")
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Label: {data['label']}")
        print(f"Confidence: {data['confidence']:.2%}")
    print()

if __name__ == "__main__":
    print("="*70)
    print("PhishGuard API Test")
    print("="*70)
    print()
    
    # Wait for server to be ready
    print("Waiting for server to be ready...")
    max_retries = 10
    for i in range(max_retries):
        try:
            response = requests.get(f"{API_URL}/health", timeout=2)
            if response.status_code == 200:
                print("Server is ready!")
                print()
                break
        except requests.exceptions.RequestException:
            if i == max_retries - 1:
                print("Server not responding. Please start it first:")
                print("  python scripts/start_api.py --reload")
                exit(1)
            time.sleep(1)
    
    test_health()
    test_predict()
    
    print("="*70)
    print("All tests completed!")
    print("="*70)
