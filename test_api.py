import requests
import argparse
import sys

def test_endpoints(base_url, image_path, image_url):
    print(f"Testing API at {base_url}")
    
    # Test Data
    files = {'file': open(image_path, 'rb')}
    
    # 1. Test VIT POST
    print("\n--- Testing VIT POST ---")
    try:
        files['file'].seek(0)
        res = requests.post(f"{base_url}/interrogate/vit", files=files, data={"mode": "fast"})
        print(f"Status: {res.status_code}")
        print(f"Response: {res.json()}")
    except Exception as e:
        print(f"Failed: {e}")

    # 2. Test VIT GET
    print("\n--- Testing VIT GET ---")
    try:
        res = requests.get(f"{base_url}/interrogate/vit", params={"url": image_url, "mode": "fast"})
        print(f"Status: {res.status_code}")
        print(f"Response: {res.json()}")
    except Exception as e:
        print(f"Failed: {e}")

    # 3. Test EVA POST
    print("\n--- Testing EVA POST ---")
    try:
        files['file'].seek(0)
        res = requests.post(f"{base_url}/interrogate/eva", files=files, data={"mode": "fast"})
        print(f"Status: {res.status_code}")
        print(f"Response: {res.json()}")
    except Exception as e:
        print(f"Failed: {e}")

    # 4. Test EVA GET
    print("\n--- Testing EVA GET ---")
    try:
        res = requests.get(f"{base_url}/interrogate/eva", params={"url": image_url, "mode": "fast"})
        print(f"Status: {res.status_code}")
        print(f"Response: {res.json()}")
    except Exception as e:
        print(f"Failed: {e}")

    # 5. Test PixAI POST
    print("\n--- Testing PixAI POST ---")
    try:
        files['file'].seek(0)
        res = requests.post(f"{base_url}/interrogate/pixai", files=files, data={"threshold": 0.35})
        print(f"Status: {res.status_code}")
        # Truncate tags for display
        json_res = res.json()
        if "tag_string" in json_res:
            json_res["tag_string"] = json_res["tag_string"][:50] + "..."
        print(f"Response: {json_res}")
    except Exception as e:
        print(f"Failed: {e}")

    # 6. Test PixAI GET
    print("\n--- Testing PixAI GET ---")
    try:
        res = requests.get(f"{base_url}/interrogate/pixai", params={"url": image_url, "threshold": 0.35})
        print(f"Status: {res.status_code}")
        json_res = res.json()
        if "tag_string" in json_res:
            json_res["tag_string"] = json_res["tag_string"][:50] + "..."
        print(f"Response: {json_res}")
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--image", required=True, help="Path to local image for POST tests")
    parser.add_argument("--image-url", default="https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/481px-Cat03.jpg", help="URL of image for GET tests")
    args = parser.parse_args()
    
    test_endpoints(args.url, args.image, args.image_url)
