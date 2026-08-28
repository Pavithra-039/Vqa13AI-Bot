import requests
import os

API_URL = "https://router.huggingface.co/hf-inference/models/dandelin/vilt-b32-finetuned-vqa"

def predict_answer(image_bytes, question):
    token = os.environ.get("HF_TOKEN")

    headers = {
        "Authorization": f"Bearer {token}"
    }

    files = {
        "file": ("image.jpg", image_bytes, "image/jpeg")
    }

    data = {
        "question": question
    }

    response = requests.post(
        API_URL,
        headers=headers,
        files=files,
        data=data,
        timeout=120
    )

    result = response.json()

    if isinstance(result, list) and len(result) > 0:
        return result[0].get("answer", "No answer found")

    if isinstance(result, dict) and "error" in result:
        return "Model error: " + result["error"]

    return "Unable to generate answer"
