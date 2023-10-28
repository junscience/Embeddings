import requests

model_id = "sentence-transformers/all-MiniLM-L6-v2"
hf_token = "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"

api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
headers = {"Authorization": f"Bearer {hf_token}"}


def query(texts):
    response = requests.post(api_url, headers=headers, json={"inputs": texts, "options": {"wait_for_model": True}})
    return response.json()
