import requests
import time
start_time = time.perf_counter()
url = "http://localhost:8000/ask"

data = {
    "question": "What is the police arrest procedure?",
    "top_summary_docs": 3,
    "top_k_per_doc": 10,
    
}

response = requests.post(url, json=data)

if response.status_code == 200:
    result = response.json()
    print(result)
else:
    print("Error:", response.status_code, response.text)
end_time = time.perf_counter()
print(f"Request took {end_time - start_time:.2f} seconds")
