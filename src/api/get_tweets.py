import requests
import csv
import os

BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAAIHk5gEAAAAAxPX4strBaDZ3t4frI47D5wXfU5o%3DttCFchDlEORtH88u1uyBpZ5BnrxRzZkICcdZdofRkzGWnfKnOw"

def fetch_banjir_tweets():
    print("Working directory:", os.getcwd())

    os.makedirs("data/raw", exist_ok=True)

    headers = {"Authorization": f"Bearer {BEARER_TOKEN}"}

    url = "https://api.twitter.com/2/tweets/search/recent"

    query = 'banjir OR kebanjiran OR "hujan deras" OR "sungai meluap" OR genangan'

    params = {
        "query": query,
        "max_results": 10,
        "tweet.fields": "created_at,lang"
    }

    response = requests.get(url, headers=headers, params=params).json()

    print("API Response:", response)

    if "data" not in response:
        print(" Tidak ada data ditemukan.")
        return

    file_path = os.path.join("data", "raw", "tweets_banjir.csv")

    with open(file_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "text"])
        for t in response["data"]:
            writer.writerow([t["id"], t["text"]])

    print("✔ Data disimpan ke", file_path)

fetch_banjir_tweets()
