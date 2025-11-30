import requests
import csv

BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAAIHk5gEAAAAAxPX4strBaDZ3t4frI47D5wXfU5o%3DttCFchDlEORtH88u1uyBpZ5BnrxRzZkICcdZdofRkzGWnfKnOw"

headers = {"Authorization": f"Bearer {BEARER_TOKEN}"}
url = "https://api.x.com/2/tweets/search/recent?query=banjir&max_results=10"

response = requests.get(url, headers=headers).json()
print(response)
with open("tweets_banjir.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "text"])

    for t in response["data"]:
        writer.writerow([t["id"], t["text"]])