import pymongo;
import requests;
import os
from dotenv import load_dotenv
load_dotenv()
CONNECTION = os.getenv("CONNECTION")
HF_TOKEN = os.getenv("HF_TOKEN")
client = pymongo.MongoClient(CONNECTION)
db = client.sample_mflix
collection = db.movies
hf_token =  HF_TOKEN
embedding_url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
def generate_embedding(text: str) -> list[float]:
  response = requests.post(
    embedding_url,
    headers={"Authorization": f"Bearer {hf_token}"},
    json={"inputs": text})
 
  if response.status_code != 200:
    raise ValueError(f"Request failed with status code {response.status_code}: {response.text}")
  
  return response.json()
# for doc in collection.find({'plot':{"$exists": True}}).limit(50):
#   doc['plot_embedding_hf'] = generate_embedding(doc['plot'])
#   collection.replace_one({'_id': doc['_id']}, doc)

query = "The War of world"
embed = generate_embedding(query)
results = collection.aggregate([
  {"$vectorSearch": {
    "queryVector": embed,
    "path": "plot_embedding_hf",
    "numCandidates": 100,
    "limit": 4,
    "index": "plotSemanticSearch",
      }}
]);
for document in results:
    print(f'Movie Name: {document["title"]},\nMovie Plot: {document["plot"]}\n')