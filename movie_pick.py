import time
import openai
from pymongo.mongo_client import MongoClient
from datetime import datetime
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv, find_dotenv
import os
import warnings
warnings.filterwarnings('ignore')


# Set up the environment
_ = load_dotenv(find_dotenv())
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
MONGO_URI = os.environ.get("MONGO_URI")
openai.api_key = OPENAI_API_KEY

# Dataset
dataset = load_dataset(
    "Pablinho/movies-dataset",
    streaming=True, split="train"
)
dataset = dataset.take(200)
dataset_df = pd.DataFrame(dataset)
dataset_df

# Document Modelling
class Movie(BaseModel):
    Release_Date: Optional[datetime]
    Title: str
    Overview: str
    Popularity: float
    Vote_Count: int
    Vote_Average: float
    Original_Language: str
    Genre: List[str]
    Poster_Url: str
    text_embeddings: Optional[List[float]] = None

# Embedding Generation
def get_embedding(text):
    if not text or not isinstance(text, str):
        return None
    try:
        embedding = openai.embeddings.create(
            input=text,
            model="text-embedding-3-small", dimensions=1536).data[0].embedding
        return embedding
    except Exception as e:
        print(f"Error in get_embedding: {e}")
        return None


def process_and_embed_record(record):
    for key, value in record.items():
        if pd.isnull(value):
            record[key] = None

    if record['Genre']:
        record['Genre'] = record['Genre'].split(', ')
    else:
        record['Genre'] = []

    text_to_embed = f"{record['Title']} {record['Overview']}"
    embedding = get_embedding(text_to_embed)
    record['text_embeddings'] = embedding
    return record


print("Processing records and generating embeddings...")
records = [
    process_and_embed_record(record)
    for record in dataset_df.to_dict(orient='records')
]
print("Embedding generation completed")

# Database Creation and Connection


def get_mongo_client(mongo_uri):
    client = MongoClient(mongo_uri, appname="pmr.movies")
    print("Connection to MongoDB successful")
    return client


mongo_client = get_mongo_client(MONGO_URI)
database_name = "movies_dataset"
collection_name = "movies"
db = mongo_client.get_database(database_name)
collection = db.get_collection(collection_name)

collection.delete_many({})

# Data Ingestion
movies = [Movie(**record).model_dump() for record in records]
collection.insert_many(movies)
print("Data ingestion completed")

# Vector Search Function


def vector_search(user_query, db, collection, vector_index="vector_index_text", max_retries=3):
    query_embedding = get_embedding(user_query)
    if query_embedding is None:
        return "Invalid query or embedding generation failed."

    vector_search_stage = {
        "$vectorSearch": {
            "index": vector_index,
            "queryVector": query_embedding,
            "path": "text_embeddings",
            "numCandidates": 150,
            "limit": 20
        }
    }

    pipeline = [vector_search_stage]

    for attempt in range(max_retries):
        try:
            results = list(collection.aggregate(pipeline))
            if results:
                explain_query_execution = db.command(
                    'explain', {
                        'aggregate': collection.name,
                        'pipeline': pipeline,
                        'cursor': {}
                    },
                    verbosity='executionStats')
                vector_search_explain = explain_query_execution['stages'][0]['$vectorSearch']
                millis_elapsed = vector_search_explain['explain']['collectStats']['millisElapsed']
                print(f"Total time for the execution to complete on the database server: {
                      millis_elapsed} milliseconds")
                return results
            else:
                print(f"No results found on attempt {
                      attempt + 1}. Retrying...")
                time.sleep(2)
        except Exception as e:
            print(f"Error on attempt {attempt + 1}: {str(e)}")
            time.sleep(2)

    return "Failed to retrieve results after multiple attempts."

# Handling User Query


class SearchResultItem(BaseModel):
    Title: str
    Overview: str
    Genre: List[str]
    Vote_Average: float
    Popularity: float


def handle_user_query(query, db, collection):

    time.sleep(2)
    get_knowledge = vector_search(query, db, collection)

    if isinstance(get_knowledge, str):  # Error message
        return get_knowledge, "No source information available."

    search_results_models = [SearchResultItem(
        **result) for result in get_knowledge]
    search_results_df = pd.DataFrame(
        [item.dict() for item in search_results_models])

    completion = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a movie recommendation system."},
            {"role": "user", "content": f"Answer this user query: {
                query} with the following context:\n{search_results_df}"}
        ]
    )

    system_response = completion.choices[0].message.content

    print(f"- User Question:\n{query}\n")
    print(f"- System Response:\n{system_response}\n")

    return system_response


# Query Example
query = """
What is a good animated film that adults would enjoy too? What makes your suggestion a good fit?
"""
response = handle_user_query(query, db, collection)
