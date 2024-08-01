# Movie Recommendation System using RAG Pipeline + MongoDB

This project implements a movie recommendation system using a Retrieval-Augmented Generation (RAG) pipeline. It combines the power of large language models with efficient vector search capabilities of MongoDB.

## Features

- Movie data processing and embedding generation
- Efficient vector search using MongoDB
- Natural language query processing
- Personalized movie recommendations with explanations

## Prerequisites

- Python 3.8+
- MongoDB Atlas account
- MongoDB URI
- OpenAI API key

## Quick Start

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up your MongoDB Atlas cluster and obtain connection string
4. Set environment variables for MongoDB URI and OpenAI API key
5. Run the main script: `python main.py`

## Usage

```python
query = "I'm looking for a sci-fi movie with plot twists. Explain your recommendation."
result = handle_user_query(query, db, collection)
print(result)
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
