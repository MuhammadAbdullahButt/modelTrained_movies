# Movie Query API
This project is a Flask-based web application that allows users to query a movie dataset using natural language. The app supports queries like "movies before 2000" or "movies directed by Christopher Nolan." It uses natural language processing (NLP) techniques and machine learning to preprocess data, extract entities, and match user queries with relevant movies.

## Features
- **Named Entity Recognition (NER):** Extracts entities like dates, actors, genres, and directors from user queries.
- **Custom Query Handling:** Supports filtering movies based on release year, cast, genre, and director.
- **TF-IDF Vectorizer:** Enhances text-based similarity matching for fallback query resolution.
- **Data Preprocessing:** Cleans and preprocesses text data from the dataset to improve query matching.
- **Web Interface:** Simple web interface for sending queries and viewing results.

## Technologies Used
- **Backend Framework:** Flask
- **Natural Language Processing:** NLTK
- **Machine Learning:** Scikit-learn (TF-IDF and cosine similarity)
- **Data Processing:** Pandas
- **Frontend:** HTML (via Flask templates)

## Prerequisites
1. **Python (>= 3.8):** Ensure you have Python installed.
2. **Virtual Environment:** It is recommended to use a virtual environment for dependency management.
   
## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/movie-query-api.git
   cd movie-query-api

