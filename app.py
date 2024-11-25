import pandas as pd 
import string
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize, pos_tag, ne_chunk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify, render_template
import re


#nltk.download('stopwords')
#nltk.download('punkt_tab')
#nltk.download('maxent_ne_chunker')
#nltk.download('words')
#nltk.download('averaged_perceptron_tagger_eng')
#nltk.download('maxent_ne_chunker_tab')

app = Flask(__name__)

# Reading the CSV file
df = pd.read_csv("movie_synthetic_data.csv")

# Filling the empty columns
df['MovieName'] = df['MovieName'].fillna('')
df['DateRelease'] = df['DateRelease'].fillna(pd.Timestamp('1900-01-01'))
df['Genre'] = df['Genre'].fillna('')
df['Cast'] = df['Cast'].fillna('')
df['DirectorName'] = df['DirectorName'].fillna('')
df['DateRelease'] = pd.to_datetime(df['DateRelease'], errors='coerce')
df = df[df['DateRelease'].notna()]

# Preprocessing steps
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Preprocess the relevant text columns
df['processed_movie_name'] = df['MovieName'].apply(preprocess_text)
df['processed_genre'] = df['Genre'].apply(preprocess_text)
df['processed_cast'] = df['Cast'].apply(preprocess_text)
df['processed_director'] = df['DirectorName'].apply(preprocess_text)

# Combine the preprocessed columns into a single text field (if needed)
df['processed_text'] = df['processed_movie_name'] + ' ' + df['processed_genre'] + ' ' + df['processed_cast'] + ' ' + df['processed_director']

# Initialize a TF-IDF Vectorizer
vectorizer = TfidfVectorizer(
    max_features = 1000,
    ngram_range = (1,2),
    stop_words = 'english'
)

X = vectorizer.fit_transform(df['processed_text'])

def extract_date_from_query(query):
    # Simple regex to find 4-digit years
    match = re.search(r'\b(\d{4})\b', query)
    if match:
        return int(match.group(1))
    return None
# NLTK-based named entity extraction
def extract_entities(query):
    entities = {'PERSON': None, 'DATE': None, 'ORG': None}
    tokens = word_tokenize(query)
    tagged_tokens = pos_tag(tokens)
    chunks = ne_chunk(tagged_tokens)
    
    # Extract entities from the chunks
    for chunk in chunks:
        if isinstance(chunk, tuple):
            continue  # Skip non-entities
        entity_label = chunk.label()
        entity_text = ' '.join(c[0] for c in chunk)
        if entity_label in entities:
            entities[entity_label] = entity_text
    date = extract_date_from_query(query)
    if date:
        entities['DATE'] = date
    print(f"Extracted entities: {entities}")  # Debug print statement
    return {key: value for key, value in entities.items() if value is not None}

def handle_query(query, threshold=0.1):
    entities = extract_entities(query)

    # Initialize filters
    date_condition = None
    actor_condition = None
    genre_condition = None
    director_condition = None

    # Handle specific conditions
    if 'DATE' in entities:
        try:
            year = int(entities['DATE'])
            date_condition = df['DateRelease'].dt.year < year
            #print(f"Date condition applied: {date_condition}")
        except ValueError:
            pass  # Skip invalid date formats

    if 'PERSON' in entities:
        actor = entities['PERSON']
        actor_condition = df['Cast'].str.contains(actor, case=False, na=False)

    # Check for keywords in the query for genre
    for genre in df['Genre'].unique():
        if genre.lower() in query.lower():
            genre_condition = df['Genre'].str.contains(genre, case=False, na=False)
            break  # Stop after finding the first genre

    # Check for keywords in the query for director
    for director in df['DirectorName'].unique():
        if director.lower() in query.lower():
            director_condition = df['DirectorName'].str.contains(director, case=False, na=False)
            break  # Stop after finding the first director

    # Combine conditions using AND logic
    conditions = [cond for cond in [date_condition, actor_condition, genre_condition, director_condition] if cond is not None]
    if conditions:
        final_condition = conditions[0]
        for cond in conditions[1:]:
            final_condition &= cond  # Combine conditions

        # Filter the dataset and return the results
        result = df[final_condition]
        if not result.empty:
            response = result[['MovieName', 'Genre', 'Cast', 'DateRelease', 'DirectorName']].to_dict(orient='records')
            return response
    return{"message":"No Results Found"}
    # Fallback: Use similarity-based search
    #processed_query = preprocess_text(query)
    #query_vector = vectorizer.transform([processed_query])
    #similarity_scores = cosine_similarity(query_vector, X)
    #matching_indices = [i for i, score in enumerate(similarity_scores[0]) if score > threshold]

    #if matching_indices:
     #   result = df.iloc[matching_indices][['MovieName', 'Genre', 'Cast', 'DateRelease', 'DirectorName']]
      #  return result.to_dict(orient='records')

    #return "No matching results found."

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/query', methods=['POST'])
def query_model():
    query = request.json['query']
    print(f"Received query: {query}")
    response = handle_query(query)
    if response:
        return jsonify(response)
    else:
        return jsonify({"message":"No result found"})

# Test if the filtering logic works without any entity extraction
#test_date = 2000
#result = df[df['DateRelease'].dt.year < test_date]
#print(result[['MovieName', 'DateRelease']])  # Check if this works

if __name__ == "__main__":
    app.run(debug=True)
