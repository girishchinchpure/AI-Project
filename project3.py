# Movie Recommendation System (Content-Based)

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


#  Load dataset
movies = pd.read_csv("movie.csv")

print("Dataset loaded successfully!\n")
print(movies.head())


#  Data preprocessing

# Fill missing genres
movies["genres"] = movies["genres"].fillna("")

# Convert all titles to lowercase for easy matching
movies["title"] = movies["title"].str.lower()


#  Convert text to vectors

tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(movies["genres"])


#  Compute similarity

similarity = cosine_similarity(tfidf_matrix)


#  Recommendation function

def recommend(movie_name):
    movie_name = movie_name.lower()

    # Check if movie exists
    if movie_name not in movies["title"].values:
        print("Movie not found in dataset.")
        return

    idx = movies[movies["title"] == movie_name].index[0]

    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:6]

    print(f"\nMovies similar to '{movie_name.title()}':\n")

    for i in scores:
        print(movies.iloc[i[0]]["title"].title())


#  Test the system

recommend("Toy Story")