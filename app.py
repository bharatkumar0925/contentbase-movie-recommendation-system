from flask import Flask, render_template, request, jsonify
from joblib import load
import pandas as pd

app = Flask(__name__)

# Load the similarity matrix
similarity_matrix = load('cosine_similarity.joblib')
movies = pd.read_csv('all_movies.csv')

# Define the recommend function
def recommend(title, similarity, data, top_n=15):
    title = title.lower()
    data = pd.read_csv('all_movies.csv')  # Reload the movies DataFrame
    data['title'] = data['title'].str.lower()
    data.set_index('title', inplace=True, drop=True)
    if title not in data.index:
        return [], []
    movie_index = data.index.get_loc(title)
    scores = similarity[movie_index]
    top_indices = scores.argsort()[::-1][1:top_n + 1]
    top_movies = data.iloc[top_indices].index.tolist()

    # Get suggestions based on user input
    suggestions = [movie for movie in data.index if title in movie.lower() and movie != title]

    return top_movies, suggestions

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/autocomplete', methods=['GET'])
def autocomplete():
    search = request.args.get('search')
    if not search:
        return jsonify([])

    # Filter movie titles for autocomplete suggestions
    movies = pd.read_csv('all_movies.csv')  # Reload the movies DataFrame
    suggestions = [title for title in movies['title'].str.lower() if search.lower() in title]
    return jsonify(suggestions[:15])  # Limit to 10 suggestions

@app.route('/recommend', methods=['POST'])
def recommend_movies():
    n = int(request.form.get('num_recommendations'))
    movie_title = request.form.get('movie_title')
    if not movie_title:
        return render_template('index.html', error='Please enter a movie title.')

    recommended_movies, suggestions = recommend(movie_title, similarity_matrix, movies, top_n=n)
    return render_template('recommendation.html', movie_title=movie_title, recommended_movies=recommended_movies, suggestions=suggestions)

if __name__ == '__main__':
    app.run(debug=True, port=5005)
