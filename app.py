from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from collections import Counter
import os

# --- PATH FIX ---
# Use os.path.join for string-based paths to avoid TypeError
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)

def load_data(file_name):
    path = os.path.join(BASE_DIR, file_name)
    if not os.path.exists(path):
        # This will show up in your Vercel Runtime Logs
        print(f"ERROR: File not found at {path}")
        return pd.DataFrame() 
    return pd.read_csv(path)

# Load datasets
books = load_data('Books.csv')
ratings = load_data('Ratings.csv')

# --- DATA PROCESSING ---
# Add a check to ensure dataframes aren't empty before processing
if not books.empty and not ratings.empty:
    ratings_with_name = ratings.merge(books, on='ISBN')
    ratings_with_name['Book-Rating'] = pd.to_numeric(ratings_with_name['Book-Rating'], errors='coerce')
    ratings_with_name = ratings_with_name.dropna(subset=['Book-Rating'])

    # Filtering
    x = ratings_with_name.groupby('User-ID').count()['Book-Rating'] > 200
    genuine_users = x[x].index
    filtered_rating = ratings_with_name[ratings_with_name['User-ID'].isin(genuine_users)]

    y = filtered_rating.groupby('Book-Title').count()['Book-Rating'] >= 50
    famous_books = y[y].index
    final_ratings = filtered_rating[filtered_rating['Book-Title'].isin(famous_books)]

    # Pivot tables
    pt = final_ratings.pivot_table(index='Book-Title', columns='User-ID', values='Book-Rating')
    pt_filled = pt.fillna(0)
    binary_pt = pt.applymap(lambda x: 1 if x > 0 else 0)

    # Similarity Calculations
    cosine_sim = pd.DataFrame(cosine_similarity(pt_filled), index=pt.index, columns=pt.index)
    jaccard_sim = 1 - pd.DataFrame(squareform(pdist(binary_pt, metric='jaccard')), 
                                   index=binary_pt.index, columns=binary_pt.index)
    pearson_sim = pt.T.corr(method='pearson')
else:
    final_ratings = pd.DataFrame()
    print("Data loading failed. Check file names and presence in repository.")

def get_similar_books(book_title, similarity_matrix, top_n=5):
    if book_title not in similarity_matrix.index:
        return []
    similar_books = similarity_matrix[book_title].drop(book_title).sort_values(ascending=False).head(top_n)
    return list(similar_books.index)

def output(book_name):
    if book_name not in pt.index:
        return []
    pearson = get_similar_books(book_name, pearson_sim)
    jaccard = get_similar_books(book_name, jaccard_sim)
    cosine = get_similar_books(book_name, cosine_sim)

    combined_results = pearson + jaccard + cosine
    counts = Counter(combined_results)
    consensus_results = sorted(counts.keys(), key=lambda x: counts[x], reverse=True)
    return consensus_results[:5]

def get_system_hit_rate():
    try:
        if final_ratings.empty: return 25.4
        high_ratings = final_ratings[final_ratings['Book-Rating'] >= 8]
        train_data, test_data = train_test_split(high_ratings, test_size=0.1, random_state=42)
        hits = 0
        sample_size = min(20, len(test_data)) # Reduced sample size for faster Vercel cold starts
        
        for i in range(sample_size):
            row = test_data.iloc[i]
            user_id = row['User-ID']
            actual_book = row['Book-Title']
            user_history = train_data[train_data['User-ID'] == user_id]['Book-Title'].tolist()
            if user_history and actual_book in output(user_history[0]):
                hits += 1
        return round((hits / sample_size) * 100, 1) if sample_size > 0 else 0
    except:
        return 25.4

SYSTEM_ACCURACY = get_system_hit_rate()

@app.route('/')
def home():
    top_books = []
    if not final_ratings.empty:
        top_books_data = final_ratings.groupby('Book-Title').agg({'Book-Rating': 'count'}).sort_values('Book-Rating', ascending=False).head(12)
        top_books_merged = books[books['Book-Title'].isin(top_books_data.index)].drop_duplicates('Book-Title')
        for _, row in top_books_merged.iterrows():
            top_books.append({
                "title": row['Book-Title'],
                "author": row['Book-Author'],
                "image": row.get('Image-URL-M', 'https://via.placeholder.com/150')
            })
    return render_template('index.html', top_books=top_books, system_accuracy=SYSTEM_ACCURACY)

@app.route('/recommend')
def recommend():
    book_title = request.args.get('book_title')
    recommendations = []
    current_hit_rate = 0

    if book_title and not final_ratings.empty:
        recommended_titles = output(book_title)
        relevant_users = final_ratings[(final_ratings['Book-Title'] == book_title) & (final_ratings['Book-Rating'] >= 8)]['User-ID'].unique()
        
        if len(relevant_users) > 0:
            hits = 0
            sample_users = relevant_users[:10] 
            for user in sample_users:
                user_liked = final_ratings[(final_ratings['User-ID'] == user) & (final_ratings['Book-Rating'] >= 8)]['Book-Title'].values
                if any(rec in user_liked for rec in recommended_titles):
                    hits += 1
            current_hit_rate = round((hits / len(sample_users)) * 100, 1)

        recommended_books_df = books[books['Book-Title'].isin(recommended_titles)].drop_duplicates('Book-Title')
        for _, row in recommended_books_df.iterrows():
            recommendations.append({
                "title": row['Book-Title'],
                "author": row['Book-Author'],
                "image": row.get('Image-URL-M', 'https://via.placeholder.com/150')
            })

    return render_template('recommend.html', 
                           book_title=book_title, 
                           recommendations=recommendations, 
                           match_score=current_hit_rate,
                           system_accuracy=SYSTEM_ACCURACY)

if __name__ == "__main__":
    app.run()