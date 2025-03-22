import os
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr

app = Flask(__name__)
base_dir = os.path.dirname(os.path.abspath(__file__))
books = pd.read_csv(os.path.join(base_dir, "Books.csv"))
ratings = pd.read_csv(os.path.join(base_dir, "Ratings.csv"))
users = pd.read_csv(os.path.join(base_dir, "Users.csv"))

ratings_with_name = ratings.merge(books, on='ISBN')
ratings_with_name['Book-Rating'] = pd.to_numeric(ratings_with_name['Book-Rating'], errors='coerce')
ratings_with_name = ratings_with_name.dropna(subset=['Book-Rating'])
ratings_with_name['Book-Rating'] = ratings_with_name['Book-Rating'].astype(float)

# Filter data
genuine_users = ratings_with_name.groupby('User-ID').count()['Book-Rating'] > 200
genuine_users = genuine_users[genuine_users].index
filtered_rating = ratings_with_name[ratings_with_name['User-ID'].isin(genuine_users)]

famous_books = filtered_rating.groupby('Book-Title').count()['Book-Rating'] >= 50
famous_books = famous_books[famous_books].index
final_ratings = filtered_rating[filtered_rating['Book-Title'].isin(famous_books)]

# Pivot tables
pt = final_ratings.pivot_table(index='Book-Title', columns='User-ID', values='Book-Rating')

# Fill NA with zeros for matrix ops
binary_pt = pt.applymap(lambda x: 1 if x >= 3 else 0 if pd.notnull(x) else 0)

# Limit for performance
limited_books = binary_pt.index[:30000]  # Reduced further for quicker test
binary_pt = binary_pt.loc[limited_books]
pt = pt.loc[limited_books]

# Similarity calculations
jaccard_similarity = 1 - pd.DataFrame(squareform(pdist(binary_pt, metric='jaccard')),
                                      index=binary_pt.index,
                                      columns=binary_pt.index)

cosine_similarity_matrix = pd.DataFrame(cosine_similarity(pt.fillna(0)),
                                        index=pt.index,
                                        columns=pt.index)

def compute_pearson_correlation(pt):
    return pt.T.corr(method='pearson')

pearson_similarity = compute_pearson_correlation(pt)

def get_similar_books(book_title, similarity_matrix, top_n=5):
    if book_title not in similarity_matrix.index:
        return []
    similar_books = similarity_matrix[book_title].drop(book_title).sort_values(ascending=False).head(top_n)
    return list(similar_books.index)

def output(book_name):
    pearson_results = get_similar_books(book_name, pearson_similarity)
    jaccard_results = get_similar_books(book_name, jaccard_similarity)
    cosine_results = get_similar_books(book_name, cosine_similarity_matrix)

    # Fallback logic: prefer Pearson > Jaccard > Cosine if ties
    all_sets = [set(pearson_results), set(jaccard_results), set(cosine_results)]
    common = set.intersection(*all_sets)
    if common:
        return list(common)
    elif len(pearson_results) >= len(jaccard_results) and len(pearson_results) >= len(cosine_results):
        return pearson_results
    elif len(jaccard_results) >= len(pearson_results) and len(jaccard_results) >= len(cosine_results):
        return jaccard_results
    else:
        return cosine_results

# ROUTES
@app.route('/')
def home():
    top_books_data = final_ratings.groupby('Book-Title').agg({
        'Book-Rating': 'count'
    }).sort_values('Book-Rating', ascending=False).head(12)

    top_books_merged = books[books['Book-Title'].isin(top_books_data.index)].drop_duplicates('Book-Title')

    top_books = []
    for _, row in top_books_merged.iterrows():
        top_books.append({
            "title": row['Book-Title'],
            "author": row['Book-Author'],
            "image": row.get('Image-URL-M', 'https://via.placeholder.com/150')
        })

    return render_template('index.html', top_books=top_books)

@app.route('/recommend')
def recommend():
    book_title = request.args.get('book_title')
    recommendations = []

    if book_title:
        recommended_titles = output(book_title)
        recommended_books_df = books[books['Book-Title'].isin(recommended_titles)].drop_duplicates('Book-Title')
        searched_book_df = books[books['Book-Title'] == book_title].drop_duplicates('Book-Title')
        recommended_books_df = pd.concat([searched_book_df, recommended_books_df], ignore_index=True)
        
        for _, row in recommended_books_df.iterrows():
            recommendations.append({
                "title": row['Book-Title'],
                "author": row['Book-Author'],
                "image": row.get('Image-URL-M', 'https://via.placeholder.com/150')
            })

    return render_template('recommend.html', book_title=book_title, recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)

