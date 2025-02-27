from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import logging
from sklearn.decomposition import TruncatedSVD
import numpy as np
from collections import Counter
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}}, supports_credentials=True)
logging.basicConfig(level=logging.INFO)

@app.route('/categorize', methods=['OPTIONS'])
def handle_options():
    response = jsonify({"message": "CORS preflight successful"})
    response.headers["Access-Control-Allow-Origin"] = "http://localhost:3000"
    response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response

def extract_keywords(titles, n=5):
    stop_words = set(stopwords.words('english'))
    words = [word.lower() for title in titles for word in nltk.word_tokenize(title) if word.isalnum()]
    word_freq = Counter(word for word in words if word not in stop_words)
    return [word for word, _ in word_freq.most_common(n)]

def predict_category_name(titles):
    keywords = extract_keywords(titles)
    return ' '.join(keywords[:2]).title() if keywords else "Miscellaneous"

@app.route('/categorize', methods=['POST'])
def categorize():
    try:
        data = request.json
        if not data or not isinstance(data, list):
            return jsonify({"error": "Invalid data format"}), 400

        titles = [tab.get('title', '') for tab in data]
        if not titles:
            return jsonify({"error": "No titles provided"}), 400

        # TF-IDF Vectorization
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        X = vectorizer.fit_transform(titles)

        # Dimensionality reduction
        svd = TruncatedSVD(n_components=min(50, X.shape[1] - 1))
        X_reduced = svd.fit_transform(X)

        # Determine optimal number of clusters
        max_clusters = min(10, len(titles))
        inertias = []
        for k in range(1, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X_reduced)
            inertias.append(kmeans.inertia_)

        optimal_clusters = np.argmin(np.diff(inertias)) + 1

        # Perform clustering
        kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
        labels = kmeans.fit_predict(X_reduced)

        # Group tabs and predict category names
        categories = {}
        for label in set(labels):
            indices = [i for i, l in enumerate(labels) if l == label]
            cluster_titles = [titles[i] for i in indices]
            category_name = predict_category_name(cluster_titles)
            categories[category_name] = indices

        return jsonify({"categories": categories})

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    app.run(debug=True)

# @app.route('/categorize', methods=['POST'])
# def categorize():
#     try:
#         data = request.json
#         if not data or not isinstance(data, list):
#             return jsonify({"error": "Invalid data format"}), 400

#         titles = [tab.get('title', '') for tab in data]
#         if not titles:
#             return jsonify({"error": "No titles provided"}), 400
#         vectorizer = TfidfVectorizer(stop_words='english')
#         X = vectorizer.fit_transform(titles)

#         num_clusters = min(5, len(titles))
#         kmeans = KMeans(n_clusters=num_clusters)
#         kmeans.fit(X)

#         categories = {}
#         print(kmeans.labels_)
#         for i, label in enumerate(kmeans.labels_):
#             if label not in categories:
#                 categories[label] = []
#             categories[label].append(i)

#         named_categories = {f"Category {i+1}": indices for i, indices in categories.items()}

#         return jsonify({"categories": named_categories})

#     except Exception as e:
#         logging.error(f"An error occurred: {str(e)}")
#         return jsonify({"error": "Internal server error"}), 500

# if __name__ == '__main__':
#     app.run(debug=True)
