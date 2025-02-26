from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import logging

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

@app.route('/categorize', methods=['POST'])
def categorize():
    try:
        data = request.json
        if not data or not isinstance(data, list):
            return jsonify({"error": "Invalid data format"}), 400

        titles = [tab.get('title', '') for tab in data]
        if not titles:
            return jsonify({"error": "No titles provided"}), 400

        vectorizer = TfidfVectorizer(stop_words='english')
        X = vectorizer.fit_transform(titles)

        num_clusters = min(5, len(titles))
        kmeans = KMeans(n_clusters=num_clusters)
        kmeans.fit(X)

        categories = {}
        for i, label in enumerate(kmeans.labels_):
            if label not in categories:
                categories[label] = []
            categories[label].append(i)

        named_categories = {f"Category {i+1}": indices for i, indices in categories.items()}

        return jsonify({"categories": named_categories})

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    app.run(debug=True)
