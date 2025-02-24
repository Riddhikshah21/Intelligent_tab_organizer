from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

app = Flask(__name__)
CORS(app)

@app.route('/categorize', methods=['POST'])
def categorize():
    data = request.json
    titles = [tab['title'] for tab in data]
    
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

if __name__ == '__main__':
    app.run(debug=True)
