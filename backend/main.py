from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import logging
from sklearn.decomposition import TruncatedSVD
import numpy as np
from collections import Counter
import nltk
import spacy
from sklearn.decomposition import LatentDirichletAllocation
from transformers import pipeline
from nltk.corpus import stopwords
import re
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict


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

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    print("Downloading spaCy model...")
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# Predefined categories with keywords
category_keywords = {
    "Travel": ["travel", "flight", "hotel", "booking", "trip", "vacation"],
    "Education": ["course", "learn", "study", "tutorial", "education"],
    "Research": ["research", "paper", "study", "analysis", "journal", "science"],
    "AI & ML": ["ai", "machine learning", "deep learning", "neural network", "data science"],
    "Comedy": ["comedy", "stand up", "funny", "humor", "joke"],
    "Social Media": ["facebook", "twitter", "instagram", "linkedin", "social"],
    "Games": ["game", "gaming", "play", "steam", "xbox", "playstation"],
    "Commute": ["commute", "traffic", "transport", "bus", "train", "subway"],
    "Programming": ["code", "programming", "developer", "software", "github"],
    "Job Search": ["job", "career", "interview", "resume", "hiring"],
    "Entertainment": ["movie", "film", "show", "stream", "watch", "music"],
    "News": ["news", "article", "report", "update", "headline"],
    "Technology": ["tech", "gadget", "device", "innovation", "startup"],
    "Health": ["health", "fitness", "workout", "diet", "medical"],
    "Shopping": ["shop", "buy", "product", "price", "deal", "amazon"]
}

def preprocess_text(text):
    doc = nlp(text.lower())
    return ' '.join([token.lemma_ for token in doc if not token.is_stop and token.is_alpha])

def get_top_terms(vectorizer, clf, class_labels, n_top=5):
    feature_names = vectorizer.get_feature_names_out()
    top_terms = {}
    for i, class_label in enumerate(class_labels):
        top_indices = np.argsort(clf.cluster_centers_[i])[-n_top:]
        top_terms[i] = [feature_names[ind] for ind in top_indices]
    return top_terms

def assign_cluster_labels(titles, cluster_labels, top_terms):
    cluster_contents = {i: [] for i in set(cluster_labels)}
    for title, label in zip(titles, cluster_labels):
        cluster_contents[label].append(title)
    
    cluster_names = {}
    for label, contents in cluster_contents.items():
        # Combine top terms and frequent words in titles
        all_words = ' '.join([preprocess_text(title) for title in contents] + [' '.join(top_terms[label])])
        words = all_words.split()
        word_freq = Counter(words)
        
        # Check if any predefined category matches
        max_match = 0
        best_category = "Miscellaneous"
        for category, keywords in category_keywords.items():
            match_count = sum(word_freq[keyword.lower()] for keyword in keywords if keyword.lower() in word_freq)
            if match_count > max_match:
                max_match = match_count
                best_category = category
        
        if max_match > 0:
            cluster_names[label] = best_category
        else:
            # Generate custom name if no predefined category matches well
            top_words = [word for word, _ in word_freq.most_common(2) if word not in ['com', 'www']]
            cluster_names[label] = ' '.join(top_words).title()
    
    return cluster_names

 
@app.route('/categorize', methods=['POST'])
def categorize():
    try:
        data = request.json
        if not data or not isinstance(data, list):
            return jsonify({"error": "Invalid data format"}), 400

        titles = [tab.get('title', '') for tab in data]
        if not titles:
            return jsonify({"error": "No titles provided"}), 400
      
        # Main clustering and labeling process
        vectorizer = TfidfVectorizer(
            stop_words='english',
            min_df=1,
            max_df=0.9,
            ngram_range=(1, 2)
        )
        X = vectorizer.fit_transform(titles)

        optimal_k = 7

        # Perform K-means clustering
        kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        kmeans.fit(X)
        
        top_terms = get_top_terms(vectorizer, kmeans, range(optimal_k))
        cluster_names = assign_cluster_labels(titles, kmeans.labels_, top_terms)
        print(cluster_names)
        result = {}
        for label, category in cluster_names.items():
            tab_indices = [i for i, cl in enumerate(kmeans.labels_) if cl == label]
            tabs = [titles[i] for i in tab_indices]
            result[category] = {
                "name": category,
                "tabIndices": tab_indices,
                "tabs": tabs
            }

        # return result
        return jsonify({"categories": result})
    
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        # Log more details about the error
        import traceback
        logging.error(traceback.format_exc())
        return jsonify({"error": "Internal server error"}), 500
    
if __name__ == '__main__':
    app.run(debug=True)
