from flask import Flask, jsonify, request
from flask_cors import CORS
import torch
from gensim.models import KeyedVectors
from model import RelationshipModel
import random

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the filtered word embeddings
filtered_word_vectors = KeyedVectors.load('filtered_word_vectors.kv')

# Load the trained model
model = RelationshipModel(embedding_dim=300, hidden_dim=64, filtered_word_vectors=filtered_word_vectors)
model.load_state_dict(torch.load('relationship_model.pth'))
model.eval()

def is_noun(word):
    return word.lower() in filtered_word_vectors.key_to_index

def get_random_start_end_words(similarity_threshold=0.2):
    while True:
        start_word = filtered_word_vectors.index_to_key[random.randint(0, len(filtered_word_vectors.index_to_key) - 1)]
        end_word = filtered_word_vectors.index_to_key[random.randint(0, len(filtered_word_vectors.index_to_key) - 1)]
        try:
            similarity = model(start_word, end_word).item()
            if similarity < similarity_threshold:
                return start_word, end_word
        except KeyError:
            continue

@app.route('/check_word', methods=['GET'])
def check_word():
    user_word = request.args.get('user_word')
    start_word = request.args.get('start_word')
    end_word = request.args.get('end_word')

    if not user_word:
        return jsonify({'error': 'Missing user_word parameter'}), 400
    if not start_word or not end_word:
        return jsonify({'error': 'Missing start_word or end_word parameter'}), 400
    if not is_noun(user_word):
        return jsonify({'error': 'User word must be a noun'}), 400

    try:
        with torch.no_grad():
            similarity_to_start = model(start_word, user_word).item()
            similarity_to_end = model(user_word, end_word).item()
            
            # Normalize the similarity scores to a range of 0 to 1
            min_similarity = min(similarity_to_start, similarity_to_end)
            max_similarity = max(similarity_to_start, similarity_to_end)
            normalized_similarity_to_start = (similarity_to_start - min_similarity) / (max_similarity - min_similarity)
            normalized_similarity_to_end = (similarity_to_end - min_similarity) / (max_similarity - min_similarity)
            
    except KeyError as e:
        return jsonify({'error': f"Word not found in the model: {str(e)}"}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    return jsonify({
        'user_word': user_word,
        'similarity_to_start': float(normalized_similarity_to_start),
        'similarity_to_end': float(normalized_similarity_to_end)
    })

@app.route('/start_game', methods=['GET'])
def start_game():
    start_word, end_word = get_random_start_end_words()
    return jsonify({
        'start_word': start_word,
        'end_word': end_word
    })

if __name__ == '__main__':
    app.run(debug=True)