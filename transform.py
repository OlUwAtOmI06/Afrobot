# main.py

import json, random, re
import numpy as np
from fuzzywuzzy import process
from sentence_transformers import SentenceTransformer
from spotify import get_spotify_token, search_song_on_spotify
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------
# CLEAN INPUT
def clean_text(text):
    return text.lower().strip()
# ---------------------------
# ATTENTION
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def scaled_dot_product_attention(Q, K, V):
    dk = K.shape[1]
    scores = np.dot(Q, K.T) / np.sqrt(dk)
    weights = softmax(scores)
    output = np.dot(weights, V)
    return output, weights

# ---------------------------
# LOAD DATA
def load_intents():
    with open("intents.json") as file:
        return json.load(file)

def prepare_qkv(intents, model):
    patterns, tags = [], []

    for intent in intents["intents"]:
        for pattern in intent["patterns"]:
            patterns.append(pattern)
            tags.append(intent["tag"])

    Q = model.encode(patterns)  # pattern embeddings
    unique_tags = sorted(list(set(tags)))
    tag_to_index = {tag: i for i, tag in enumerate(unique_tags)}
    
    V = np.array([np.eye(len(unique_tags))[tag_to_index[tag]] for tag in tags])  # one-hot per pattern

    return Q, V, unique_tags

# ---------------------------
# GET RESPONSE
def get_response(intents, mood, user_input, awaiting_mood, spotify_token, model, Q, V, unique_tags):
    user_input = clean_text(user_input)

    if awaiting_mood:
        for mood_entry in mood.get("mood", []):
            best_match, score = process.extractOne(user_input, mood_entry["patterns"])
            if score > 90:
                song_choice = random.choice(mood_entry["responses"])
                song_url = search_song_on_spotify(song_choice, spotify_token)
                if song_url:
                    return mood_entry["tag"], f"{song_choice} - Listen here: {song_url}", False
                else:
                    return mood_entry["tag"], f"{song_choice} (No Spotify link found)", False
        return None, "I no understand you o! Try again. Which mood you dey feel?", True

    if "recommend music" in user_input:
        return "music", "Which mood you dey feel?", True

    

    user_q = model.encode([user_input])
    similarity_scores = cosine_similarity(user_q, Q)  # Cosine similarity
    confidence = np.max(similarity_scores)
    predicted_index = np.argmax(similarity_scores)

    print(f"🔍 Predicted Tag: {unique_tags[predicted_index]} (Confidence: {confidence:.2f})")


    if confidence < 0.2:
        return None, "I no too understand. Fit rephrase?", False

    predicted_tag = unique_tags[predicted_index]
    print(f"🔍 Predicted Tag: {predicted_tag} (Confidence: {confidence:.2f})")

    for intent in intents["intents"]:
        if intent["tag"] == predicted_tag:
            return predicted_tag, random.choice(intent["responses"]), False

    return None, "I no understand you o! Try again.", False

# ---------------------------
# MAIN LOOP
def main():
    intents = load_intents()
    with open('mood.json') as file:
        mood = json.load(file)

    model = SentenceTransformer('all-MiniLM-L6-v2')
    Q, V, unique_tags = prepare_qkv(intents, model)
    spotify_token = get_spotify_token()
    awaiting_mood = False

    print("AfroBot: Howfa! Wetin you wan talk?")

    while True:
        user_input = input("You: ")
        tag, response, awaiting_mood = get_response(
            intents, mood, user_input, awaiting_mood,
            spotify_token, model, Q, V, unique_tags
        )
        print(f"AfroBot: {response}")
        if tag == "goodbye":
            break

if __name__ == "__main__":
    main()
