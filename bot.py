import random
import json
import torch
from fuzzywuzzy import process
from afro_intent_model import AfroBotIntentClassifier
from spotify import get_spotify_token, search_song_on_spotify  # Assuming you have this

# === Load data ===
with open('data/intents.json', 'r') as f:
    intents = json.load(f)

with open('data/mood.json', 'r') as f:
    mood_data = json.load(f)

# === Initialize AfroBot ===
bot = AfroBotIntentClassifier()

# Prepare label encodings & input structure
bot.prepare_data(intents)
bot.build_model()

# Load the trained weights
bot.classifier.load_state_dict(torch.load('afrobot_model.pth'))
bot.classifier.eval()

# === Spotify Setup ===
awaiting_mood = False
spotify_token = get_spotify_token()

# === Mood Handler ===
def recommend_song_by_mood(user_input):
    """Finds a song recommendation based on mood patterns."""
    for mood_entry in mood_data.get("mood", []):
        best_match, score = process.extractOne(user_input.lower(), mood_entry["patterns"])
        if score > 85:  # Threshold for fuzzy matching
            song_choice = random.choice(mood_entry["responses"])
            song_url = search_song_on_spotify(song_choice, spotify_token)
            if song_url:
                return f"{song_choice} - Listen here: {song_url}"
            else:
                return f"{song_choice} (Sorry, I no fit find Spotify link)"
    return "I no understand that mood o! Try another one."

# === Music Keyword Check ===
music_keywords = ["recommend music", "play music", "suggest song", "music", "song", "jam", "ginger", "vibe"]

# === Chat Loop ===
print("AfroBot: Howfa! Wetin you wan talk? ")

while True:
    user_input = input("You: ").strip()

    # Check if user is asking for music
    if any(kw in user_input.lower() for kw in music_keywords):
        print("AfroBot: Which mood you dey feel? 🎶 (e.g., happy, sad, ginger)")
        awaiting_mood = True
        continue

    # Mood music handler
    if awaiting_mood:
        response = recommend_song_by_mood(user_input)
        print(f"AfroBot: {response}")
        awaiting_mood = False
        continue

    # Predict the intent from the input
    predicted_tag, confidence = bot.predict(user_input)

    if predicted_tag == 'goodbye':
        for intent in intents["intents"]:
            if intent["tag"] == predicted_tag:
                print(f"AfroBot: {random.choice(intent['responses'])}")
        break

    elif confidence > 0.5:
        # Handle other intents
        for intent in intents["intents"]:
            if intent["tag"] == predicted_tag:
                print(f"AfroBot: {random.choice(intent['responses'])}")
    else:
        print("AfroBot: I no too understand wetin you talk 🤔. Fit rephrase am?")
