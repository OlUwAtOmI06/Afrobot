import json
import random
from afro_intent_model import AfroBotIntentClassifier
from fuzzywuzzy import process
from spotify import get_spotify_token, search_song_on_spotify  # Assuming you have this
import time

# Load intents and mood
with open('data/intents.json', 'r') as f:
    intents = json.load(f)

with open('data/mood.json', 'r') as f:
    mood_data = json.load(f)

# Initialize AfroBot
bot = AfroBotIntentClassifier()
bot.prepare_data(intents)
bot.build_model()
bot.train(epochs=50)

awaiting_mood = False
spotify_token = get_spotify_token()

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

# Start chat
print("AfroBot: Howfa! Wetin you wan talk? ")

while True:
    user_input = input("You: ").strip()

    # If waiting for mood, handle mood recommendation
    if awaiting_mood:
        response = recommend_song_by_mood(user_input)
        print(f"AfroBot: {response}")
        awaiting_mood = False
        continue

    predicted_tag, confidence = bot.predict(user_input)

    if predicted_tag == 'goodbye':
        for intent in intents["intents"]:
            if intent["tag"] == predicted_tag:
                print(f"AfroBot: {random.choice(intent['responses'])}")
        break

    elif confidence > 0.5:
        # Special case for music recommendation
        if predicted_tag == "music":
            print("AfroBot: Which mood you dey feel? 🎶 (e.g., happy, sad, ginger)")
            awaiting_mood = True
        else:
            for intent in intents["intents"]:
                if intent["tag"] == predicted_tag:
                    print(f"AfroBot: {random.choice(intent['responses'])}")
    else:
        print("AfroBot: I no too understand wetin you talk 🤔. Fit rephrase am?")
