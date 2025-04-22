import json
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from fuzzywuzzy import process
from spotify import get_spotify_token, search_song_on_spotify

# Step 1: Train your model on intents
def train_intent_classifier():
    with open("intents.json") as file:
        data = json.load(file)

    X = []
    y = []

    # Loop through each intent and create training pairs
    for intent in data["intents"]:
        tag = intent["tag"]
        for pattern in intent["patterns"]:
            X.append(pattern.lower())  # input text
            y.append(tag)              # label

    # Step 2: Create the model pipeline
    model = Pipeline([
        ('vectorizer', TfidfVectorizer(ngram_range=(1, 2))),  # Converts text to vectors
        ('classifier', LogisticRegression(max_iter = 1000, class_weight = 'balanced'))  # Logistic Regression for intent classification
    ])

    # Step 3: Train the model
    model.fit(X, y)
    
    return model
import re

def clean_input(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s']", "", text)
    return text

# Step 3: Get the response based on user input
def get_response(intents, mood, user_input, awaiting_mood, spotify_token, model):
    user_input = clean_input(user_input)

    if awaiting_mood:
        for mood_entry in mood.get("mood", []):
            best_match, score = process.extractOne(user_input, mood_entry["patterns"])
            if score > 90:  # Set a threshold (e.g., 90%)
                song_choice = random.choice(mood_entry["responses"])
                
                # Search for the song on Spotify
                song_url = search_song_on_spotify(song_choice, spotify_token)
                
                if song_url:
                    return mood_entry["tag"], f"{song_choice} - Listen here: {song_url}", False
                else:
                    return mood_entry["tag"], f"{song_choice} (Sorry, I no fit find Spotify link)", False

        return None, "I no understand you o! Try again. Which mood you dey feel?", True

    if "recommend music" in user_input:
        return "music", "Which mood you dey feel?", True

    # Use the AI model to predict the intent based on user input
    predicted_tag = model.predict([user_input])[0]
    print(f"DEBUUG: Predicted Tag = {predicted_tag}")

    # Find a response for the predicted intent
    for intent in intents["intents"]:
        if intent["tag"] == predicted_tag:
            return predicted_tag, random.choice(intent["responses"]), False
            #// fallback
    for intent in intents["intents"]:
        for pattern in intent["patterns"]:
            if user_input in pattern.lower():  # Check if the user input directly matches a pattern
                return intent['tag'], random.choice(intent["responses"]), False

    return None, "I no understand you o! Try again.", False

# Step 4: Main program to run the bot
def main():
    # Step 1: Load intents and moods
    with open('intents.json', 'r') as file1:
        data = json.load(file1)
    with open('mood.json', 'r') as file2:
        mood = json.load(file2)
    
    # Step 2: Train the intent classification model
    model = train_intent_classifier()
    
    # Step 3: Initialize the Spotify token and other settings
    print("AfroBot: Howfa! Wetin you wan talk? ")
    awaiting_mood = False  # Track if user is expected to reply with a mood
    spotify_token = get_spotify_token() 

    while True:
        # Step 4: Get user input and generate response
        user_input = input("You: ")
        tag, response, awaiting_mood = get_response(data, mood, user_input, awaiting_mood, spotify_token, model)
        
        # Step 5: Show response
        print(f"AfroBot: {response} ")
        
        if tag == 'goodbye':
            break

# Run the bot
main()