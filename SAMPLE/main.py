from fuzzywuzzy import process
from spotify import get_spotify_token, search_song_on_spotify
import random
import json
import pandas as pd

# Load preprocessed chat data (message, sender, datetime)
df = pd.read_csv("clean_chat.csv")

# Clean up column names to ensure there are no leading/trailing spaces
df.columns = df.columns.str.strip()

def get_response(intents, mood, user_input, awaiting_mood, spotify_token):
    user_input = user_input.lower()

    # Keep the mood logic intact
    if awaiting_mood:
        for mood_entry in mood.get("mood", []):
            best_match_tuple = process.extractOne(user_input, mood_entry["patterns"])
            best_match = best_match_tuple[0]  # Best matching pattern
            score = best_match_tuple[1]  # Score of the match

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

    # Use preprocessed data for communication
    best_match_tuple = process.extractOne(user_input, df["Message"])  # Ensure 'Message' is the correct column name
    best_match = best_match_tuple[0]  # Best matching message
    score = best_match_tuple[1]  # Score of the match

    if score > 70:  # Match threshold (you can adjust this)
        response = df.iloc[df[df["Message"] == best_match].index[0]]["Message"]
        return "chat", response, False

    # If no match is found, respond with a default message
    return None, "I no understand you o! Try again.", False


def main():
    with open('mood.json', 'r') as file2:
        mood = json.load(file2)
    print("AfroBot: Howfa! Wetin you wan talk? ")
    awaiting_mood = False  # Track if user is expected to reply with a mood
    spotify_token = get_spotify_token() 

    while True:
        user_input = input("You: ")
        tag, response, awaiting_mood = get_response(None, mood, user_input, awaiting_mood, spotify_token)
        print(f"AfroBot: {response} ")
        if tag == 'goodbye':
            break

main()

