from fuzzywuzzy import process
from spotify import get_spotify_token, search_song_on_spotify 
import random
import json
def get_response(intents, mood,user_input, awaiting_mood, spotify_token):
    user_input = user_input.lower()
    if awaiting_mood:
        for mood_entry in mood.get("mood", []):  
            best_match, score = process.extractOne(user_input, mood_entry["patterns"])
            if score > 90:  # Set a threshold (e.g., 70%)
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


    for intent in intents["intents"]:
        for pattern in intent["patterns"]:
            if user_input in pattern.lower():  # Check if the user input directly matches a pattern
                return intent['tag'], random.choice(intent["responses"]), False

    return None,"I no understand you o! Try again.", False

def main():
    with open('intents.json', 'r') as file1:
        data = json.load(file1)
    with open('mood.json', 'r') as file2:
        mood = json.load(file2)
    print("AfroBot: Howfa! Wetin you wan talk? ")
    awaiting_mood = False  # Track if user is expected to reply with a mood
    spotify_token = get_spotify_token() 


    while True:
        user_input = input("You: ")
        tag, response, awaiting_mood = get_response(data, mood ,user_input, awaiting_mood, spotify_token)
        print(f"AfroBot: {response} ")
        if tag == 'goodbye':
            break
    

main()