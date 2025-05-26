import json
import random
import time
from afro_intent_model import AfroBotIntentClassifier
from fuzzywuzzy import process
from spotify import get_spotify_token, search_song_on_spotify
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
#YARNNGPT
def speak_response(text):
    from yarngpt.audiotokenizer import AudioTokenizer
    from transformers import AutoModelForCausalLM
    import torchaudio
    import torch

    hf_path = "saheedniyi/YarnGPT"
    wav_tokenizer_config_path = "path/to/wavtokenizer_config.yaml"
    wav_tokenizer_model_path = "path/to/wavtokenizer_model.ckpt"

    if not hasattr(speak_response, "audio_tokenizer"):
        speak_response.audio_tokenizer = AudioTokenizer(
            hf_path, wav_tokenizer_model_path, wav_tokenizer_config_path
        )
        speak_response.model = AutoModelForCausalLM.from_pretrained(
            hf_path, torch_dtype=torch.float16
        ).to(speak_response.audio_tokenizer.device)

    audio_tokenizer = speak_response.audio_tokenizer
    model = speak_response.model

    prompt = audio_tokenizer.create_prompt(text, speaker_name="idera")
    input_ids = audio_tokenizer.tokenize_prompt(prompt)
    output = model.generate(
        input_ids=input_ids,
        temperature=0.1,
        repetition_penalty=1.1,
        max_length=4000,
    )
    codes = audio_tokenizer.get_codes(output)
    audio = audio_tokenizer.get_audio(codes)

    torchaudio.save("afrobot_voice.wav", audio, sample_rate=24000)
    print("[🔊 YarnGPT done speaking]")


# Load T5 paraphrasing model
# Load T5 paraphrasing model from a valid public repo
t5_tokenizer = AutoTokenizer.from_pretrained("ramsrigouthamg/t5_paraphraser")
t5_model = AutoModelForSeq2SeqLM.from_pretrained("ramsrigouthamg/t5_paraphraser")

def paraphrase_t5(text, max_length=128):
    input_text = f"paraphrase: {text} </s>"
    encoding = t5_tokenizer.encode_plus(
        input_text, padding='max_length', truncation=True,
        max_length=256, return_tensors="pt"
    )
    input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

    outputs = t5_model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        num_beams=4,
        num_return_sequences=1,
        early_stopping=True
    )

    return t5_tokenizer.decode(outputs[0], skip_special_tokens=True)


# Load sentence-transformers model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Load intents and mood
with open('intents.json', 'r') as f:
    intents = json.load(f)

with open('mood.json', 'r') as f:
    mood_data = json.load(f)

# Prepare sentence embeddings
intent_embeddings = {}
for intent in intents["intents"]:
    patterns = intent["patterns"]
    embeddings = embedder.encode(patterns, convert_to_tensor=True)
    intent_embeddings[intent["tag"]] = (patterns, embeddings)

mood_embeddings = {}
for mood in mood_data["mood"]:
    patterns = mood["patterns"]
    embeddings = embedder.encode(patterns, convert_to_tensor=True)
    mood_embeddings[mood["tag"]] = (patterns, embeddings)

# Initialize AfroBot
bot = AfroBotIntentClassifier()
bot.prepare_data(intents)
bot.build_model()
bot.train(epochs=200)

awaiting_mood = False
spotify_token = get_spotify_token()


def find_best_tag_by_embedding(user_input, embeddings_dict, threshold=0.6):
    input_embedding = embedder.encode(user_input, convert_to_tensor=True)
    best_score = 0
    best_tag = None

    for tag, (patterns, pattern_embeddings) in embeddings_dict.items():
        cosine_scores = util.cos_sim(input_embedding, pattern_embeddings)
        max_score = float(cosine_scores.max())
        if max_score > best_score and max_score > threshold:
            best_score = max_score
            best_tag = tag
    return best_tag, best_score

#def recommend_song_by_mood(user_input):
    #"""Finds a song recommendation based on mood patterns."""
    #for mood_entry in mood_data.get("mood", []):
        #best_match, score = process.extractOne(user_input.lower(), mood_entry["patterns"])
        #if score > 85:  # Threshold for fuzzy matching
            #song_choice = random.choice(mood_entry["responses"])
            #song_url = search_song_on_spotify(song_choice, spotify_token)
            #if song_url:
                #return f"{song_choice} - Listen here: {song_url}"
            #else:
                #return f"{song_choice} (Sorry, I no fit find Spotify link)"
    #return "I no understand that mood o! Try another one."

def get_response(intents, mood,user_input, awaiting_mood, spotify_token):
    original_input = user_input
    user_input = paraphrase_t5(user_input).lower()
    if awaiting_mood:
        tag, score = find_best_tag_by_embedding(user_input, mood_embeddings)
        if tag:
            mood_entry = next((m for m in mood["mood"] if m["tag"] == tag), None)
            if mood_entry:
                song_choice = random.choice(mood_entry["responses"])
                song_url = search_song_on_spotify(song_choice, spotify_token)
                if song_url:
                    return tag, f"{song_choice} - Listen here: {song_url}", False
                else:
                    return tag, f"{song_choice} (Sorry, I no fit find Spotify link)", False
        return None, "I no understand you o! Try again. Which mood you dey feel?", True

    if "recommend music" in user_input:
        return "music", "Which mood you dey feel?", True

    # Try transformer-based prediction
    tag, confidence = bot.predict(user_input)
    if confidence > 0.5:
        for intent in intents["intents"]:
            if intent["tag"] == tag:
                return tag, random.choice(intent["responses"]), False

    # Fallback to embedding similarity for intent detection
    tag, score = find_best_tag_by_embedding(user_input, intent_embeddings)
    if tag:
        intent = next((i for i in intents["intents"] if i["tag"] == tag), None)
        if intent:
            return tag, random.choice(intent["responses"]), False

    return None, "I no too understand wetin you talk 🤔. Fit rephrase am?", False

# Start chat
print("AfroBot Howfa! Wetin you wan talk? ")

while True:
    user_input = input("You: ").strip()

    # Use get_response to handle both mood and normal inputs
    tag, response, awaiting_mood = get_response(intents, mood_data, user_input, awaiting_mood, spotify_token)
    
    if tag == 'goodbye':
        print(f"AfroBot: {response}")
        speak_response(response)
        break
    
    else:
        print(f"AfroBot: {response}")
        speak_response(response)