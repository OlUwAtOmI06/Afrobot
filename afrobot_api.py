from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import json
import random
import time
import os
import tempfile
from afro_intent_model import AfroBotIntentClassifier
from fuzzywuzzy import process
from spotify import get_spotify_token, search_song_on_spotify
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import threading
import uuid

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables for session management
user_sessions = {}

class AfroBotSession:
    def __init__(self, session_id):
        self.session_id = session_id
        self.awaiting_mood = False
        self.conversation_history = []

# Initialize models and data (loaded once at startup)
def initialize_bot():
    print("🤖 Initializing AfroBot...")
    
    # Load T5 paraphrasing model
    global t5_tokenizer, t5_model, embedder, intents, mood_data
    global intent_embeddings, mood_embeddings, bot, spotify_token
    
    t5_tokenizer = AutoTokenizer.from_pretrained("ramsrigouthamg/t5_paraphraser")
    t5_model = AutoModelForSeq2SeqLM.from_pretrained("ramsrigouthamg/t5_paraphraser")
    
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
    
    spotify_token = get_spotify_token()
    print("✅ AfroBot initialized successfully!")

def speak_response(text, session_id):
    """Generate speech audio file for the response"""
    try:
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

        # Save with session-specific filename
        audio_file = f"afrobot_voice_{session_id}.wav"
        torchaudio.save(audio_file, audio, sample_rate=24000)
        return audio_file
    except Exception as e:
        print(f"Error generating speech: {e}")
        return None

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

def get_response(user_input, session):
    original_input = user_input
    user_input = paraphrase_t5(user_input).lower()
    
    if session.awaiting_mood:
        tag, score = find_best_tag_by_embedding(user_input, mood_embeddings)
        if tag:
            mood_entry = next((m for m in mood_data["mood"] if m["tag"] == tag), None)
            if mood_entry:
                song_choice = random.choice(mood_entry["responses"])
                song_url = search_song_on_spotify(song_choice, spotify_token)
                if song_url:
                    response = f"{song_choice} - Listen here: {song_url}"
                    session.awaiting_mood = False
                    return tag, response, False
                else:
                    response = f"{song_choice} (Sorry, I no fit find Spotify link)"
                    session.awaiting_mood = False
                    return tag, response, False
        return None, "I no understand you o! Try again. Which mood you dey feel?", True

    if "recommend music" in user_input:
        session.awaiting_mood = True
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

# API Routes

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "message": "AfroBot API is running!",
        "timestamp": time.time()
    })

@app.route('/api/session', methods=['POST'])
def create_session():
    """Create a new chat session"""
    session_id = str(uuid.uuid4())
    user_sessions[session_id] = AfroBotSession(session_id)
    
    return jsonify({
        "session_id": session_id,
        "message": "AfroBot Howfa! Wetin you wan talk?",
        "status": "success"
    })

@app.route('/api/chat', methods=['POST'])
def chat():
    """Main chat endpoint"""
    try:
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({
                "error": "Message is required",
                "status": "error"
            }), 400
        
        user_message = data['message'].strip()
        session_id = data.get('session_id')
        include_audio = data.get('include_audio', False)
        
        # Create session if not provided
        if not session_id or session_id not in user_sessions:
            session_id = str(uuid.uuid4())
            user_sessions[session_id] = AfroBotSession(session_id)
        
        session = user_sessions[session_id]
        
        # Get response from bot
        tag, response, awaiting_mood = get_response(user_message, session)
        session.awaiting_mood = awaiting_mood
        
        # Add to conversation history
        session.conversation_history.append({
            "user": user_message,
            "bot": response,
            "tag": tag,
            "timestamp": time.time()
        })
        
        # Prepare response
        response_data = {
            "session_id": session_id,
            "message": response,
            "tag": tag,
            "awaiting_mood": awaiting_mood,
            "status": "success"
        }
        
        # Generate audio if requested
        if include_audio:
            audio_file = speak_response(response, session_id)
            if audio_file:
                response_data["audio_available"] = True
                response_data["audio_url"] = f"/api/audio/{session_id}"
            else:
                response_data["audio_available"] = False
        
        # Check if conversation should end
        if tag == 'goodbye':
            response_data["conversation_ended"] = True
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

@app.route('/api/audio/<session_id>', methods=['GET'])
def get_audio(session_id):
    """Get generated audio file"""
    try:
        audio_file = f"afrobot_voice_{session_id}.wav"
        if os.path.exists(audio_file):
            return send_file(audio_file, mimetype='audio/wav')
        else:
            return jsonify({
                "error": "Audio file not found",
                "status": "error"
            }), 404
    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

@app.route('/api/history/<session_id>', methods=['GET'])
def get_conversation_history(session_id):
    """Get conversation history for a session"""
    if session_id in user_sessions:
        return jsonify({
            "session_id": session_id,
            "history": user_sessions[session_id].conversation_history,
            "status": "success"
        })
    else:
        return jsonify({
            "error": "Session not found",
            "status": "error"
        }), 404

@app.route('/api/sessions', methods=['GET'])
def list_sessions():
    """List all active sessions"""
    sessions = []
    for session_id, session in user_sessions.items():
        sessions.append({
            "session_id": session_id,
            "message_count": len(session.conversation_history),
            "awaiting_mood": session.awaiting_mood
        })
    
    return jsonify({
        "sessions": sessions,
        "total_sessions": len(sessions),
        "status": "success"
    })

@app.route('/api/session/<session_id>', methods=['DELETE'])
def delete_session(session_id):
    """Delete a session"""
    if session_id in user_sessions:
        del user_sessions[session_id]
        
        # Clean up audio file if exists
        audio_file = f"afrobot_voice_{session_id}.wav"
        if os.path.exists(audio_file):
            os.remove(audio_file)
        
        return jsonify({
            "message": "Session deleted successfully",
            "status": "success"
        })
    else:
        return jsonify({
            "error": "Session not found",
            "status": "error"
        }), 404

@app.route('/', methods=['GET'])
def home():
    """API documentation endpoint"""
    return jsonify({
        "name": "AfroBot API",
        "version": "1.0.0",
        "description": "A conversational AI bot with music recommendations and speech synthesis",
        "endpoints": {
            "GET /api/health": "Health check",
            "POST /api/session": "Create new session",
            "POST /api/chat": "Send message to bot",
            "GET /api/audio/<session_id>": "Get generated audio",
            "GET /api/history/<session_id>": "Get conversation history",
            "GET /api/sessions": "List all sessions",
            "DELETE /api/session/<session_id>": "Delete session"
        },
        "example_usage": {
            "create_session": "POST /api/session",
            "send_message": "POST /api/chat with {'message': 'Hello', 'session_id': 'session_id', 'include_audio': true}"
        }
    })

if __name__ == '__main__':
    # Initialize the bot when starting the server
    initialize_bot()
    
    # Start the Flask app
    print("🚀 Starting AfroBot API server...")
    app.run(host='0.0.0.0', port=5000, debug=True)