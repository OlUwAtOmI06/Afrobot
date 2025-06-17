from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import random
import time
import uuid
import os
import threading
from afro_intent_model import AfroBotIntentClassifier
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = Flask(__name__)
CORS(app)

# Session management with cleanup
user_sessions = {}
SESSION_TIMEOUT = 3600  # 1 hour
MAX_SESSIONS = 1000

# Global variables for models (lazy loading)
t5_tokenizer = None
t5_model = None
embedder = None
intents = None
mood_data = None
intent_embeddings = None
mood_embeddings = None
bot = None
models_loaded = False
loading_lock = threading.Lock()

class AfroBotSession:
    def __init__(self, session_id):
        self.session_id = session_id
        self.awaiting_mood = False
        self.conversation_history = []
        self.created_at = time.time()
        self.last_activity = time.time()
    
    def update_activity(self):
        self.last_activity = time.time()

def cleanup_expired_sessions():
    """Remove expired sessions"""
    current_time = time.time()
    expired_sessions = [
        sid for sid, session in user_sessions.items()
        if current_time - session.last_activity > SESSION_TIMEOUT
    ]
    for sid in expired_sessions:
        del user_sessions[sid]
    
    # Also limit total sessions
    if len(user_sessions) > MAX_SESSIONS:
        # Remove oldest sessions
        oldest_sessions = sorted(
            user_sessions.items(),
            key=lambda x: x[1].last_activity
        )[:len(user_sessions) - MAX_SESSIONS]
        for sid, _ in oldest_sessions:
            del user_sessions[sid]

def load_models_lazy():
    """Load models only when needed"""
    global t5_tokenizer, t5_model, embedder, intents, mood_data
    global intent_embeddings, mood_embeddings, bot, models_loaded
    
    with loading_lock:
        if models_loaded:
            return True
        
        try:
            print("Loading models...")
            
            # Check if required files exist
            required_files = ['intents.json', 'mood.json']
            for file in required_files:
                if not os.path.exists(file):
                    raise FileNotFoundError(f"Required file {file} not found")
            
            # Load data files
            with open('intents.json', 'r') as f:
                intents = json.load(f)
            with open('mood.json', 'r') as f:
                mood_data = json.load(f)
            
            # Load models
            t5_tokenizer = AutoTokenizer.from_pretrained("ramsrigouthamg/t5_paraphraser")
            t5_model = AutoModelForSeq2SeqLM.from_pretrained("ramsrigouthamg/t5_paraphraser")
            embedder = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Precompute embeddings
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
            
            # Initialize and train bot (consider pre-training and saving the model)
            bot = AfroBotIntentClassifier()
            bot.prepare_data(intents)
            bot.build_model()
            
            # WARNING: This is still problematic - consider loading pre-trained model
            print("Training model... This may take a while.")
            bot.train(epochs=50)  # Reduced epochs for faster startup
            
            models_loaded = True
            print("Models loaded successfully!")
            return True
            
        except Exception as e:
            print(f"Error loading models: {e}")
            return False

def paraphrase_t5(text, max_length=128):
    if not models_loaded:
        return text
    
    try:
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
    except Exception as e:
        print(f"Paraphrasing error: {e}")
        return text

def find_best_tag_by_embedding(user_input, embeddings_dict, threshold=0.6):
    if not models_loaded:
        return None, 0
    
    try:
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
    except Exception as e:
        print(f"Embedding error: {e}")
        return None, 0

def get_response(user_input, session):
    if not models_loaded:
        return None, "Bot is still loading. Please try again in a moment.", False
    
    try:
        original_input = user_input.lower()
        
        # Check for music recommendation BEFORE paraphrasing
        music_keywords = ["recommend music", "suggest music", "music recommendation", 
                         "play music", "want music", "need music", "music please"]
        if any(keyword in original_input for keyword in music_keywords):
            session.awaiting_mood = True
            return "music", "Which mood you dey feel?", True
        
        # Now paraphrase for other processing
        user_input = paraphrase_t5(user_input).lower()
        
        if session.awaiting_mood:
            tag, score = find_best_tag_by_embedding(user_input, mood_embeddings)
            if tag:
                mood_entry = next((m for m in mood_data["mood"] if m["tag"] == tag), None)
                if mood_entry:
                    song_choice = random.choice(mood_entry["responses"])
                    response = f"{song_choice}"
                    session.awaiting_mood = False
                    return tag, response, False
            return None, "I no understand you o! Try again. Which mood you dey feel?", True

        tag, confidence = bot.predict(user_input)
        if confidence > 0.5:
            for intent in intents["intents"]:
                if intent["tag"] == tag:
                    return tag, random.choice(intent["responses"]), False

        tag, score = find_best_tag_by_embedding(user_input, intent_embeddings)
        if tag:
            intent = next((i for i in intents["intents"] if i["tag"] == tag), None)
            if intent:
                return tag, random.choice(intent["responses"]), False

        return None, "I no too understand wetin you talk 🤔. Fit rephrase am?", False
    
    except Exception as e:
        print(f"Response generation error: {e}")
        return None, "Sorry, something went wrong. Please try again.", False

# API routes
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy" if models_loaded else "loading",
        "models_loaded": models_loaded,
        "active_sessions": len(user_sessions),
        "timestamp": time.time()
    })

@app.route('/api/session', methods=['POST'])
def create_session():
    if not load_models_lazy():
        return jsonify({
            "error": "Models are still loading. Please try again.",
            "status": "loading"
        }), 503
    
    cleanup_expired_sessions()
    
    session_id = str(uuid.uuid4())
    user_sessions[session_id] = AfroBotSession(session_id)
    
    return jsonify({
        "session_id": session_id,
        "message": "AfroBot Howfa! Wetin you wan talk?",
        "status": "success"
    })

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        if not load_models_lazy():
            return jsonify({
                "error": "Models are still loading. Please try again.",
                "status": "loading"
            }), 503
        
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({"error": "Message is required", "status": "error"}), 400
        
        user_message = data['message'].strip()
        if not user_message:
            return jsonify({"error": "Message cannot be empty", "status": "error"}), 400
        
        session_id = data.get('session_id')
        
        # Session handling
        if not session_id or session_id not in user_sessions:
            cleanup_expired_sessions()
            session_id = str(uuid.uuid4())
            user_sessions[session_id] = AfroBotSession(session_id)
        
        session = user_sessions[session_id]
        session.update_activity()
        
        tag, response, awaiting_mood = get_response(user_message, session)
        session.awaiting_mood = awaiting_mood
        
        session.conversation_history.append({
            "user": user_message,
            "bot": response,
            "tag": tag,
            "timestamp": time.time()
        })
        
        # Limit conversation history
        if len(session.conversation_history) > 100:
            session.conversation_history = session.conversation_history[-50:]
        
        response_data = {
            "session_id": session_id,
            "message": response,
            "tag": tag,
            "awaiting_mood": awaiting_mood,
            "status": "success"
        }
        
        if tag == 'goodbye':
            response_data["conversation_ended"] = True
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Chat error: {e}")
        return jsonify({"error": "Internal server error", "status": "error"}), 500

@app.route('/api/history/<session_id>', methods=['GET'])
def get_conversation_history(session_id):
    cleanup_expired_sessions()
    if session_id in user_sessions:
        session = user_sessions[session_id]
        session.update_activity()
        return jsonify({
            "session_id": session_id,
            "history": session.conversation_history,
            "status": "success"
        })
    else:
        return jsonify({"error": "Session not found", "status": "error"}), 404

@app.route('/api/sessions', methods=['GET'])
def list_sessions():
    cleanup_expired_sessions()
    sessions = []
    for session_id, session in user_sessions.items():
        sessions.append({
            "session_id": session_id,
            "message_count": len(session.conversation_history),
            "awaiting_mood": session.awaiting_mood,
            "created_at": session.created_at,
            "last_activity": session.last_activity
        })
    return jsonify({
        "sessions": sessions, 
        "total_sessions": len(sessions), 
        "status": "success"
    })

@app.route('/api/session/<session_id>', methods=['DELETE'])
def delete_session(session_id):
    if session_id in user_sessions:
        del user_sessions[session_id]
        return jsonify({"message": "Session deleted successfully", "status": "success"})
    else:
        return jsonify({"error": "Session not found", "status": "error"}), 404

if __name__ == '__main__':
    # Set environment variable to avoid tokenizer warnings
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    print("Starting AfroBot API server...")
    print("Models will be loaded on first request to improve startup time.")
    
    # Use a production server for deployment
    # Consider using gunicorn: gunicorn -w 4 -b 0.0.0.0:5001 first_api:app
    app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)