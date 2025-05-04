import json
from afro_intent_model import AfroBotIntentClassifier # if saved separately

# Load intents from file
with open('intents.json', 'r') as f:
    intents = json.load(f)

# Initialize the classifier
model = AfroBotIntentClassifier()

# Prepare data and train
model.prepare_data(intents)
model.train(epochs= 100)

# Test with example inputs
test_inputs = [
    "Hello there!",
    "Play me some Afrobeat",
    "Alright, goodbye now",
    "I want to listen to Burna",
    "Yo!",
    "Catch you later"
]

# Print predictions
for text in test_inputs:
    tag, confidence = model.predict(text)
    print(f"Input: '{text}' => Predicted Tag: {tag} (Confidence: {confidence:.2f})")
