import json
from afro_intent_model import AfroBotIntentClassifier # if saved separately
import torch
from sklearn.metrics import accuracy_score

# Load intents from file
with open('intents.json', 'r') as f:
    intents = json.load(f)

# Initialize the classifier
model = AfroBotIntentClassifier()

# Prepare data and train
model.prepare_data(intents)
model.train(epochs= 200)

#torch.save(model.state_dict(), 'afrobot_model.pth')  # Save as afrobot_model.pth
#print("Model saved successfully!")
# Test with example inputs
test_inputs = [
    "Hello there!",
    "I will put alot of effort",
    "Alright, goodbye now",
    "How your area",
    "Anita gives me butterflies",
    "hunger wan finish me",
    "How is the sun today",
    "How do i study effectively",
    "Do you have a preferred name?",
    "What is your function",
    "crack a joke",
    "You get an news info",
    "What if i fail"

]
true_labels = ["greetings", "motivation ", "goodbye", "checking_in", "love_and_relationship", "food_recommendation","Weather", "Study_tips","name","work",
              "funny_response","news_update", "Mental_health"]

# Store predicted labels
predicted_labels = []

# Print predictions
for text in test_inputs:
    tag, confidence = model.predict(text)
    print(f"Input: '{text}' => Predicted Tag: {tag} (Confidence: {confidence:.2f})")
    predicted_labels.append(tag)

accuracy = accuracy_score(true_labels, predicted_labels)
print(f"Model Accuracy: {accuracy:.2f}")
