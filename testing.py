import json
from afro_intent_model import AfroBotIntentClassifier # if saved separately
import torch
from sklearn.metrics import accuracy_score
with open('test_intents.json', 'r') as f:
    test_intents = json.load(f)

model = AfroBotIntentClassifier(model_path="afrobot_model.pth")

correct = 0
total = 0

for intent in test_intents["intents"]:
    tag = intent["tag"]
    for pattern in intent["patterns"]:
        predicted_tag, _ = model.predict(pattern)
        if predicted_tag == tag:
            correct += 1
        total += 1

print(f"Test Accuracy: {correct / total:.2f}")
