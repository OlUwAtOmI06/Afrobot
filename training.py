import json
from afro_intent_model import AfroBotIntentClassifier
import torch

# Load training data
with open('train_intents.json', 'r') as f:
    intents = json.load(f)

# Train the model
model = AfroBotIntentClassifier()
model.prepare_data(intents)
model.train(epochs=200)

# Save only the trained classifier weights
torch.save(model.classifier.state_dict(), 'afrobot_model.pth')
print(" Model saved successfully!")
