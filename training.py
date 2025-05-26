import json
from afro_intent_model import AfroBotIntentClassifier # if saved separately
import torch
with open('train_intents.json', 'r') as f:
    intents = json.load(f)

model = AfroBotIntentClassifier()
model.prepare_data(intents)
model.train(epochs=200)

torch.save(model, 'afrobot_model.pth')
print("Model saved successfully!")
