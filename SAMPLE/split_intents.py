import json
import random
import math

# Load original intents.json
with open("intents.json", "r") as f:
    intents = json.load(f)

# Prepare new splits
train_data = {"intents": []}
test_data = {"intents": []}

for intent in intents["intents"]:
    patterns = intent["patterns"]
    random.shuffle(patterns)

    split_idx = math.ceil(0.9 * len(patterns))
    train_patterns = patterns[:split_idx]
    test_patterns = patterns[split_idx:]

    # Add to train set
    if train_patterns:
        train_data["intents"].append({
            "tag": intent["tag"],
            "patterns": train_patterns,
            "responses": intent["responses"]
        })

    # Add to test set
    if test_patterns:
        test_data["intents"].append({
            "tag": intent["tag"],
            "patterns": test_patterns,
            "responses": intent["responses"]
        })

# Save split data
with open("train_intents.json", "w") as f:
    json.dump(train_data, f, indent=4)

with open("test_intents.json", "w") as f:
    json.dump(test_data, f, indent=4)

print("✅ Done! 'train_intents.json' and 'test_intents.json' created.")
