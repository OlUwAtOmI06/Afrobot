

from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import LabelEncoder

class AfroBotIntentClassifier:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.embedding_model = SentenceTransformer(model_name)
        self.label_encoder = LabelEncoder()
        self.classifier = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def prepare_data(self, intents):
        self.patterns = []
        self.tags = []
        
        for intent in intents["intents"]:
            for pattern in intent["patterns"]:
                self.patterns.append(pattern)
                self.tags.append(intent["tag"])

        # Encode text patterns
        self.embeddings = self.embedding_model.encode(self.patterns)
        
        # Encode labels
        self.labels = self.label_encoder.fit_transform(self.tags)
        self.num_classes = len(set(self.labels))

    def build_model(self):
        embedding_dim = self.embeddings.shape[1]
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_classes)
        ).to(self.device)

    def train(self, epochs=100, lr=0.001):
        if self.classifier is None:
            self.build_model()

        X = torch.tensor(self.embeddings, dtype=torch.float32).to(self.device)
        y = torch.tensor(self.labels, dtype=torch.long).to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=lr)

        for epoch in range(epochs):
            self.classifier.train()
            optimizer.zero_grad()
            outputs = self.classifier(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 5 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    def predict(self, user_input):
        user_embedding = self.embedding_model.encode([user_input])
        X = torch.tensor(user_embedding, dtype=torch.float32).to(self.device)
        
        self.classifier.eval()
        with torch.no_grad():
            output = self.classifier(X)
            probs = torch.softmax(output, dim=1)
            confidence, predicted_idx = torch.max(probs, 1)
            predicted_tag = self.label_encoder.inverse_transform(predicted_idx.cpu().numpy())[0]

        return predicted_tag, confidence.item()
