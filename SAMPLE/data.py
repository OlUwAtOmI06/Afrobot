import pandas as pd
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the cleaned chat data
df = pd.read_csv("clean_chat.csv")

# Display the first few rows
print("Original Data:")
print(df.head())

# Step 1: Convert text to lowercase
df["message"] = df["message"].str.lower()

# Step 2: Remove punctuation
def remove_punctuation(text):
    return text.translate(str.maketrans("", "", string.punctuation))

df["message"] = df["message"].apply(remove_punctuation)

# Step 3: Tokenize the text into words
df["tokens"] = df["message"].apply(word_tokenize)

# Step 4: Remove stopwords (including Nigerian Pidgin stopwords)
nigerian_stopwords = set(stopwords.words("english"))  # Default stopwords
nigerian_stopwords.update(["na", "dey", "go", "e", "abi", "wetin", "wan"])  # Custom Pidgin stopwords

df["tokens"] = df["tokens"].apply(lambda x: [word for word in x if word not in nigerian_stopwords])

# Step 5: Replace Nigerian slang with standard words (optional)
slang_dict = {
    "omo": "person",
    "wahala": "trouble",
    "abi": "right",
    "dey": "is",
    "how far": "how are you",
}

def replace_slang(text):
    words = text.split()
    return " ".join([slang_dict[word] if word in slang_dict else word for word in words])

df["message"] = df["message"].apply(replace_slang)

# Step 6: Lemmatization (optional but useful)
lemmatizer = WordNetLemmatizer()

df["tokens"] = df["tokens"].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])

# Display the preprocessed data
print("\nPreprocessed Data:")
print(df.head())

# Save the preprocessed data
df.to_csv("preprocessed_chat.csv", index=False)

print("\nData cleaned and saved as 'preprocessed_chat.csv'")
