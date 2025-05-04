# Afrobot
Afrobeat Chatbot

This is an Afrobeat music recommendation system built using Natural Language Processing (NLP) to recommend songs based on user preferences. The chatbot interacts with users in Nigerian Pidgin and regional dialects, offering a unique solution to Afrobeat music discovery.

Features:

Song Recommendations: Provides Afrobeat music recommendations based on user input.
Spotify Integration: Fetches song links directly from Spotify for an enhanced user experience.
NLP Capabilities: Handles inputs in Nigerian Pidgin and regional dialects, improving the chatbot’s responsiveness and accuracy.
Open-Source: Community contributions are encouraged to improve and expand the project.


Programming Language: Python
Libraries & Tools: NLTK, Spotify API, logistic Regression 

Deployment: still deciding 


Follow these steps to run the Afrobeat chatbot locally:

Clone the Repository
Clone the repository to your local machine using the following command:
git clone https://github.com/OlUwAtOmI06/Afrobot.git

Navigate to the Project Directory
Change into the project directory:
cd Afrobot

Set Up a Virtual Environment (Optional but Recommended)

Create and activate a virtual environment to keep dependencies isolated:
On Windows:

python -m venv venv
venv\Scripts\activate
On macOS/Linux:

python -m venv venv
source venv/bin/activate

Install Dependencies
Install the necessary Python libraries:
pip install -r requirements.txt

pip install nltk spacy flask spotipy google-cloud

Set Up Spotify API
You will need to set up a Spotify Developer account and get your API keys.
Go to Spotify Developer Dashboard to create an app and get your CLIENT_ID and CLIENT_SECRET.

Once you have the keys, set them as environment variables:

On Windows:
set SPOTIPY_CLIENT_ID="your-client-id"
set SPOTIPY_CLIENT_SECRET="your-client-secret"

On macOS/Linux:
export SPOTIPY_CLIENT_ID="your-client-id"
export SPOTIPY_CLIENT_SECRET="your-client-secret"

Run the Chatbot
Now you're ready to run the chatbot:
python app.py

Once the application starts, you can interact with the chatbot through the terminal. Type your preferences, and the chatbot will recommend Afrobeat music!


Contributing

Contributions are welcome! If you'd like to contribute, please fork the repository, make changes, and submit a pull request. You can help by:

Reporting issues
Fixing bugs
Enhancing NLP capabilities
Adding more Afrobeat music recommendations
