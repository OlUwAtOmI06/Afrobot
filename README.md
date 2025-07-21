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

Deployment: Render

Demo Video : https://drive.google.com/file/d/1jMpVqYRU5-mMBo1GPsnUd1Eajum9c36q/view?usp=sharing


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
# Afrobot 🎵

## Afrobeat Chatbot

An intelligent Afrobeat music recommendation system built using Natural Language Processing (NLP) to recommend songs based on user preferences. The chatbot interacts with users in Nigerian Pidgin and regional dialects, offering a unique solution to Afrobeat music discovery.

## 🎯 Features

- **Smart Music Recommendations**: Provides Afrobeat music recommendations based on user input
- **Spotify Integration**: Fetches song links directly from Spotify for an enhanced user experience  
- **NLP Capabilities**: Handles inputs in Nigerian Pidgin and regional dialects
- **Open-Source**: Community contributions are encouraged to improve and expand the project

## 🛠️ Tech Stack

- **Programming Language**: Python
- **Libraries & Tools**: NLTK, Spotify API, Logistic Regression
- **Deployment**: Render

## 🎥 Demo

Check out the demo video: [Afrobot Demo](https://drive.google.com/file/d/1jMpVqYRU5-mMBot1GPsnUd1Eajum9c36q/view?usp=sharing)

## 🚀 Getting Started

### Prerequisites

- Python 3.7 or higher
- Spotify Developer Account
- Git

### Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/OlUwAtOmI06/Afrobot.git
   cd Afrobot
   ```

2. **Set Up Virtual Environment** (Optional but Recommended)
   
   **On Windows:**
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```
   
   **On macOS/Linux:**
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   
   **Alternative manual installation:**
   ```bash
   pip install nltk spacy flask spotipy google-cloud
   ```

4. **Set Up Spotify API**
   
   You'll need to set up a Spotify Developer account and get your API keys:
   - Go to [Spotify Developer Dashboard](https://developer.spotify.com/dashboard/)
   - Create an app and get your `CLIENT_ID` and `CLIENT_SECRET`
   
   **Set environment variables:**
   
   **On Windows:**
   ```bash
   set SPOTIFY_CLIENT_ID="your-client-id"
   set SPOTIFY_CLIENT_SECRET="your-client-secret"
   ```
   
   **On macOS/Linux:**
   ```bash
   export SPOTIFY_CLIENT_ID="your-client-id"
   export SPOTIFY_CLIENT_SECRET="your-client-secret"
   ```

5. **Run the Chatbot**
   ```bash
   python app.py
   ```

## 💬 Usage

Once the application starts, you can interact with the chatbot through the terminal. Type your preferences, and the chatbot will recommend Afrobeat music!

**Example interactions:**
- "I want some upbeat Afrobeat songs"
- "Recommend some Burna Boy songs"
- "Play something for dancing"

## 🤝 Contributing

Contributions are welcome! If you'd like to contribute, please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Make your changes
4. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
5. Push to the branch (`git push origin feature/AmazingFeature`)
6. Open a Pull Request

### Ways to contribute:
- 🐛 Reporting issues
- 🔧 Fixing bugs
- 🧠 Enhancing NLP capabilities
- 🎵 Adding more Afrobeat music recommendations
- 📚 Improving documentation

## 📝 License

This project is open source. Please check the repository for license details.

## 📞 Support

If you encounter any issues or have questions, please:
- Open an issue on GitHub
- Check existing issues for solutions
- Review the demo video for guidance

## 🌟 Acknowledgments

- Nigerian music community for inspiration
- Spotify for their comprehensive music API
- Contributors who help improve this project

---

**Enjoy discovering new Afrobeat music! 🎶**
