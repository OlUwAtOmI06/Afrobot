from flask import Flask, redirect, request, session
import requests

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Use a secure random string here

# Replace these with your actual Spotify credentials
CLIENT_ID = 'your_client_id'  
CLIENT_SECRET = 'your_client_secret'
REDIRECT_URI = 'http://localhost:5000/callback'
@app.route('/')
def home():
    return 'Welcome! <a href="/login">Login with Spotify</a>'

@app.route('/login')
def login():
    auth_url = f"https://accounts.spotify.com/authorize?client_id={CLIENT_ID}&response_type=code&redirect_uri={REDIRECT_URI}&scope=user-library-read"
    return redirect(auth_url)

@app.route('/callback')
def callback():
    code = request.args.get('code')
    token_url = 'https://accounts.spotify.com/api/token'
    response = requests.post(token_url, {
        'grant_type': 'authorization_code',
        'code': code,
        'redirect_uri': REDIRECT_URI,
        'client_id': CLIENT_ID,
        'client_secret': CLIENT_SECRET
    })
    session['access_token'] = response.json().get('access_token')
    return 'You are logged in!'

if __name__ == '__main__':
    app.run(debug=True)