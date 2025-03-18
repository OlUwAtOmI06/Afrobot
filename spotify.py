import requests

# Spotify API Credentials
CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")

# Get Spotify Token
def get_spotify_token():
    url = "https://accounts.spotify.com/api/token"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = {"grant_type": "client_credentials", "client_id": CLIENT_ID, "client_secret": CLIENT_SECRET}

    response = requests.post(url, headers=headers, data=data)
    return response.json().get("access_token")

# Search for song on Spotify
def search_song_on_spotify(song_name, token):
    url = f"https://api.spotify.com/v1/search?q={song_name}&type=track&limit=1"
    headers = {"Authorization": f"Bearer {token}"}

    response = requests.get(url, headers=headers)
    data = response.json()
    
    try:
        song = data["tracks"]["items"][0]
        return song["external_urls"]["spotify"]  # Return song link
    except (IndexError, KeyError):
        return None  # If song is not found
