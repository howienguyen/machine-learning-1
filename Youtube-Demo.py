# # Sample Python code for youtube.search.list
# # See instructions for running these code samples locally:
# # https://developers.google.com/explorer-help/code-samples#python

# import os

# import google_auth_oauthlib.flow
# import googleapiclient.discovery
# import googleapiclient.errors

# scopes = ["https://www.googleapis.com/auth/youtube",
#           "https://www.googleapis.com/auth/youtube.force-ssl",
#           "https://www.googleapis.com/auth/youtube.readonly",
#           "https://www.googleapis.com/auth/youtubepartner"]

# def main():
#     # Disable OAuthlib's HTTPS verification when running locally.
#     # *DO NOT* leave this option enabled in production.
#     os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

#     api_service_name = "youtube"
#     api_version = "v3"
#     client_secrets_file = "client_secret_629872965311-23ccuh2oj36005u3s4906gjma5qb0n7p.apps.googleusercontent.com.json"

#     # Get credentials and create an API client
#     flow = google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file(
#         client_secrets_file, scopes)
    
#     # credentials = flow.run_console()
#     credentials = flow.run_local_server(port=8080, open_browser=False)
    
#     youtube = googleapiclient.discovery.build(
#         api_service_name, api_version, credentials=credentials)

#     request = youtube.search().list(
#         part="snippet",
#         q="rock+music",
#         type="video",
#         maxResults=10
#     )
#     response = request.execute()

#     print(response)

# if __name__ == "__main__":
#     main()


# import googleapiclient.discovery
# import json

# def main():
#     api_service_name = "youtube"
#     api_version = "v3"
#     # Generate this key in your Google Cloud Console under APIs & Services > Credentials
#     DEVELOPER_KEY = "****" 

#     youtube = googleapiclient.discovery.build(
#         api_service_name, api_version, developerKey=DEVELOPER_KEY)

#     request = youtube.search().list(
#         part="snippet",
#         q="rock+music",
#         type="video",
#         videoEmbeddable="true",
#         maxResults=1
#     )
#     response = request.execute()
#     print(json.dumps(response))

# if __name__ == "__main__":
#     main()


import requests
from rich import print_json # Import rich's JSON formatter
import dotenv
import os

dotenv.load_dotenv()

def main():
    DEVELOPER_KEY = os.getenv("GOOGLE_DEVELOPER_API_KEY")
    url = "https://www.googleapis.com/youtube/v3/search"

    if not DEVELOPER_KEY:
        raise RuntimeError(
            "Missing GOOGLE_DEVELOPER_API_KEY in environment. "
            "Set it in a .env file or your shell environment."
        )
    
    params = {
        "part": "snippet",
        # Don't pre-encode with '+'; requests will handle URL encoding.
        "q": "single rock music song",
        "type": "video",
        "videoEmbeddable": "true",
        "maxResults": 3,
        "key": DEVELOPER_KEY
    }

    try:
        response = requests.get(url, params=params, timeout=20)

        # If YouTube returns an error, it usually includes a useful JSON body.
        if not response.ok:
            try:
                error_data = response.json()
                print_json(data=error_data)
            except ValueError:
                print(response.text)
            response.raise_for_status()

        data = response.json()
        print_json(data=data)
        
    except requests.exceptions.RequestException as e:
        print(f"Error making API request: {e}")

if __name__ == "__main__":
    main()