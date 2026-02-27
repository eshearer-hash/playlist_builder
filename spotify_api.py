import logging
import threading
import urllib.parse
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler

import requests
from requests.adapters import HTTPAdapter, Retry
from urllib3.util.retry import Retry as _Retry

logger = logging.getLogger(__name__)

# Make urllib3 retries visible so we can see 429 backoffs
logging.basicConfig(level=logging.WARNING)
logging.getLogger("urllib3.util.retry").setLevel(logging.DEBUG)

def get_access_token(client_id: str, client_secret: str) -> dict[str, str]:
    response = requests.post(
        "https://accounts.spotify.com/api/token",
        data={"grant_type": "client_credentials"},
        auth=(client_id, client_secret),
    )
    response.raise_for_status()
    token = response.json()["access_token"]
    headers: dict[str, str] = {"Authorization": f"Bearer {token}"}
    return headers


def get_user_access_token(
    client_id: str,
    client_secret: str,
    redirect_uri: str = "http://127.0.0.1:5000/callback/spotify",
    scopes: str = "user-top-read user-read-recently-played",
) -> dict[str, str]:
    """OAuth Authorization Code flow. Opens browser, captures callback, returns headers."""
    result: dict = {}

    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            query = urllib.parse.urlparse(self.path).query
            params = urllib.parse.parse_qs(query)
            if "code" in params:
                result["code"] = params["code"][0]
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(b"<h1>Auth successful! You can close this tab.</h1>")
            else:
                result["error"] = params.get("error", ["unknown"])[0]
                self.send_response(400)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(b"<h1>Auth failed.</h1>")

        def log_message(self, format, *args):
            pass  # silence request logs

    parsed = urllib.parse.urlparse(redirect_uri)
    port = parsed.port or 5000

    server = HTTPServer(("127.0.0.1", port), _Handler)
    server_thread = threading.Thread(target=server.handle_request, daemon=True)
    server_thread.start()

    auth_url = (
        "https://accounts.spotify.com/authorize?"
        + urllib.parse.urlencode({
            "client_id": client_id,
            "response_type": "code",
            "redirect_uri": redirect_uri,
            "scope": scopes,
        })
    )
    print(f"Opening browser for Spotify login...\n{auth_url}")
    webbrowser.open(auth_url)

    server_thread.join(timeout=120)
    server.server_close()

    if "error" in result:
        raise RuntimeError(f"Spotify auth failed: {result['error']}")
    if "code" not in result:
        raise TimeoutError("No callback received within 120 seconds")

    resp = requests.post(
        "https://accounts.spotify.com/api/token",
        data={
            "grant_type": "authorization_code",
            "code": result["code"],
            "redirect_uri": redirect_uri,
        },
        auth=(client_id, client_secret),
    )
    resp.raise_for_status()
    token = resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def get_user_top_artists(
    headers: dict[str, str],
    time_range: str = "long_term",
) -> list[dict]:
    """Fetch all of the user's top artists for the given time range."""
    session = _session_with_retries(headers)
    artists: list[dict] = []
    url = "https://api.spotify.com/v1/me/top/artists"
    params: dict[str, str | int] = {"limit": 50, "time_range": time_range, "offset": 0}

    while url:
        resp = session.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        for a in data["items"]:
            artists.append({
                "id": a["id"],
                "name": a["name"],
                "genres": a.get("genres", []),
                "popularity": a.get("popularity"),
                "image_url": a["images"][0]["url"] if a.get("images") else None,
            })
        url = data.get("next")
        params = {}  # next URL includes query params

    print(f"Found {len(artists)} top artists ({time_range})")
    return artists

class _LoggingRetry(Retry):
    """Retry subclass that prints when sleeping for rate-limit backoff."""
    def sleep(self, response=None):
        if response:
            retry_after = self.get_retry_after(response)
            if retry_after:
                print(f"  [429] Rate-limited — waiting {retry_after:.0f}s before retry...")
        super().sleep(response)


def _session_with_retries(headers: dict[str, str], timeout: int = 30) -> requests.Session:
    """Build a requests Session that retries on 429/5xx with backoff."""
    session = requests.Session()
    session.headers.update(headers)
    retry = _LoggingRetry(
        total=5,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        respect_retry_after_header=True,
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))

    # Patch session.request to enforce a default timeout
    _original_request = session.request
    def _request_with_timeout(*args, **kwargs):
        kwargs.setdefault("timeout", timeout)
        return _original_request(*args, **kwargs)
    session.request = _request_with_timeout

    return session


def _fetch_album_tracks(session: requests.Session, album_id: str) -> set[str]:
    """Fetch all track IDs for a single album (handles pagination)."""
    track_ids: set[str] = set()
    url = f"https://api.spotify.com/v1/albums/{album_id}/tracks"
    params: dict[str, str | int] = {"limit": 50}
    while url:
        resp = session.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        for track in data["items"]:
            track_ids.add(track["id"])
        url = data.get("next")
        params = {}
    return track_ids


def get_artist_song_ids(artist_name: str, HEADERS: str, limit: int = 100) -> list[str]:
    """Get the top song IDs for an artist from Spotify, ranked by popularity.

    Searches for the artist by name, fetches all their albums (albums,
    singles, and compilations), collects every track ID, then returns
    the top `limit` tracks sorted by popularity.
    """
    session = _session_with_retries(HEADERS)

    # Search for the artist
    print(f"[{artist_name}] Searching...")
    resp = session.get(
        "https://api.spotify.com/v1/search",
        params={"q": artist_name, "type": "artist", "limit": 1},
    )
    resp.raise_for_status()
    artists = resp.json()["artists"]["items"]
    if not artists:
        print(f"[{artist_name}] No artist found")
        return []

    artist_id = artists[0]["id"]
    print(f"[{artist_name}] Found artist (ID: {artist_id})")

    # Fetch all albums (paginated), including albums, singles, and compilations
    albums = []
    url = f"https://api.spotify.com/v1/artists/{artist_id}/albums"
    params = {"include_groups": "album,single,compilation", "limit": 50}
    page = 0
    while url:
        print(f"[{artist_name}] Fetching albums page {page}...")
        resp = session.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        albums.extend(data["items"])
        url = data.get("next")
        params = {}  # next URL already includes query params
        page += 1

    print(f"[{artist_name}] {len(albums)} albums/singles/compilations")

    # Fetch all track IDs from each album sequentially (avoid nested thread pools)
    track_ids: set[str] = set()
    for i, album in enumerate(albums):
        try:
            tracks = _fetch_album_tracks(session, album["id"])
            track_ids.update(tracks)
        except requests.HTTPError as exc:
            print(f"[{artist_name}] Failed album {album['id']}: {exc}")
        if (i + 1) % 10 == 0:
            print(f"[{artist_name}] Processed {i + 1}/{len(albums)} albums...")

    print(f"[{artist_name}] {len(track_ids)} unique tracks")

    # Rank by popularity and return the top `limit`
    all_ids = list(track_ids)
    scored: list[tuple[str, int]] = []
    for i in range(0, len(all_ids), 50):
        batch = all_ids[i : i + 50]
        print(f"[{artist_name}] Scoring batch {i // 50 + 1}/{(len(all_ids) + 49) // 50}...")
        resp = session.get(
            "https://api.spotify.com/v1/tracks",
            params={"ids": ",".join(batch)},
        )
        resp.raise_for_status()
        for track in resp.json().get("tracks", []):
            if track:
                scored.append((track["id"], track.get("popularity", 0)))

    scored.sort(key=lambda t: t[1], reverse=True)
    top_ids = [tid for tid, _ in scored[:limit]]
    print(f"[{artist_name}] Done — returning top {len(top_ids)} tracks")
    return top_ids

def get_tracks_metadata(track_ids: list[str], headers: dict[str, str]) -> list[dict]:
    """Fetch track metadata for multiple track IDs using the bulk endpoint.

    Spotify's GET /v1/tracks accepts up to 50 IDs per request.
    Automatically batches larger lists.
    """
    session = _session_with_retries(headers)
    results: list[dict] = []

    for i in range(0, len(track_ids), 50):
        batch = track_ids[i : i + 50]
        resp = session.get(
            "https://api.spotify.com/v1/tracks",
            params={"ids": ",".join(batch)},
        )
        resp.raise_for_status()
        for track in resp.json().get("tracks", []):
            if track is None:
                continue
            album = track.get("album", {})
            results.append({
                "id": track["id"],
                "name": track.get("name"),
                "artists": [a["name"] for a in track.get("artists", [])],
                "artist_ids": [a["id"] for a in track.get("artists", [])],
                "album_name": album.get("name"),
                "album_id": album.get("id"),
                "album_type": album.get("album_type"),
                "album_uri": album.get("uri"),
                "album_total_tracks": album.get("total_tracks"),
                "album_release_date": album.get("release_date"),
                "album_release_date_precision": album.get("release_date_precision"),
                "album_image_url": album.get("images", [{}])[0].get("url")
                    if album.get("images") else None,
                "album_artists": [a["name"] for a in album.get("artists", [])],
                "available_markets": track.get("available_markets"),
                "disc_number": track.get("disc_number"),
                "duration_ms": track.get("duration_ms"),
                "explicit": track.get("explicit"),
                "isrc": track.get("external_ids", {}).get("isrc"),
                "ean": track.get("external_ids", {}).get("ean"),
                "upc": track.get("external_ids", {}).get("upc"),
                "external_url": track.get("external_urls", {}).get("spotify"),
                "href": track.get("href"),
                "is_playable": track.get("is_playable"),
                "popularity": track.get("popularity"),
                "preview_url": track.get("preview_url"),
                "track_number": track.get("track_number"),
                "uri": track.get("uri"),
                "is_local": track.get("is_local"),
            })

    return results


def get_track_metadata(track_id: str, HEADERS: dict[str, str]) -> dict:
    """Fetch track metadata for a single track ID.

    Note: /v1/audio-features and /v1/audio-analysis were deprecated by
    Spotify in Nov 2024 and now return 403. Only /v1/tracks is used.
    """
    session = _session_with_retries(HEADERS)
    resp = session.get(f"https://api.spotify.com/v1/tracks/{track_id}")
    resp.raise_for_status()
    track = resp.json()

    album = track.get("album", {})

    return {
        "id": track_id,

        # --- Track metadata ---
        "name": track.get("name"),
        "artists": [a["name"] for a in track.get("artists", [])],
        "artist_ids": [a["id"] for a in track.get("artists", [])],
        "album_name": album.get("name"),
        "album_id": album.get("id"),
        "album_type": album.get("album_type"),
        "album_uri": album.get("uri"),
        "album_total_tracks": album.get("total_tracks"),
        "album_release_date": album.get("release_date"),
        "album_release_date_precision": album.get("release_date_precision"),
        "album_image_url": album.get("images", [{}])[0].get("url")
            if album.get("images") else None,
        "album_artists": [a["name"] for a in album.get("artists", [])],
        "available_markets": track.get("available_markets"),
        "disc_number": track.get("disc_number"),
        "duration_ms": track.get("duration_ms"),
        "explicit": track.get("explicit"),
        "isrc": track.get("external_ids", {}).get("isrc"),
        "ean": track.get("external_ids", {}).get("ean"),
        "upc": track.get("external_ids", {}).get("upc"),
        "external_url": track.get("external_urls", {}).get("spotify"),
        "href": track.get("href"),
        "is_playable": track.get("is_playable"),
        "popularity": track.get("popularity"),
        "preview_url": track.get("preview_url"),
        "track_number": track.get("track_number"),
        "uri": track.get("uri"),
        "is_local": track.get("is_local"),
    }