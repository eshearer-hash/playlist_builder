from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from requests.adapters import HTTPAdapter, Retry

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

def _session_with_retries(headers: dict[str, str]) -> requests.Session:
    """Build a requests Session that retries on 429/5xx with backoff."""
    session = requests.Session()
    session.headers.update(headers)
    retry = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        respect_retry_after_header=True,
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
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


def get_artist_song_ids(artist_name: str, HEADERS: str) -> list[str]:
    """Get every song ID for an artist from Spotify.

    Searches for the artist by name, fetches all their albums (albums,
    singles, and compilations), then collects every track ID across
    those albums. Returns a deduplicated list of track IDs.
    """
    session = _session_with_retries(HEADERS)

    # Search for the artist
    resp = session.get(
        "https://api.spotify.com/v1/search",
        params={"q": artist_name, "type": "artist", "limit": 1},
    )
    resp.raise_for_status()
    artists = resp.json()["artists"]["items"]
    if not artists:
        print(f"No artist found for '{artist_name}'")
        return []

    artist_id = artists[0]["id"]
    print(f"Found artist: {artists[0]['name']} (ID: {artist_id})")

    # Fetch all albums (paginated), including albums, singles, and compilations
    albums = []
    url = f"https://api.spotify.com/v1/artists/{artist_id}/albums"
    params = {"include_groups": "album,single,compilation", "limit": 50}
    while url:
        resp = session.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        albums.extend(data["items"])
        url = data.get("next")
        params = {}  # next URL already includes query params

    print(f"Found {len(albums)} albums/singles/compilations")

    # Fetch all track IDs from each album in parallel
    track_ids: set[str] = set()
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = {
            pool.submit(_fetch_album_tracks, session, album["id"]): album["id"]
            for album in albums
        }
        for future in as_completed(futures):
            try:
                track_ids.update(future.result())
            except requests.HTTPError as exc:
                print(f"Failed to fetch tracks for album {futures[future]}: {exc}")

    print(f"Found {len(track_ids)} unique tracks")
    return list(track_ids)

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