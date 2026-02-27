import concurrent.futures
import glob
import os
import random
import subprocess
import time
from pathlib import Path

import tidalapi
from tqdm.auto import tqdm

import spotify_api

ORPHEUS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "OrpheusDL")
DOWNLOADS_DIR = os.path.join(ORPHEUS_DIR, "downloads")
def download_song(tidal_id: str) -> str:
    """Download a song from TIDAL using OrpheusDL and return the file path."""
    url = f"https://tidal.com/browse/track/{tidal_id}"

    # check if already downloaded
    existing_files = glob.glob(os.path.join(DOWNLOADS_DIR, f"{tidal_id}.*"))
    if existing_files:
        return existing_files[0]

    # Record timestamp before download; subtract 1s buffer for filesystem precision
    start_time = time.time() - 1

    result = subprocess.run(
        ["python3", "orpheus.py", url],
        cwd=ORPHEUS_DIR,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Download Failed for {tidal_id}")

    # Find newly created files (exclude .lrc lyrics files) by mtime
    new_files = [
        f for f in glob.glob(os.path.join(DOWNLOADS_DIR, "**", "*"), recursive=True)
        if os.path.isfile(f) and not f.endswith(".lrc") and os.path.getmtime(f) >= start_time
    ]

    if not new_files:
        raise FileNotFoundError(
            f"No new audio file found after downloading tidal ID {tidal_id}.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

    # Rename the file to use the tidal ID as the filename
    original_path = new_files[0]
    ext = os.path.splitext(original_path)[1]
    new_path = os.path.join(DOWNLOADS_DIR, f"{tidal_id}{ext}")
    os.rename(original_path, new_path)

    return new_path

def download_songs(tidal_ids: list[dict]) -> list[str]:
    """Download multiple TIDAL tracks concurrently with staggered starts.

    Uses at most max_workers simultaneous downloads. Submissions are
    staggered with a small random delay so requests don't all fire at once.
    """
    tidal_ids = [tidal_id["tidal_id"] for tidal_id in tidal_ids if tidal_id["tidal_id"] is not None]
    results = []
    with tqdm(total=len(tidal_ids), desc="Downloading TIDAL tracks") as pbar:
        for i in tidal_ids:
            try:
                results.append(download_song(i))
            except Exception:
                pbar.update(1)
                continue
            pbar.update(1)
    return results


def create_session(session_file: Path = Path("tidal_session.json")) -> tidalapi.Session:
    """Create or restore a TIDAL session, prompting for login if needed.

    On first run, prints a URL for the user to visit to authorize the app.
    Subsequent calls restore the saved session from session_file.
    """
    session = tidalapi.Session()
    session.login_session_file(session_file)
    return session


def _get_tidal_id_by_isrc(session: tidalapi.Session, isrc: str) -> str | None:
    """Look up a TIDAL track ID by ISRC. Returns None if not found."""
    try:
        params = {"filter[isrc]": isrc}
        res = session.request.request(
            "GET", "tracks", params=params, base_url=session.config.openapi_v2_location
        ).json()
        data = res.get("data", [])
        if data:
            return str(data[0]["id"])
        return None
    except Exception:
        return None


def _search_tidal_by_name(session: tidalapi.Session, name: str, artists: list[str]) -> str | None:
    """Fall back to searching TIDAL by track name + first artist. Returns None if not found."""
    try:
        query = f"{name} {artists[0]}" if artists else name
        results = session.search(query, models=[tidalapi.Track])
        tracks = results.get("tracks", [])
        if tracks:
            return str(tracks[0].id)
        return None
    except Exception:
        return None


def _get_tidal_ids_by_isrcs(
    isrcs: list[str],
    session: tidalapi.Session,
    max_workers: int = 10,
) -> dict[str, str | None]:
    """Look up TIDAL track IDs for a list of ISRCs (concurrent)."""
    isrc_to_tidal: dict[str, str | None] = {}
    with tqdm(total=len(isrcs), desc="Looking up TIDAL IDs") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_get_tidal_id_by_isrc, session, isrc): isrc for isrc in isrcs}
            for future in concurrent.futures.as_completed(futures):
                isrc = futures[future]
                isrc_to_tidal[isrc] = future.result()
                pbar.update(1)
    return isrc_to_tidal


def spotify_to_tidal_ids(
    spotify_track_ids: list[str],
    spotify_headers: dict[str, str],
    tidal_session: tidalapi.Session,
) -> list[dict]:
    """Batch-convert Spotify track IDs to TIDAL track IDs using bulk APIs."""
    # Step 1: Bulk-fetch Spotify track metadata to get ISRCs
    spotify_tracks: list[dict] = []
    with tqdm(total=len(spotify_track_ids), desc="Fetching Spotify tracks") as pbar:
        for i in range(0, len(spotify_track_ids), 50):
            batch = spotify_track_ids[i : i + 50]
            spotify_tracks.extend(
                spotify_api.get_tracks_metadata(batch, spotify_headers)
            )
            pbar.update(len(batch))

    spotify_id_to_meta: dict[str, dict] = {t["id"]: t for t in spotify_tracks}
    spotify_id_to_isrc: dict[str, str | None] = {
        t["id"]: t.get("isrc") for t in spotify_tracks
    }

    # Step 2: Bulk ISRC -> TIDAL ID lookups
    isrcs = [isrc for isrc in spotify_id_to_isrc.values() if isrc]
    isrc_to_tidal = _get_tidal_ids_by_isrcs(isrcs, tidal_session) if isrcs else {}

    # Step 3: Assemble results, falling back to name search when ISRC lookup fails
    results: list[dict] = []
    fallback_needed = []
    for spotify_id in spotify_track_ids:
        isrc = spotify_id_to_isrc.get(spotify_id)
        tidal_id = isrc_to_tidal.get(isrc) if isrc else None
        if tidal_id is None:
            fallback_needed.append(spotify_id)
        results.append({
            "spotify_id": spotify_id,
            "isrc": isrc,
            "tidal_id": tidal_id,
        })

    if fallback_needed:
        fallback_set = set(fallback_needed)
        fallback_entries = [e for e in results if e["tidal_id"] is None and e["spotify_id"] in fallback_set]

        def _do_fallback(entry):
            meta = spotify_id_to_meta.get(entry["spotify_id"], {})
            return entry["spotify_id"], _search_tidal_by_name(
                tidal_session, meta.get("name", ""), meta.get("artists", []),
            )

        with tqdm(total=len(fallback_entries), desc="Searching TIDAL by name (fallback)") as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
                futures = {pool.submit(_do_fallback, e): e for e in fallback_entries}
                fb_lookup: dict[str, str | None] = {}
                for future in concurrent.futures.as_completed(futures):
                    sid, tid = future.result()
                    fb_lookup[sid] = tid
                    pbar.update(1)
        for entry in fallback_entries:
            entry["tidal_id"] = fb_lookup.get(entry["spotify_id"])

    return results

def get_track_lyrics(tidal_id: str, session: tidalapi.Session) -> str | None:
    """Fetch lyrics for a TIDAL track ID. Returns None if not found."""
    try:
        track = session.track(tidal_id)
        lyrics = track.lyrics()
        return lyrics.text
    except Exception:
        return None