from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from tqdm import tqdm
from pathlib import Path
from db_connection import Songs
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

downloads_dir = Path("OrpheusDL/downloads")
TRANSCRIPTION_PROMPT = """This is a song recording. Transcribe all lyrics exactly as heard, following these rules:
- Preserve all repetitions, including choruses sung multiple times
- If lyrics switch language (e.g. Spanish, French), transcribe them in their original language
- Distinguish spoken/ad-libbed lines with parentheses, e.g. (spoken: yeah, that's right)
- If a word or phrase is genuinely unclear, write [unclear] rather than guessing
- Use one blank line between song sections; do not add section labels
- Transcribe informal or slurred speech phonetically as heard, not corrected to standard spelling
- Include background vocals in brackets, e.g. [background: oh yeah]"""

def _transcribe_one(song):
    audio_file = downloads_dir / f"{song['tidal_id']}.m4a"
    result = client.audio.transcriptions.create(
        model="gpt-4o-transcribe",
        file=audio_file,
        response_format="text",
        prompt=TRANSCRIPTION_PROMPT,
    )
    lyrics = result.strip()
    Songs.update(lyrics=lyrics).where(Songs.id == song["id"]).execute()
    return {"id": song["id"], "tidal_id": song["tidal_id"], "lyrics": lyrics}


def transcribe_songs(songs_to_transcribe):
    transcribed = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_transcribe_one, s): s for s in songs_to_transcribe}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Transcribing songs"):
            song = futures[future]
            try:
                out = future.result()
                transcribed.append(out)
            except Exception as e:
                print(f"[ERROR] {song['tidal_id']}: {e}")