from mutagen.mp4 import MP4
from PIL import Image
import io

def extract_image_from_m4a_mutagen(file_path: str) -> tuple[bytes, str] | None:
    audio = MP4(file_path)
    
    if 'covr' not in audio.tags:
        return None
    
    cover = audio.tags['covr'][0]
    
    return Image.open(io.BytesIO(bytes(cover)))
