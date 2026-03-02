from peewee import *
from playhouse.postgres_ext import JSONField, ArrayField
from playhouse.migrate import PostgresqlMigrator, migrate as pw_migrate
from pgvector.peewee import VectorField
from dotenv import load_dotenv
import os

load_dotenv()

db = PostgresqlDatabase(os.getenv("POSTGRESQL_DATA_URL"))

class BaseModel(Model):
    """All models inherit this to share the database connection."""
    class Meta:
        database = db

class Songs(BaseModel):
    id = UUIDField(primary_key=True)
    spotify_id = TextField(unique=True)
    tidal_id = TextField(unique=True, null=True)  # may be null if no TIDAL match
    song_name = TextField(unique=False)
    artists = ArrayField(TextField)
    album_name = TextField(null=True)  # may be null if no album info available
    album_total_tracks = IntegerField(null=True)  # may be null if no album info available
    release_date = DateField(null=True)  # may be null if no release date available
    disc_number = IntegerField(null=True)  # may be null if no disc number available
    duration_ms = IntegerField(null=True)  # may be null if no duration info available
    explicit = BooleanField(null=True)  # may be null if no explicit info available
    isrc = TextField(null=True)  # may be null if no ISRC available
    spotify_external_url = TextField(null=True)  # may be null if no external URL available
    spotify_href = TextField(null=True)  # may be null if no href available
    popularity = IntegerField(null=True)  # may be null if no popularity info available
    track_number = IntegerField(null=True)  # may be null if no track number available
    spotify_uri = TextField(null=True)  # may be null if no Spotify URI available
    lyrics = TextField(null=True)  # may be null if no lyrics available

class SongAudioFeatures(BaseModel):
    song = ForeignKeyField(Songs, backref='audio_features', unique=True)
    # Audio features
    acousticness = FloatField(null=True)
    analysis_url = TextField(null=True)
    danceability = FloatField(null=True)
    duration_ms = IntegerField(null=True)
    energy = FloatField(null=True)
    instrumentalness = FloatField(null=True)
    key = IntegerField(null=True)
    liveness = FloatField(null=True)
    loudness = FloatField(null=True)
    mode = IntegerField(null=True)
    speechiness = FloatField(null=True)
    tempo = FloatField(null=True)
    time_signature = IntegerField(null=True)
    track_href = TextField(null=True)
    type = TextField(null=True)
    uri = TextField(null=True)
    valence = FloatField(null=True)
    # Track analysis metadata
    num_samples = IntegerField(null=True)
    duration = FloatField(null=True)
    sample_md5 = TextField(null=True)
    end_of_fade_in = FloatField(null=True)
    start_of_fade_out = FloatField(null=True)
    tempo_confidence = FloatField(null=True)
    time_signature_confidence = FloatField(null=True)
    key_confidence = FloatField(null=True)
    mode_confidence = FloatField(null=True)
    # Structural analysis (JSON arrays)
    bars = JSONField(null=True)
    beats = JSONField(null=True)
    sections = JSONField(null=True)
    segments = JSONField(null=True)
    tatums = JSONField(null=True)
    # Embeddings
    image_embeddings = VectorField(dimensions=768, null=True)
    lyrics_embeddings = VectorField(dimensions=768, null=True)