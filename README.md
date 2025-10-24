# Spotify-Music-Recommendation-System

## Overview
This project is an end-to-end recommendation system based on Spotify audio data from Kaggle with SQL EDA, ML models, A/B testing. Goal : help users discover new good songs.
## Dataset
* Source : Kaggle data set(Spotify audio features)
* Overview : The dataset contains core audio features similar to those provided by the Spotify Web API
    - track_id : Unique identifier assigned by Spotify to each track. Used as a primary key for joins.
    - track_name : The song’s title. May include remix or live version labels.
    - acousticness : Confidence measure of whether the track is acoustic (closer to 1.0 = more acoustic).
    - danceability : Describes how suitable a track is for dancing based on tempo, rhythm stability, and beat strength.
    - duration_ms : Track duration in milliseconds.
    - energy : Represents intensity and activity; high values feel fast and loud.
    - instrumentalness : Predicts whether a track contains no vocals (values close to 1.0 = instrumental).
    - key : Musical key of the track (0 = C, 1 = C♯/D♭, … 11 = B).
    - liveness : Detects the presence of an audience; high values suggest a live performance.
    - loudness : Average loudness of the track in decibels. Higher (closer to 0) = louder.
    - mode : Indicates modality: 1 = Major, 0 = Minor. Major often sounds “happier.”
    - speechiness : Measures the presence of spoken words. High = more speech-like (e.g., rap or podcasts).
    - tempo : Estimated beats per minute of the song.
    - time_signiture : Estimated beats per bar (4 = common time, 3 = waltz).
    - valence : Describes the musical positivity or mood (high = happy/cheerful, low = sad/tense).
    - popularity : Spotify’s popularity index based on play count and recency.

## Data Cleaning Process (SQL Server)
1. Null Value Check
    * Used UNPIVOT to count missing values for every column.
    * Verified no major missingness across key numeric features.
2. Data Type Fixing
    * Converted imported NVARCHAR fields to correct numeric types (FLOAT, INT, TINYINT, BIT) using TRY_CONVERT.
    * Created a new clean table [spotify_song_info.csv] from the raw source.
3. Duplicate Removal
    * Identified duplicate track_ids using GROUP BY + HAVING COUNT > 1.
    * Used a CTE with ROW_NUMBER() to keep one record per track.
4. Realistic Value Validation
    * Checked that numerical features fell within expected Spotify ranges:
        ** Acoustic features (0–1)
        ** Loudness (−60 to 0 dB)
        ** Tempo (0–300 BPM)
        ** Key (−1–11)
        ** Mode ∈ {0, 1}
        ** Time Signature (3–7)
    * Flagged 2,103 invalid rows; corrected data rather than dropping when possible.
5. Encoding Fix
    * Detected that time_signature used 0/1 encoding for 3/4 and 4/4 meters.
    * Updated values: 0 → 3, 1 → 4 for consistency with Spotify’s definition.
6. Final Verification
    * Confirmed no duplicates, correct data types, realistic ranges, and valid encodings.
    * Saved clean dataset for EDA and ML stages.

## Planned Features
* Exploartory Data Analysis (EDA)
* Machine Learning Models
* Neural Network Architecture
* End-to-End Pipeline
* A/B Testing
* Visualization & Dashboard

## Tech Stack
* Python, SQL, Pandas, Scikit-learn
* Streamlit, FastAPI
* MLflow, prefect

