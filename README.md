# Spotify-Music-Recommendation-System

## Overview
This project is an end-to-end recommendation system based on Spotify audio data from Kaggle with SQL EDA, ML models, A/B testing. Goal : help users discover new good songs.
## Dataset
* Source : Kaggle data set(Spotify audio features)
* Overview : The dataset contains core audio features similar to those provided by the Spotify Web API

| Column | Type | Description |
|:--------|:------|:-------------|
| **track_id** | *string* | Unique identifier assigned by Spotify to each track. |
| **track_name** | *string* | The song’s title (may include remix or live version labels). |
| **acousticness** | *float (0–1)* | Confidence measure of whether the track is acoustic (closer to 1.0 = more acoustic). |
| **danceability** | *float (0–1)* | Describes how suitable a track is for dancing based on tempo, rhythm stability, and beat strength. |
| **duration_ms** | *integer* | Track duration in milliseconds. |
| **energy** | *float (0–1)* | Represents intensity and activity; higher = more energetic. |
| **instrumentalness** | *float (0–1)* | Predicts whether a track contains no vocals. |
| **key** | *integer (−1–11)* | Musical key of the track (0 = C, 1 = C♯/D♭, … 11 = B; −1 = no key detected). |
| **liveness** | *float (0–1)* | Detects the presence of an audience; high values suggest live performance. |
| **loudness** | *float (−60–0 dB)* | Average loudness in decibels. Higher (closer to 0) = louder. |
| **mode** | *integer (0 or 1)* | Indicates modality: 1 = Major, 0 = Minor. |
| **speechiness** | *float (0–1)* | Measures the presence of spoken words. High = more speech-like (e.g., rap or podcasts). |
| **tempo** | *float (BPM)* | Estimated tempo of the track in beats per minute. |
| **time_signature** | *integer (3–7)* | Estimated beats per bar. (Originally encoded as 0/1, remapped to 3/4 and 4/4.) |
| **valence** | *float (0–1)* | Describes the musical positivity or mood (high = happy, low = sad). |
| **popularity** | *integer (0–100)* | Spotify’s popularity index based on play count and recency. |


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

