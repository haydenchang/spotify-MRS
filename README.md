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
  
## Planned Features
* Exploartory Data Analysis (EDA)
* Machine Learning Models
* End-to-End Pipeline
* A/B Testing
* Visualization & Dashboard
## Tech Stack
* Python, SQL, Pandas, Scikit-learn
* Streamlit, FastAPI
* MLflow, prefect

