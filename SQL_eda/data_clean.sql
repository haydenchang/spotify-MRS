-- Find Null values
SELECT column_name, COUNT(*) AS null_count
FROM [DS].[dbo].[raw_spotify_song_info.csv]
UNPIVOT (
    value FOR column_name IN (track_id, track_name, acousticness,
        danceability, duration_ms, energy, instrumentalness, 
        liveness, loudness, mode, speechiness, tempo, time_signature,
        valence, popularity)
) AS u
WHERE value IS NULL
GROUP BY column_name;

-- Data type
SELECT COLUMN_NAME, DATA_TYPE
FROM INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_NAME = 'raw_spotify_song_info.csv';

-- Fix data type
SELECT
    track_id,
    track_name,
    TRY_CONVERT(FLOAT, acousticness)     AS acousticness,
    TRY_CONVERT(FLOAT, danceability)     AS danceability,
    TRY_CONVERT(INT,   duration_ms)      AS duration_ms,
    TRY_CONVERT(FLOAT, energy)           AS energy,
    TRY_CONVERT(FLOAT, instrumentalness) AS instrumentalness,
    TRY_CONVERT(TINYINT, [key])          AS [key],
    TRY_CONVERT(FLOAT, liveness)         AS liveness,
    TRY_CONVERT(FLOAT, loudness)         AS loudness,
    TRY_CONVERT(BIT,   [mode])           AS [mode],
    TRY_CONVERT(FLOAT, speechiness)      AS speechiness,
    TRY_CONVERT(FLOAT, tempo)            AS tempo,
    TRY_CONVERT(TINYINT, time_signature) AS time_signature,
    TRY_CONVERT(FLOAT, valence)          AS valence,
    TRY_CONVERT(TINYINT, popularity)     AS popularity
INTO [DS].[dbo].[spotify_song_info.csv]
FROM [DS].[dbo].[raw_spotify_song_info.csv]

-- Check duplicate
SELECT track_id, COUNT(*) AS count
FROM [DS].[dbo].[spotify_song_info.csv]
GROUP BY track_id
HAVING COUNT(*) > 1;

-- Delete duplicate
WITH cte AS (
	SELECT *,
		ROW_NUMBER() OVER (PARTITION BY track_id ORDER BY track_name) AS row_number
	FROM [DS].[dbo].[spotify_song_info.csv]
)
DELETE FROM cte WHERE row_number > 1;

-- Realistic value
SELECT *
FROM [DS].[dbo].[spotify_song_info.csv]
WHERE
    danceability     < 0 OR danceability     > 1
 OR acousticness     < 0 OR acousticness     > 1
 OR energy           < 0 OR energy           > 1
 OR instrumentalness < 0 OR instrumentalness > 1
 OR liveness         < 0 OR liveness         > 1
 OR loudness         < -60 OR loudness       > 0
 OR speechiness      < 0 OR speechiness      > 1
 OR valence          < 0 OR valence          > 1
 OR tempo            < 0 OR tempo            > 300
 OR time_signature   < 3 OR time_signature   > 7
 OR popularity       < 0 OR popularity       > 100
 OR [key]            < -1 OR [key]            > 11
 OR [mode]           NOT IN (0,1)
 OR duration_ms      <= 0;

-- Fix time_signature
SELECT time_signature, COUNT(*) AS count
FROM [DS].[dbo].[spotify_song_info.csv]
GROUP BY time_signature
ORDER BY time_signature;

UPDATE [DS].[dbo].[spotify_song_info.csv]
SET time_signature =
    CASE
        WHEN time_signature = 0 THEN 3
        WHEN time_signature = 1 THEN 4
        ELSE time_signature
    END;