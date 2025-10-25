-- Data overview
SELECT
	COUNT(*) AS total_songs,
	COUNT(DISTINCT artist_name) AS artist_count
FROM [DS].[dbo].[spotify_song_info.csv]

-- mean, median, std, min, max for numeric values
SELECT
	column_name,
	avg_value	AS [avg],
	std_value	AS [std],
	min_value	AS [min],
	max_value	AS [max]
FROM (
	SELECT 'acousticness' AS column_name,
		ROUND(AVG(CAST(acousticness AS FLOAT)), 2)		AS avg_value,
		ROUND(STDEV(CAST(acousticness AS FLOAT)), 2)	AS std_value,
		MIN(acousticness)								AS min_value,
		MAX(acousticness)								AS max_value
	FROM [DS].[dbo].[spotify_song_info.csv]

	UNION ALL
	SELECT 'danceability' AS column_name,
		ROUND(AVG(CAST(danceability AS FLOAT)), 2)		AS avg_value,
		ROUND(STDEV(CAST(danceability AS FLOAT)), 2)	AS std_value,
		MIN(danceability)								AS min_value,
		MAX(danceability)								AS max_value
	FROM [DS].[dbo].[spotify_song_info.csv]

	UNION ALL
	SELECT 'duration_ms' AS column_name,
		ROUND(AVG(CAST(duration_ms AS FLOAT)), 2)				AS avg_value,
		ROUND(STDEV(CAST(duration_ms AS FLOAT)), 2)			AS std_value,
		MIN(duration_ms)								AS min_value,
		MAX(duration_ms)								AS max_value
	FROM [DS].[dbo].[spotify_song_info.csv]

	UNION ALL
	SELECT 'instrumentalness' AS column_name,
		ROUND(AVG(CAST(instrumentalness AS FLOAT)), 2)		AS avg_value,
		ROUND(STDEV(CAST(instrumentalness AS FLOAT)), 2)		AS std_value,
		MIN(instrumentalness)							AS min_value,
		MAX(instrumentalness)							AS max_value
	FROM [DS].[dbo].[spotify_song_info.csv]

	UNION ALL
	SELECT 'liveness' AS column_name,
		ROUND(AVG(CAST(liveness AS FLOAT)), 2)				AS avg_value,
		ROUND(STDEV(CAST(liveness AS FLOAT)), 2)				AS std_value,
		MIN(liveness)									AS min_value,
		MAX(liveness)									AS max_value
	FROM [DS].[dbo].[spotify_song_info.csv]

	UNION ALL
	SELECT 'loudness' AS column_name,
		ROUND(AVG(CAST(loudness AS FLOAT)), 2)				AS avg_value,
		ROUND(STDEV(CAST(loudness AS FLOAT)), 2)				AS std_value,
		MIN(loudness)									AS min_value,
		MAX(loudness)									AS max_value
	FROM [DS].[dbo].[spotify_song_info.csv]

	UNION ALL	
	SELECT 'speechiness' AS column_name,
		ROUND(AVG(CAST(speechiness AS FLOAT)), 2)				AS avg_value,
		ROUND(STDEV(CAST(speechiness AS FLOAT)), 2)			AS std_value,
		MIN(speechiness)								AS min_value,
		MAX(speechiness)								AS max_value
	FROM [DS].[dbo].[spotify_song_info.csv]

	UNION ALL
	SELECT 'tempo' AS column_name,
		ROUND(AVG(CAST(tempo AS FLOAT)), 2)					AS avg_value,
		ROUND(STDEV(CAST(tempo AS FLOAT)), 2)					AS std_value,
		MIN(tempo)										AS min_value,
		MAX(tempo)										AS max_value
	FROM [DS].[dbo].[spotify_song_info.csv]

	UNION ALL	
	SELECT 'time_signature' AS column_name,
		ROUND(AVG(CAST(time_signature AS FLOAT)), 2)			AS avg_value,
		ROUND(STDEV(CAST(time_signature AS FLOAT)), 2)		AS std_value,
		MIN(time_signature)								AS min_value,
		MAX(time_signature)								AS max_value
	FROM [DS].[dbo].[spotify_song_info.csv]

	UNION ALL
	SELECT 'valence' AS column_name,
		ROUND(AVG(CAST(valence AS FLOAT)), 2)					AS avg_value,
		ROUND(STDEV(CAST(valence AS FLOAT)), 2)				AS std_value,
		MIN(valence)									AS min_value,
		MAX(valence)									AS max_value
	FROM [DS].[dbo].[spotify_song_info.csv]

	UNION ALL
	SELECT 'popularity' AS column_name,
		ROUND(AVG(CAST(popularity AS FLOAT)), 2)				AS avg_value,
		ROUND(STDEV(CAST(popularity AS FLOAT)), 2)			AS std_value,
		MIN(popularity)									AS min_value,
		MAX(popularity)									AS max_value
	FROM [DS].[dbo].[spotify_song_info.csv]
) s
ORDER BY column_name;

-- Top tracks
SELECT
	artist_name,
	track_name,
	popularity,
	RANK() OVER (ORDER BY popularity DESC) AS track_rank
FROM [DS].[dbo].[spotify_song_info.csv];

-- Top artist
SELECT
	artist_name,
	ROUND(AVG(CAST(popularity AS float)), 2) AS avg_popularity,
	RANK() OVER (ORDER BY AVG(CAST(popularity AS FLOAT)) DESC) AS artist_rank
FROM [DS].[dbo].[spotify_song_info.csv]
GROUP BY artist_name
ORDER BY artist_rank;

-- Outlier
WITH stats AS (
	SELECT
		AVG(CAST(duration_ms AS FLOAT)) AS avg_duration,
		STDEV(CAST(duration_ms AS FLOAT)) AS std_duration
	FROM [DS].[dbo].[spotify_song_info.csv]
)
SELECT *
FROM (
	SELECT
		spotify.track_name,
		spotify.artist_name,
		spotify.duration_ms,
		CASE
			WHEN CAST(spotify.duration_ms AS FLOAT) > s.avg_duration + 3 * s.std_duration THEN 'High Outlier'
			WHEN CAST(spotify.duration_ms AS FLOAT) < s.avg_duration - 3 * s.std_duration THEN 'Low Outlier'
		END AS outlier_flag
	FROM [DS].[dbo].[spotify_song_info.csv] AS spotify
	CROSS JOIN stats AS s
) AS result
WHERE outlier_flag IS NOT NULL
ORDER BY duration_ms ASC;

