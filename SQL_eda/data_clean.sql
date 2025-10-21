WITH cte AS (
	SELECT *,
		ROW_NUMBER() OVER (PARTITION BY track_id ORDER BY track_name) AS row_number
	FROM [DS].[dbo].[raw_spotify_song_info.csv]
)
DELETE FROM cte WHERE row_number > 1;

DELETE FROM [DS].[dbo].[raw_spotify_song_info.csv]
WHERE track_name IS NULL OR energy IS NULL;

