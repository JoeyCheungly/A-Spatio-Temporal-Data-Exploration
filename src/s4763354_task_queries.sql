
-- Building of database was done in Python. Please refer to the first secion "Import to Postgres" in "Task1_algorithms.py" file. 



----------------------------------------------------- Task 1 ----------------------------------------------------------
-- Find all check-ins in Bars within Brooklyn during New Year’s Eve and New Year’s Day. 
SELECT *
FROM db_checkin
WHERE "venueCategory" = 'Bar'
AND latitude BETWEEN  40.5707 AND 40.7395
AND longitude BETWEEN - 73.9764 AND - 73.8331
AND CAST("utcTimestamp" AS timestamp with time zone)AT TIME ZONE 'UTC' BETWEEN '2012-12-31 00:00:00 +0000' AND '2013-01-01 23:59:59 +0000';

-- Find all check-ins in Parks within Manhattan during the first two days of 2013
SELECT *
FROM db_checkin
WHERE "venueCategory" = 'Park'
AND latitude BETWEEN  40.68 AND 40.88
AND longitude BETWEEN -74.02 AND -73.93 
AND CAST("utcTimestamp" AS timestamp with time zone)AT TIME ZONE 'UTC' BETWEEN '2013-01-01 00:00:00 +0000' AND '2013-01-02 23:59:59 +0000';




----------------------------------------------------- Task 2 ----------------------------------------------------------
-- Find all check-ins within a maximum distance of 5 km to Museum of the City of New York in the afternoon of Independence Day (4th July).
SELECT *
FROM db_checkin
WHERE CAST("utcTimestamp" AS timestamp with time zone )AT TIME ZONE 'UTC' >= '2012-07-04 12:00:00 +0000'
AND CAST("utcTimestamp" AS timestamp with time zone )AT TIME ZONE 'UTC' <= '2012-07-04 17:00:00 +0000'
AND (6371 * 2 * ASIN(SQRT(POWER(SIN((RADIANS(latitude - 40.792406)) / 2), 2) + 
						  COS(RADIANS(latitude)) * COS(RADIANS(40.792406)) * 
						  POWER(SIN((RADIANS(longitude - -73.952038)) / 2), 2)
        )
    )
) <= 5   

-- Find all check-ins within a maximum distance of 3 km to Central Park in Valentines' Day.
SELECT *
FROM db_checkin
WHERE CAST("utcTimestamp" AS timestamp with time zone )AT TIME ZONE 'UTC' >= '2013-02-14 00:00:00 +0000'
AND CAST("utcTimestamp" AS timestamp with time zone )AT TIME ZONE 'UTC' <= '2013-02-14 23:59:59 +0000'
AND (6371 * 2 * ASIN(SQRT(POWER(SIN((RADIANS(latitude - 40.781512)) / 2), 2) + 
						  COS(RADIANS(latitude)) * COS(RADIANS(40.781512)) * 
						  POWER(SIN((RADIANS(longitude - -73.966515)) / 2), 2)
        )
    )
) <= 3





----------------------------------------------------- Task 3 ----------------------------------------------------------
-- Find 5 nearest neighbour of the the given check-in at location [40.78430372 , -73.97968281] within the same day which is 4th of April 2012. 
SELECT "index","userId","venueId","venueCategoryId","venueCategory","latitude","longitude","timezoneOffset","utcTimestamp","geom", distance
FROM (SELECT *,ST_DistanceSphere(
    public.ST_MakePoint(db_checkin.latitude, db_checkin.longitude),
    public.ST_MakePoint(40.78430372 , -73.97968281)) as distance
FROM db_checkin
WHERE CAST("utcTimestamp" AS Date ) = '2012-04-04'
AND latitude != 40.78430372 AND longitude != -73.9796828)subquery
ORDER BY distance LIMIT 5

-- Find 3 nearest neighbour of the the given check-in at location [40.96537083,-74.06281492] within the same day which is 14th of May 2012.
SELECT "index","userId","venueId","venueCategoryId","venueCategory","latitude","longitude","timezoneOffset","utcTimestamp","geom"
FROM (SELECT *,ST_DistanceSphere(
    public.ST_MakePoint(db_checkin.latitude, db_checkin.longitude),
    public.ST_MakePoint(40.96537083,-74.06281492)) as distance
FROM db_checkin
WHERE CAST("utcTimestamp" AS Date ) = '2012-05-14' 
	  AND latitude != 40.75673368 AND longitude != -73.95445704)subquery
ORDER BY distance LIMIT 3




----------------------------------------------------- Task 4 ----------------------------------------------------------
-- Find the number of people travelling from John F. Kennedy International Airport in each season.
SELECT CASE 
    WHEN EXTRACT(MONTH FROM CAST("utcTimestamp" AS date)) IN (12, 1, 2) THEN 'Winter'
    WHEN EXTRACT(MONTH FROM CAST("utcTimestamp" AS date)) IN (3, 4, 5) THEN 'Spring'
    WHEN EXTRACT(MONTH FROM CAST("utcTimestamp" AS date)) IN (6, 7, 8) THEN 'Summer'
    ELSE 'Fall'
  END AS season,
  COUNT(*) AS num_points
FROM db_checkin
WHERE latitude BETWEEN  40.660 AND 40.670
AND longitude BETWEEN -73.810 AND  -73.390
GROUP BY season	

--Find the number of people travelling from LaGuardia Airport in each season.
SELECT CASE 
    WHEN EXTRACT(MONTH FROM CAST("utcTimestamp" AS date)) IN (12, 1, 2) THEN 'Winter'
    WHEN EXTRACT(MONTH FROM CAST("utcTimestamp" AS date)) IN (3, 4, 5) THEN 'Spring'
    WHEN EXTRACT(MONTH FROM CAST("utcTimestamp" AS date)) IN (6, 7, 8) THEN 'Summer'
    ELSE 'Fall'
  END AS season,
  COUNT(*) AS num_points
FROM db_checkin
WHERE latitude BETWEEN  40.773110 AND 40.785975
AND longitude BETWEEN -73.890441 AND  -73.849975
GROUP BY season	




----------------------------------------------------- Task 5 ----------------------------------------------------------
-- Find top 5 most popular venues that people love to go to during Thanksgiving in Manhattan.
SELECT "venueCategory" as venue,
  COUNT(*) AS num_points
FROM db_checkin
WHERE CAST("utcTimestamp" AS Date ) = '2012-11-22'
AND latitude BETWEEN  40.68 AND 40.88
AND longitude BETWEEN -74.02 AND  -73.93
GROUP BY venue
ORDER BY num_points DESC
LIMIT 5

-- Find 3 most popular venues that people love to go to during Mother's Day in Brooklyn.
SELECT "venueCategory" as venue,
  COUNT(*) AS num_points
FROM db_checkin
WHERE CAST("utcTimestamp" AS Date ) = '2012-05-13'
AND latitude BETWEEN  40.5707 AND 40.7395
AND longitude BETWEEN -73.9764 AND  -73.8331
GROUP BY venue
ORDER BY num_points DESC
LIMIT 3