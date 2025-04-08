SELECT
  user_pseudo_id,
  geo.country AS country,
  app_info.id AS app_id,
  CAST(DATE(TIMESTAMP_MICROS(user_first_touch_timestamp)) AS DATE) AS first_open,
  COUNT(*) AS engagement_event_count,
  SUM((
    SELECT value.int_value
    FROM UNNEST(event_params)
    WHERE key = "engagement_time_msec"
  )) / 1000 AS total_time_seconds
FROM 
  `ftm-b9d99.analytics_159643920.events_202*`
WHERE
  event_name = 'user_engagement'
  AND app_info.id = 'org.curiouslearning.container'
  AND CAST(DATE(TIMESTAMP_MICROS(user_first_touch_timestamp)) AS DATE) BETWEEN '2023-01-01'
  AND CURRENT_DATE()

GROUP BY
  user_pseudo_id,
  geo.country,
  app_info.id,
  first_open
ORDER BY total_time_seconds DESC
