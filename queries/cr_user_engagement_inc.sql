-- Declare rolling window for incremental update
DECLARE run_start_date DATE DEFAULT DATE_SUB(CURRENT_DATE(), INTERVAL 3 DAY);
DECLARE run_end_date DATE DEFAULT CURRENT_DATE();

-- Step 1: Delete overlapping data
DELETE FROM `dataexploration-193817.user_data.cr_user_engagement_inc`
WHERE first_open BETWEEN run_start_date AND run_end_date;

-- Step 2: Insert fresh aggregated data
INSERT INTO `dataexploration-193817.user_data.cr_user_engagement_inc` (
  user_pseudo_id,
  country,
  app_id,
  first_open,
  engagement_event_count,
  total_time_seconds
)
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
  `ftm-b9d99.analytics_159643920.events_*`
WHERE
  _TABLE_SUFFIX BETWEEN FORMAT_DATE('%Y%m%d', run_start_date)
                   AND FORMAT_DATE('%Y%m%d', run_end_date)
  AND event_name = 'user_engagement'
  AND app_info.id = 'org.curiouslearning.container'
  AND CAST(DATE(TIMESTAMP_MICROS(user_first_touch_timestamp)) AS DATE)
      BETWEEN run_start_date AND run_end_date
GROUP BY
  user_pseudo_id,
  geo.country,
  app_info.id,
  first_open
ORDER BY total_time_seconds DESC;
