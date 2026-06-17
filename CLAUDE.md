# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Streamlit dashboard for Curious Learning's user engagement analysis. Reads pre-aggregated user/session Parquet snapshots from GCS to produce country-, device-, and language-level engagement views. Python 3.12.

## Running and deploying

```bash
pip install -r requirements.txt
streamlit run main.py
```

Local runs still require GCP credentials — `settings.get_gcp_credentials()` reads a service account from Secret Manager (`projects/405806232197/secrets/service_account_json`) and authenticates against BigQuery project `dataexploration-193817`. There is no mock/offline mode; without GCP access, the app fails on data load.

Container build/push (deploy target is Cloud Run via GCR):
```bash
docker build --no-cache --platform linux/amd64 -t gcr.io/dataexploration-193817/cl-dashboard-engagement:latest .
docker push gcr.io/dataexploration-193817/cl-dashboard-engagement:latest
```
Note: the Dockerfile `git clone`s from GitHub at build time rather than `COPY`ing local files, so local edits don't ship until pushed to `main`.

There is no test suite, linter config, or type checker wired up.

## Architecture

**Entry point.** `main.py` uses `st-pages` to build the nav from `.streamlit/pages.toml`. Add a page by creating a file under `app_pages/` and registering it in `pages.toml` — there is no auto-discovery.

**Data load, gated on `st.session_state`:**

`settings.init_data()` → `users.ensure_user_data_initialized()` loads three Parquet snapshots from GCS (`user_data_parquet_cache/...`): unity user progress, day-1 uninstalls, CR app launches with device data. Deduplicates, filters by `start_date` (2024-05-01), removes day-1 uninstalls, and stores `df_unity_users` + `df_cr_app_launch` in session state.

**Every page file must call `initialize()` and `init_data()` at the top** — these are the gating calls that populate `st.session_state`. Page code then reads `st.session_state["df_cr_app_launch" | "df_unity_users"]` directly; data is not passed in.

**Caching strategy.** Everything uses `@st.cache_data(ttl="1d")` or `@st.cache_resource(ttl="1d")`. The BQ client and GCP credentials are `cache_resource`; dataframes are `cache_data`. The 1-day TTL is the refresh interval for the whole dashboard. When adding new BQ/GCS reads, follow this pattern or you'll re-query on every Streamlit rerun.

**Two user populations, two `user_id` columns.** Pages that work with either Unity or CR users (e.g. `cr_engagement.py`) reassign `df["user_id"]` from `user_pseudo_id` (Unity) or `cr_user_id` (CR) before passing the frame to `ui_components`. Components downstream assume the `user_id` column exists.

## File map

- `settings.py` — GCP auth, logger, `init_data()`, the canonical `default_daterange` / `start_date`.
- `users.py` — GCS Parquet loaders, user-data init, BigQuery dropdown lookups (country/language lists).
- `ui_components.py` — all Plotly chart helpers (histograms, scatter, pareto, box plot, device analysis). Components take a dataframe + a `key` string; pass unique keys per page or Streamlit will complain.
- `ui_widgets.py` — small reusable widgets (selectors, CSV download helper).
- `app_pages/` — one file per nav entry; thin glue that pulls from `session_state` and calls into `ui_components`.
- `Queries/cr_app_launch_device_data.sql` — reference SQL for the CR Parquet snapshot in GCS. Not run by the app; the Parquet is produced by an external pipeline.
