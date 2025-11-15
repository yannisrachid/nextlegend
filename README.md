# NextLegend

NextLegend is a Streamlit-based football scouting platform powered by
Wyscout data exports with a dedicated data-engineering pipeline. The
application lives in the `nextlegend/` directory and relies on the
datasets produced by the preparation scripts.

## Prerequisites

The project runs on **Python 3.10+**. It is recommended to work within
the provided virtual environment (`source venv/bin/activate`).
Dependencies can be installed through:

    pip install -r nextlegend/requirements.txt

NextLegend uses **AWS S3** as its primary data layer. The following
environment variables must be configured:

    AWS_ACCESS_KEY_ID
    AWS_SECRET_ACCESS_KEY
    AWS_DEFAULT_REGION
    NEXTLEGEND_S3_BUCKET

## Running the Application

Activate the environment and launch Streamlit:

    source venv/bin/activate
    streamlit run nextlegend/Home.py

Always run the full data pipeline beforehand to ensure that the
application loads the latest scores, percentiles and similarity files.

## Weekly Data Refresh

NextLegend is designed around a weekly refresh workflow. The standard
procedure is:

**1. Clean the raw dataset**

    python nextlegend/scripts/clean_dataset.py

**2. Upload the cleaned file to S3**

    python nextlegend/scripts/upload_cleaned_to_s3.py --input ./nextlegend/data/wyscout_players_cleaned.csv

**3. Generate scores, percentiles, summaries and similarity matrices**

    python nextlegend/scripts/build_roles_pipeline.py --raw_in "" --in data/wyscout_players_cleaned.csv

**4. Enrich with Transfermarkt metadata and write back to S3**

    python nextlegend/scripts/transfermarkt_matching.py
    

## Project Structure Overview

NextLegend is built around three primary components.

### Application (Streamlit)

The Streamlit interface is located in `nextlegend/Home.py` and the
`nextlegend/pages/` directory. It contains the following modules: player
reporting, rankings, comparisons, projection modelling, exploratory
statistics, vizualisation dashboards and a prospect manager. A shared
sidebar component ensures consistent branding, and the global dark theme
(`nextlegend/.streamlit/config.toml`) maintains a unified visual
identity.

### Data Preparation Pipeline

The core pipeline is implemented in
`nextlegend/scripts/build_roles_pipeline.py`. It performs dataset
standardisation, scoring, role attribution, percentile computation
(league and global), summary metric generation and similarity modelling.
All resulting datasets are exported back to S3, where they are consumed
by the application.

### Infrastructure and Layout

Styling assets (CSS, theming), execution helpers (such as
`run_nextlegend.sh`) and S3 integration form the supporting
infrastructure. Raw data is fetched from S3 when available, but local
processing through `nextlegend/data/` remains fully supported.

The pipeline automatically regenerates `wyscout_players_cleaned.csv` at
each execution and writes all downstream outputs (scores, percentiles,
similarity matrices and enriched CSV files) to S3.

## Notes and Recommendations

Always update the datasets through the pipeline before launching the
application. This ensures consistency between the scoring logic,
similarity computations and league-difficulty adjustments.

When adding new roles or metrics, update `player_profiles.json`
carefully, as it defines metric weights, inversion logic and eligibility
thresholds.

Eligibility relies on two constraints: a player must have played at
least **15% of available league minutes** for league-level percentiles,
and at least **270 minutes globally** for global distributions. These
values are configurable within the pipeline.

The file `league_translation_meta.csv` enables league-difficulty
adjustments used in the Projection module. Without it, the system
defaults to a neutral coefficient.
