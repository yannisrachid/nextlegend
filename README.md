# NextLegend

This repo contains two separate applications:

## 1) `nextlegend/` (Streamlit POC)
The `nextlegend/` folder is the legacy Streamlit app used for:
- POC work, user validation, and quick experiments
- internal testing and offline analysis

It is not the production app. Keep it stable unless explicitly asked to change it.

### Running the Streamlit POC
Prereqs: Python 3.10+, venv recommended.

```bash
source venv/bin/activate
pip install -r nextlegend/requirements.txt
streamlit run nextlegend/Home.py
```

The POC relies on Wyscout exports and a local/S3 pipeline. If you use it, refresh the data first:
```bash
python nextlegend/scripts/clean_dataset.py
python nextlegend/scripts/upload_cleaned_to_s3.py --input ./nextlegend/data/wyscout_players_cleaned.csv
python nextlegend/scripts/build_roles_pipeline.py --raw_in "" --in data/wyscout_players_cleaned.csv
python nextlegend/scripts/transfermarkt_matching.py
```

## 2) `new_nextlegend/` (Production app)
The `new_nextlegend/` folder is the real production application.
It is a full-stack web product with:
- Frontend: Next.js
- Backend: FastAPI
- Database: Postgres
- Batch pipeline: Docker job (scores, percentiles, similarities)
- Transfermarkt enrichment
- External S3 storage
- Agentic AI features (scouting assistant and AI workflows)

Production URLs:
- Frontend: `https://app.nextlegend.fr`
- API: `https://api.nextlegend.fr`

Auth model:
- Session cookie (`nl_session`) via the API.
- Frontend waits for `/auth/me` before redirecting (no auth loop).

Development entry points:
- `new_nextlegend/README.md`
- `new_nextlegend/docs/README.md`
- `new_nextlegend/apps/api/README.md`

### Quick start (dev)
```bash
cd new_nextlegend
cp .env.example .env
sudo docker compose --env-file .env -f infra/compose/docker-compose.yml up --build
```

---

## Notes
- `nextlegend/` is the Streamlit POC for validation and testing.
- `new_nextlegend/` is the production app with the full stack and agentic AI.
