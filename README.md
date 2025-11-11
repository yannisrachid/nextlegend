# NextLegend

Plateforme Streamlit de scouting football alimentée par les exports Wyscout. L’application vit dans `nextlegend/` et se base sur les jeux de données générés par les scripts de préparation décrits dans `codex/documentation.MD`.

## Pré-requis

- Python 3.10+ (utiliser le venv `source venv/bin/activate`).
- Dépendances : `pip install -r nextlegend/requirements.txt`.
- Accès AWS S3 (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`) et variable `NEXTLEGEND_S3_BUCKET` pointant sur le bucket `nextlegend`.

## Lancer l’application

```bash
source venv/bin/activate
streamlit run nextlegend/Home.py
```

Assurez-vous d’avoir exécuté le pipeline de données avant d’ouvrir Streamlit pour charger les derniers scores/percentiles.

## Rafraîchissement hebdomadaire des données

1. Nettoyer `data/wyscout_players_final.csv` via `python nextlegend/scripts/clean_dataset.py`.
2. Uploader `nextlegend/data/wyscout_players_cleaned.csv` via `python nextlegend/scripts/upload_cleaned_to_s3.py --input ./nextlegend/data/wyscout_players_cleaned.csv`.
3. Générer scores, percentiles, similarités avec `python nextlegend/scripts/build_roles_pipeline.py --raw_in "" --in data/wyscout_players_cleaned.csv`.
4. Exécuter `python nextlegend/scripts/transfermarkt_matching.py` pour fusionner les métadonnées Transfermarkt et réécrire le CSV enrichi sur S3.

Les commandes détaillées, vérifications et conseils de dépannage sont regroupés dans `codex/data_refresh_runbook.MD`.
