# 🚀 Projet : NextLegend — Plateforme d’analyse & scouting football (Streamlit)

## 🎯 Objectif global

Générer une webapp **Streamlit** nommée **NextLegend** qui reproduit le corps complet d’une application d’analyse/scouting de joueurs, avec :

1. **Dashboard “Joueur du jour”**,
2. **Analyse de joueurs**,
3. **Comparaison** (multi-joueurs et multi-saisons),
4. **Ranking** (classements par poste/championnat),
5. **Scouting / Recherche avancée** (filtres, similarité),
6. **Pages techniques** (À propos / Changelog minimal / Mentions).

**Important :**

* Ne JAMAIS utiliser le nom “DataScout” ni ses marques. Tout le branding = **NextLegend**.
* Source unique des données : **`wyscout_players_final.csv`** (présent localement dans le dossier data/ du projet).
* Si des colonnes manquent, **détecter automatiquement** les noms proches et proposer un **mapping configurable**.
* App responsive, thème sombre, esthétique moderne.

---

## 🧱 Stack & librairies

* **Python 3.10+**, **Streamlit** (1.33+)
* **pandas**, **numpy**, **scikit-learn** (standardisation, similarité), **plotly** (radars, barres, distributions), **pydeck** (optionnel cartes)
* **mplsoccer** (librairie python de génération de radar, visualisations, pizza chart, voir des exemples dans les notebooks dans le dossier old_work)
* **streamlit-option-menu** (navigation), **streamlit-aggrid** ou **st.dataframe** (tables interactives)
* **pydantic** (validation config), **toml**/**yaml** (config mapping colonnes)
* **rapidfuzz** (fuzzy matching de noms de colonnes)
* **cache** : `st.cache_data` / `st.cache_resource`

---

## 📁 Arborescence cible

```
nextlegend/
├─ app.py
├─ pages/
│  ├─ 01_Joueur_du_jour.py
│  ├─ 02_Analyse_de_joueurs.py
│  ├─ 03_Comparaison.py
│  ├─ 04_Ranking.py
│  ├─ 05_Scouting.py
│  ├─ 90_A_propos.py
│  └─ 99_Changelog.py
├─ data/
│  └─ wyscout_players_final.csv
├─ config/
│  ├─ schema_map.yaml
│  └─ settings.toml
├─ components/
│  ├─ charts_radar.py
│  ├─ charts_bars.py
│  ├─ metrics_cards.py
│  ├─ player_badges.py
│  ├─ similarity.py
│  └─ tables.py
├─ assets/
│  ├─ logo_nextlegend.png
│  └─ favicon.ico
└─ styles/
   └─ theme.css
```

---

## 🔌 Données & Schéma (basé Wyscout)

**Fichier :** `data/wyscout_players_final.csv`
**Exemples de colonnes attendues (adaptables) :**

* Identité : `player_id`, `player_name`, `age`, `height_cm`, `nationality`, `strong_foot`
* Contexte : `team`, `league`, `season`, `minutes`, `matches`
* Postes : `position`, `role_primary`, `role_secondary`
* Création : `assists`, `xA`, `key_passes`, `deep_completions`, `crosses_completed`
* Finition : `goals`, `npxG`, `shots_on_target`, `conversion_rate`
* Portage/Dribbles : `dribbles_attempted`, `dribbles_succeeded`, `progressive_runs`, `progressive_carries`
* Construction/Passes : `passes_completed`, `passes_attempted`, `progressive_passes`, `final_third_passes`, `long_pass_accuracy`
* Défense : `interceptions`, `tackles_sliding`, `def_duels_won`, `def_duels_total`, `blocks`, `fouls`
* Aérien : `aerial_duels_won`, `aerial_duels_total`, `headed_goals`
* Discipline : `yellow_cards`, `red_cards`

**Exigences :**

* Charger le CSV, convertir types, gérer valeurs manquantes.
* Créer **KPIs par 90** : `stat_per90 = stat * 90 / minutes` (sécuriser `minutes > 0`).
* Créer **centiles** par **cohorte** (au minimum par **poste**; idéalement par `poste + ligue` et éventuellement **seuil minutes** configurable, ex. `minutes >= 450`).
* Normaliser (z-score) les features utiles pour similarité.
* **Mapping de schéma** automatique : si la colonne n’existe pas, tenter `rapidfuzz` pour trouver la plus proche et alerter l’utilisateur (panneau config).

---

## 🧭 Navigation de l’app

* Barre latérale : logo NextLegend + sélecteurs globaux (saison, ligue, seuil minutes), commutateur thème clair/sombre (si possible via CSS).
* **Onglets/pages** :

  1. **Joueur du jour** (dashboard focalisé)
  2. **Analyse de joueurs** (profil complet)
  3. **Comparaison** (jusqu’à 4 joueurs / 2 saisons)
  4. **Ranking** (classements filtrables)
  5. **Scouting** (recherche multicritères + similarité)
  6. **A propos / Changelog** (texte simple)

---

## 📊 Composants & visuels

### Charts Radar (plotly)

* Fonction `radar_centiles(player_row, feature_groups, cohort_frame)`
* Couleur primaire NextLegend, tracés semi-transparents pour comparaisons, tooltip activé.
* Deux modes : **Centiles** (par défaut) et **Valeurs per90** (toggle).

### Barres horizontales par catégorie

* Catégories : **Finition**, **Création**, **Offensif**, **Construction**, **Précision**, **Aérien**, **Défense**.
* Entrées = centiles (0–100) calculés dans la cohorte active.
* Option “afficher valeurs exactes” (per90) sous la barre.

### Cartes de métriques (metrics_cards)

* `matches`, `minutes`, `goals`, `assists`, `xG`, `xA`, `succ_dribbles`, `prog_runs`, `def_duels_win_rate`, `aerial_win_rate`, etc.
* KPI composite **Performance Index** (pondérations configurables, cf. `settings.toml`).

### Badges joueur (player_badges)

* Âge, taille, pied fort, postes, pays, club, saison.

### Tables

* **Tables interactives** (AgGrid ou DataFrame) avec tri, filtre, export CSV.

---

## 🧩 Pages — Comportement détaillé

### 1) Joueur du jour (`01_Joueur_du_jour.py`)

* Sélection **aléatoire pondérée** (ex. joueurs avec minutes ≥ seuil) OU “dernier profil consulté”.
* Afficher :

  * En-tête (nom, club, poste, saison), **Performance Index**, **Meilleur rôle** (déduit du poste/role_primary)
  * **Forces / Axes d’amélioration** :

    * Forces = top 5 centiles > 80 (sur un pool de features clés)
    * Axes = 5 plus faibles centiles (< 30)
  * **Radar centiles** (features clés)
  * **Barres par catégorie**
  * Bloc "métadonnées" (matches, minutes, etc.)

### 2) Analyse de joueurs (`02_Analyse_de_joueurs.py`)

* Filtres : saison, ligue, équipe, poste, nom joueur.
* Panneau gauche : carte identité + métriques.
* Panneau droit :

  * Radar (toggle centiles/valeurs)
  * Barres catégories
  * **Distribution de cohorte** (violin/box pour situer le joueur sur 1–2 features au choix)
* Bouton “Copier l’image du radar” (export PNG via `plotly.io.to_image`)

### 3) Comparaison (`03_Comparaison.py`)

* Choisir **jusqu’à 4 joueurs** (ou 1 joueur × 2 saisons).
* Radars superposés (couleurs distinctes).
* Tableau comparatif (centiles et valeurs per90) — colonnes = joueurs/saisons.
* Mini-heatmap de centiles pour lecture rapide.

### 4) Ranking (`04_Ranking.py`)

* Filtres : poste (obligatoire), saison, ligue(s), seuil minutes.
* Choix de **métrique de tri** (ex. `progressive_passes_per90`, `xA_per90`, `def_duels_win_rate`, **Performance Index**).
* Table triable + sparklines (si historique multi-saisons).
* Bouton “Enregistrer la shortlist” (écrit un CSV export dans `/data/exports/`).

### 5) Scouting (`05_Scouting.py`)

* Filtres multiples : âge (min–max), taille, pied, poste, ligue, minutes, métriques (sliders), statut contrat (si dispo).
* **Similarité** :

  * Entrée = “Joueur référence” + **poids par catégorie** (Création/Défense/Offensif/Construction/Précision/Aérien).
  * Standardiser features, calcul **cosine similarity**.
  * Restituer top N profils similaires avec cartes + mini-radar par joueur.
* Toggle “même pied fort”, “même poste”.

### 6) À propos / Changelog

* Texte statique, version app, liens (internes).

---

## 🧮 Calculs & indicateurs

### Per 90

```
safe_per90(value, minutes) = value * 90 / max(minutes, 1)
```

### Taux (%)

```
win_rate = won / max(total, 1)
accuracy = completed / max(attempted, 1)
```

### Centiles par cohorte

* Définir cohorte par défaut : `poste` (+ option `ligue`) avec filtre `minutes >= seuil`.
* Utiliser `scipy.stats.rankdata` ou `pandas.qcut`/`percentileofscore`.
* Stocker centiles pour chaque *feature clé* dans un DataFrame “long” exploitable par les charts.

### Performance Index (exemple)

* Pondérations configurables (`settings.toml`) par catégorie :

  * Création 25%, Offensif 15%, Construction 15%, Précision 10%, Défense 20%, Aérien 10%, Finition 5%.
* Calculer par moyenne pondérée de centiles.

---

## ⚙️ Fichiers de config

### `config/schema_map.yaml`

* Dictionnaire `nom_interne: [candidats_csv]`
* Exemple :

```yaml
player_name: ["player_name", "name", "player"]
minutes: ["minutes", "mins_played", "time_played"]
goals: ["goals", "goals_total"]
xA: ["xa", "expected_assists", "x_assist"]
progressive_passes: ["progressive_passes", "prog_passes"]
```

### `config/settings.toml`

```toml
[min_thresholds]
minutes = 450

[index_weights]
finition = 0.05
creation = 0.25
offensif = 0.15
construction = 0.15
precision = 0.10
defense = 0.20
aerien = 0.10

[ui]
theme = "dark"
primary_color = "#7BD389"
accent_color = "#FF7A59"
```

---

## 🧰 Composants à générer (stubs + code)

* `components/charts_radar.py`

  * `make_radar(df_centiles_long, entities, features_groups, mode="centile") -> plotly.Figure`
* `components/charts_bars.py`

  * `bars_by_category(centiles_dict: dict[str, dict[str,int]]) -> list[plotly.Figure]`
* `components/metrics_cards.py`

  * `render_metrics(player_row)`
* `components/player_badges.py`

  * `render_badges(player_row)`
* `components/similarity.py`

  * `standardize(df, features)`
  * `cosine_sim_matrix(df_std[features])`
  * `top_similar(df_std, ref_player_id, features, top_n=20, same_foot=None, same_position=None, weights_by_category=None)`
* `components/tables.py`

  * `ranking_table(df, sort_metric, ascending=False)`
  * `comparison_table(df_players, features)`

---

## 🧪 Exigences UX

* **Thème sombre** + **CSS** personnalisé (`styles/theme.css`) : coins arrondis, cartes, micro-ombres, espaces.
* Éléments UI : selectbox, multiselect, sliders, radio/toggles (“Centiles” vs “Valeurs per90”), boutons “Exporter PNG/CSV”.
* **Performance** : cache des DataFrames dérivés (per90, centiles, index).
* **Robustesse** : si colonne manquante → afficher **alerte + suggestion mapping**.

---

## ✅ Critères d’acceptation

* Lancement avec `streamlit run app.py`.
* Les pages chargent sans erreur avec `wyscout_players_final.csv`.
* Un joueur affiche radar + barres + forces/faiblesses + métriques.
* Comparaison superpose correctement les radars et aligne les tableaux.
* Ranking trie et filtre sans lag (>5k lignes OK).
* Scouting renvoie une **liste de joueurs similaires** crédible (+ filtres).

---

## 🧭 Tâches de génération pour Codex (ordre)

1. Créer l’arborescence et les fichiers vides.
2. Implémenter `app.py` (page d’accueil + routing Streamlit multi-pages).
3. Implémenter **chargement CSV + mapping dynamique** (`schema_map.yaml`).
4. Implémenter calculs per90, centiles, index.
5. Générer **01_Joueur_du_jour** (avec logic joueur aléatoire et UI complète).
6. Générer **02_Analyse_de_joueurs** (profil complet).
7. Générer **03_Comparaison** (jusqu’à 4 joueurs + tableau).
8. Générer **04_Ranking** (filtres + tri + export).
9. Générer **05_Scouting** (filtres + similarité + mini-radars).
10. Ajouter **styles/theme.css** et finitions visuelles.

---

## ▶️ Commande de lancement

```bash
streamlit run app.py
```

**Nom de l’app, logos, textes, couleurs** : utiliser exclusivement la marque **NextLegend**.
