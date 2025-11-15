
import json
import pandas as pd
import numpy as np

def _zscore(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors='coerce')
    mu = s.mean()
    sd = s.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mu) / sd

def load_profiles(json_path: str):
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def _parse_positions(val):
    if isinstance(val, str):
        toks = [t.strip() for t in val.split(",") if t.strip()]
        return toks
    return []

def score_profiles(df: pd.DataFrame, profiles: dict, minutes_col: str = "minutes_played"):
    df = df.copy()

    # Ensure all required columns exist (create NaN if missing)
    required = set()
    for spec in profiles.values():
        required |= set(spec["weights"].keys())
        required |= set(spec.get("lower_is_better", []))

    for c in required:
        if c not in df.columns:
            df[c] = np.nan

    all_scores = []
    for prof_name, spec in profiles.items():
        cols = list(spec["weights"].keys())

        Z = pd.DataFrame({c: _zscore(df[c]) for c in cols})

        # invert lower-is-better
        for c in spec.get("lower_is_better", []):
            if c in Z.columns:
                Z[c] = -Z[c]

        Z = Z.fillna(0.0)

        w = pd.Series(spec["weights"])
        w = w / w.sum()
        score = Z.mul(w, axis=1).sum(axis=1)

        # minutes threshold
        mm = spec.get("min_minutes", 0)
        if minutes_col in df.columns:
            score = score.where(df[minutes_col] >= mm, np.nan)

        # Percentile scaling 0-100
        pct = score.rank(pct=True).mul(100)
        all_scores.append(pct.rename(prof_name))

    return pd.concat(all_scores, axis=1)

def best_profile_for_player(score_df: pd.DataFrame, top_n: int = 3):
    # Top-N profils quel que soit le poste
    return score_df.apply(lambda r: r.nlargest(top_n), axis=1)

def best_profile_by_position(df: pd.DataFrame, score_df: pd.DataFrame, profiles: dict, position_col: str = "position", top_n: int = 3):
    # Limite les profils aux position_groups compatibles avec le(s) poste(s) du joueur
    # position_col peut contenir plusieurs postes séparés par des virgules
    pos_list = df[position_col].apply(_parse_positions)

    # Pré-calcul: map profil -> set(position_groups)
    prof2pos = {name: set(spec.get("position_groups", [])) for name, spec in profiles.items()}

    rows = []
    for idx, allowed_positions in pos_list.items():
        row_scores = score_df.loc[idx]
        # profils compatibles
        allowed_profiles = [p for p, ps in prof2pos.items() if len(ps & set(allowed_positions)) > 0] or list(score_df.columns)
        topk = row_scores[allowed_profiles].nlargest(top_n)
        rows.append(topk)

    return pd.DataFrame(rows, index=score_df.index)

def tidy_scores(score_df: pd.DataFrame):
    # Long format: index=player row; columns: profile, score
    id_col = score_df.index.name or "row"
    out = score_df.reset_index().melt(id_vars=[score_df.index.name] if score_df.index.name else ["index"],
                                      var_name="profile", value_name="score")
    return out
