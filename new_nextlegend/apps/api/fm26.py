"""FM save provenance and conservative, namespaced identity resolution."""
import re
import unicodedata
from collections import defaultdict
from datetime import date
from sqlalchemy import text

SCHEMA = """
CREATE TABLE IF NOT EXISTS fm_imports (
 id BIGSERIAL PRIMARY KEY, source_sha256 TEXT UNIQUE NOT NULL,
 game_date DATE NOT NULL, build TEXT NOT NULL, known_build BOOLEAN NOT NULL,
 reader_version TEXT NOT NULL, extracted_at TIMESTAMPTZ NOT NULL,
 imported_at TIMESTAMPTZ NOT NULL DEFAULT now(), player_count INTEGER NOT NULL,
 metadata JSONB NOT NULL
);
CREATE TABLE IF NOT EXISTS fm_players (
 import_id BIGINT NOT NULL REFERENCES fm_imports(id), uid BIGINT NOT NULL,
 unique_id BIGINT, name TEXT NOT NULL, birth_date DATE, club_name TEXT,
 ca INTEGER CHECK(ca BETWEEN 0 AND 200), pa INTEGER CHECK(pa BETWEEN 0 AND 200),
 payload JSONB NOT NULL, PRIMARY KEY(import_id,uid)
);
CREATE INDEX IF NOT EXISTS fm_players_name_idx ON fm_players(lower(name));
CREATE INDEX IF NOT EXISTS fm_players_unique_idx ON fm_players(unique_id);
CREATE TABLE IF NOT EXISTS fm_player_matches (
 import_id BIGINT NOT NULL, fm_uid BIGINT NOT NULL,
 provider TEXT NOT NULL CHECK(provider IN ('wyscout','eyeball')),
 provider_id TEXT NOT NULL, status TEXT NOT NULL CHECK(status IN ('accepted','review','rejected')),
 method TEXT NOT NULL, evidence JSONB NOT NULL,
 reviewed_by TEXT, reviewed_at TIMESTAMPTZ,
 PRIMARY KEY(import_id,fm_uid,provider,provider_id),
 FOREIGN KEY(import_id,fm_uid) REFERENCES fm_players(import_id,uid)
);
CREATE UNIQUE INDEX IF NOT EXISTS fm_match_one_target_idx
 ON fm_player_matches(import_id,provider,provider_id) WHERE status='accepted';
CREATE UNIQUE INDEX IF NOT EXISTS fm_match_one_source_idx
 ON fm_player_matches(import_id,provider,fm_uid) WHERE status='accepted';
"""

def normalize(value):
    value = unicodedata.normalize('NFKD', str(value or '')).casefold()
    return ' '.join(re.sub(r'[^a-z0-9 ]', ' ', ''.join(c for c in value if not unicodedata.combining(c))).split())

def parse_date(value):
    try:
        return date.fromisoformat(str(value)[:10]) if value else None
    except ValueError:
        return None

def match_players(fm_rows, targets):
    """Never compare IDs across providers. Only unique corroborated identities are accepted."""
    by_name = defaultdict(list)
    by_initial = defaultdict(list)
    for r in fm_rows:
        for alias in {normalize(r.get(k)) for k in ('name','full_name','common_name')} - {''}:
            by_name[alias].append(r)
        first, last = normalize(r.get('first_name')), normalize(r.get('last_name'))
        if first and last: by_initial[first[0]+' '+last].append(r)
    proposals = []
    for target in targets:
        name = normalize(target['name'])
        dob = parse_date(target.get('birth_date'))
        alias = normalize(target.get('alias'))
        # Existing TM identity is corroboration only if its name agrees with Wyscout.
        parts, aparts = name.split(), alias.split()
        alias_ok = bool(alias and (name == alias or (len(parts)>1 and len(aparts)>1 and
            len(parts[0])==1 and parts[0]==aparts[0][0] and parts[1:]==aparts[1:])))
        if not dob and alias_ok:
            dob = parse_date(target.get('alternate_birth_date'))
        year = target.get('birth_year') or (dob.year if dob else None)
        inferred_years = set()
        observed = parse_date(target.get('observed_at'))
        source_age = target.get('source_age')
        if not year and observed and source_age is not None and 10 <= float(source_age) <= 65:
            estimate = observed.year - int(source_age)
            inferred_years = {estimate-1, estimate, estimate+1}
        candidates = []
        pool = {r['uid']:r for key in ({name,alias} if alias_ok else {name}) for r in by_name.get(key,[])}
        if len(parts)>1 and len(parts[0])==1:
            pool.update({r['uid']:r for r in by_initial.get(name,[])})
        for fm in pool.values():
            fd = parse_date(fm.get('birth_date'))
            if dob and fd and dob != fd:
                continue
            if year and fd and int(year) != fd.year:
                continue
            if inferred_years and fd and fd.year not in inferred_years:
                continue
            club = normalize(target.get('club_name'))
            same_club = bool(club and club == normalize(fm.get('club_name')))
            exact_dob = bool(dob and fd == dob)
            if not exact_dob and not ((year or inferred_years) and fd and same_club):
                continue
            candidates.append((fm, exact_dob, same_club))
        for fm, exact_dob, club in candidates:
            proposals.append(dict(fm_uid=fm['uid'],provider=target['provider'],provider_id=str(target['provider_id']),
                status='accepted' if len(candidates)==1 and (exact_dob or (year and not inferred_years and club and len(parts)>=2 and name in {normalize(fm.get('name')),normalize(fm.get('full_name'))})) else 'review',
                method='exact_name_dob' if exact_dob else ('name_age_club_review' if inferred_years else 'exact_name_year_club'),
                evidence=dict(name=target['name'],dob=str(dob) if dob else None,birth_year=year,
                    same_club=club,estimated_birth_years=sorted(inferred_years),candidate_count=len(candidates),fm_name=fm['name'])))
    reverse = defaultdict(set)
    for p in proposals:
        if p['status']=='accepted': reverse[(p['provider'],p['fm_uid'])].add(p['provider_id'])
    for p in proposals:
        if len(reverse[(p['provider'],p['fm_uid'])])>1:
            p['status']='review'
            p['evidence']['ambiguous_target']=True
    return proposals

def ensure_schema(session):
    session.execute(text(SCHEMA))
