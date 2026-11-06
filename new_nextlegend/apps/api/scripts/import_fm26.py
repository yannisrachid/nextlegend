"""Idempotent local FM JSON -> PostgreSQL import. Never overwrites football facts."""
import argparse
import json
import sys
from pathlib import Path
from collections import Counter
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fm26 import ensure_schema, match_players

def main():
    import os
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    meta=json.loads(args.manifest.read_text())
    rows=[]; seen=set()
    engine=create_engine(os.environ['DATABASE_URL'])
    with Session(engine) as session, session.begin():
        session.execute(text("SELECT pg_advisory_xact_lock(hashtext('fm26_import'))"))
        ensure_schema(session)
        import_id=session.execute(text('''INSERT INTO fm_imports
          (source_sha256,game_date,build,known_build,reader_version,extracted_at,player_count,metadata)
          VALUES (:sha,:date,:build,:known,:reader,:extracted,:count,CAST(:meta AS JSONB))
          ON CONFLICT(source_sha256) DO UPDATE SET source_sha256=excluded.source_sha256 RETURNING id'''),
          dict(sha=meta['source_sha256'],date=meta['game_date'],build=meta['build'],known=meta['known_build'],
               reader=meta['reader_version'],extracted=meta['extracted_at'],count=meta['rows'],meta=json.dumps(meta))).scalar_one()
        insert=text('''INSERT INTO fm_players(import_id,uid,unique_id,name,birth_date,club_name,ca,pa,payload)
          VALUES (:import_id,:uid,:unique_id,:name,:birth_date,:club_name,:ca,:pa,CAST(:payload AS JSONB))
          ON CONFLICT(import_id,uid) DO NOTHING''')
        existing_count=session.execute(text('SELECT count(*) FROM fm_players WHERE import_id=:id'),{'id':import_id}).scalar_one()
        batch=[]
        with args.input.open() as source:
            records=(json.loads(line) for line in source if line.strip()) if args.input.suffix=='.jsonl' else iter(json.load(source))
            for r in records:
                if r['uid'] in seen:
                    raise ValueError('Duplicate FM UID; entire import rolled back')
                seen.add(r['uid'])
                rows.append({k:r.get(k) for k in ('uid','unique_id','name','full_name','first_name','last_name','common_name','birth_date','club_name')})
                if existing_count != meta['rows']:
                    batch.append(dict(import_id=import_id,uid=r['uid'],unique_id=r['unique_id'],name=r['name'],
                        birth_date=r['birth_date'],club_name=r['club_name'],ca=r['ability']['current'],pa=r['ability']['potential'],
                        payload=json.dumps(r,ensure_ascii=False)))
                if len(batch)>=500:
                    session.execute(insert,batch);batch=[]
            if batch: session.execute(insert,batch)
        if len(rows)!=meta['rows']:
            raise ValueError('FM manifest count mismatch; entire import rolled back')
        targets=[dict(r) for r in session.execute(text("""WITH latest AS (
          SELECT DISTINCT ON (ps.player_id) ps.player_id,ps.team_in_selected_period,ps.tm_player_name,ps.tm_birth_date,ps.tm_birth_year,pm.age source_age,ps.updated_at observed_at
          FROM player_seasons ps LEFT JOIN player_metrics pm ON pm.player_season_id=ps.id
          ORDER BY ps.player_id,ps.calendar DESC,ps.minutes_played DESC NULLS LAST
        ), youth AS (
          SELECT DISTINCT ON (provider_player_id) * FROM youth_player_rankings
          ORDER BY provider_player_id,season DESC,minutes_played DESC NULLS LAST
        ) SELECT p.id::text provider_id,'wyscout' provider,p.name,p.birth_date::text birth_date,
          EXTRACT(YEAR FROM p.birth_date)::int birth_year,l.team_in_selected_period club_name,
          l.tm_player_name alias,l.tm_birth_date alternate_birth_date,l.tm_birth_year alternate_birth_year,l.source_age,l.observed_at
          FROM players p LEFT JOIN latest l ON l.player_id=p.id
          UNION ALL SELECT provider_player_id,'eyeball',display_name,birth_date,birth_year,club_name,
          NULL,NULL,NULL,age,updated_at FROM youth""")).mappings()]
        matches=match_players(rows,targets)
        # Preserve explicitly reviewed decisions. Rebuild machine decisions transactionally.
        session.execute(text('DELETE FROM fm_player_matches WHERE import_id=:id AND reviewed_by IS NULL'),{'id':import_id})
        locked=list(session.execute(text('SELECT fm_uid,provider,provider_id FROM fm_player_matches WHERE import_id=:id'),{'id':import_id}).mappings())
        for m in matches:
            if any(x['provider']==m['provider'] and (x['fm_uid']==m['fm_uid'] or x['provider_id']==m['provider_id']) for x in locked): continue
            session.execute(text('''INSERT INTO fm_player_matches(import_id,fm_uid,provider,provider_id,status,method,evidence)
               VALUES (:import_id,:fm_uid,:provider,:provider_id,:status,:method,CAST(:evidence AS JSONB)) ON CONFLICT DO NOTHING'''),
               dict(m,import_id=import_id,evidence=json.dumps(m['evidence'])))
        report=dict(import_id=import_id,players=len(rows),matches=dict(Counter(m['status'] for m in matches)),
                    providers=dict(Counter((m['provider']+':'+m['status']) for m in matches)),
                    note='FM is a dated simulation, not current real-world performance. Unknown build requires validation.',
                    review=[m for m in matches if m['status']=='review'])
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print(json.dumps({k:v for k,v in report.items() if k!='review'}))

if __name__=='__main__': main()
