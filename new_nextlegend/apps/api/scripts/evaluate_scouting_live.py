"""Explicit opt-in live-model evaluation; read-only database evidence, local output."""
import json
from pathlib import Path
from db import SessionLocal
from scouting_intelligence import run_intelligence
from scouting_queries import query_players
from sqlalchemy import text

cases=[
 ('fm_budget','En 2026/2027, donne les 5 joueurs de moins de 23 ans ayant au moins 450 minutes, PA FM au moins 140 avec correspondance vérifiée, valeur Transfermarkt connue au plus 20 millions euros. Classe par score décroissant, affiche âge, minutes, score, PA, valeur et date de fraîcheur. Ne relâche aucun filtre. Réponse en français, 220 mots maximum.'),
 ('groups','Saison 2026/2027 : compare les compétitions sur les joueurs U23 avec au moins 450 minutes. Ne retiens que les championnats ayant au moins 20 observations éligibles. Donne les 5 meilleurs par score moyen, le nombre de joueurs distincts et la moyenne xG/90 avec son dénominateur non nul. Ne confonds pas observations et joueurs distincts. 200 mots maximum.'),
 ('empty','Trouve en Wyscout 2026/2027 les joueurs de 15 à 16 ans avec au moins 10000 minutes, PA FM supérieur ou égal à 190 et xG/90 supérieur à 2. Ne change aucun critère même sans résultat. 120 mots maximum.'),
 ('youth','Sur Eyeball 2026/2027, quels sont les 3 attaquants ATT de 15 à 18 ans ayant au moins 180 minutes et un score supérieur ou égal à 50 ? Classe par score décroissant, donne les minutes, âge et liens. Ne remplace pas Eyeball par Wyscout. 150 mots maximum.')]
out=[]
for name,prompt in cases:
 with SessionLocal() as s:
  result=run_intelligence(s,prompt,season='2026/2027',language='fr')
  queries=[e for e in result.get('evidence',[]) if e['tool']=='query_players' and e.get('sql')]
  record={'case':name,'prompt':prompt,'response':result,'executed_queries':len(queries)}
  for e in queries:
   r=query_players(s,e['parameters']);e['verified_rows']=r['rows']
  # Independent SQL oracle for the grouped cohort (no FM joins or compiler).
  if name=='groups':
   record['oracle']=[dict(r) for r in s.execute(text('''SELECT c.name competition,COUNT(*) observations,COUNT(DISTINCT ps.player_id) unique_players,AVG(ps.global_score_adjusted) score,AVG(pm.xg_per_90) xg_per_90,COUNT(pm.xg_per_90) xg_per_90_known FROM player_seasons ps JOIN player_metrics pm ON pm.player_season_id=ps.id JOIN competitions c ON c.id=ps.competition_id WHERE ps.calendar='2026/2027' AND pm.age BETWEEN 10 AND 22 AND ps.minutes_played>=450 GROUP BY c.name HAVING COUNT(*)>=20 ORDER BY score DESC NULLS LAST,c.name LIMIT 5''')).mappings()]
  out.append(record)
  Path('/tmp/scouting-live-evaluation.json').write_text(json.dumps(out,default=str,ensure_ascii=False,indent=2))
  print(name,'queries=',len(queries),'tokens=',result.get('usage',{}).get('total_tokens'),flush=True)
  print(result['report'],flush=True)
