"""Typed analytical plans compiled to bounded, read-only PostgreSQL queries."""
import re
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import text

METRICS = ('goals','assists','xg','xa','goals_per_90','non_penalty_goals_per_90',
 'xg_per_90','xa_per_90','shots_per_90','key_passes_per_90','dribbles_per_90',
 'successful_dribbles_percent','progressive_passes_per_90','accurate_passes_percent',
 'def_duels_won_percent','aerial_duels_won_percent','interceptions_per_90',
 'successful_def_actions_per_90','save_percent','goals_prevented_per_90','team_strength_z')

class Threshold(BaseModel):
    model_config=ConfigDict(extra='forbid',allow_inf_nan=False)
    metric: str = Field(description='Exact metric key returned by query_catalog')
    operator: Literal['gte','lte','gt','lt','eq'] = 'gte'
    value: float

class PlayerQuery(BaseModel):
    model_config=ConfigDict(extra='forbid',allow_inf_nan=False)
    source: Literal['wyscout','eyeball'] = 'wyscout'
    season: str = Field(default='2026/2027',pattern=r'^\d{4}(/\d{4})?$')
    min_age: int = Field(default=10,ge=10,le=65)
    max_age: int = Field(default=65,ge=10,le=65)
    min_minutes: int = Field(default=0,ge=0,le=15000)
    competitions: list[str] = Field(default_factory=list,max_length=20,description='Exact catalog competition names; empty means all')
    positions: list[str] = Field(default_factory=list,max_length=20,description='Exact provider positions from catalog; empty means all')
    club: str = Field(default='',max_length=120)
    filters: list[Threshold] = Field(default_factory=list,max_length=12)
    metrics: list[str] = Field(default_factory=list,max_length=12)
    order_by: str = Field(default='score',description='score, minutes, age, fm_pa, market_value_eur or a catalog metric')
    direction: Literal['asc','desc'] = 'desc'
    group_by: Literal['competition','club','position'] | None = None
    min_group_size: int = Field(default=1,ge=1,le=10000)
    limit: int = Field(default=10,ge=1,le=30)

    @model_validator(mode='after')
    def ages(self):
        if self.min_age>self.max_age: raise ValueError('min_age must not exceed max_age')
        return self

def query_catalog(session):
    def values(sql): return [r[0] for r in session.execute(text(sql))]
    youth=values("SELECT DISTINCT k FROM youth_player_rankings, jsonb_object_keys(metrics) k ORDER BY k")
    youth=[k for k in youth if re.fullmatch(r'[a-zA-Z][a-zA-Z0-9_]{0,79}',k)]
    return {
      'wyscout': {'metrics':list(METRICS),'positions':values('SELECT DISTINCT position FROM player_seasons WHERE position IS NOT NULL ORDER BY 1'),
        'competitions':values('SELECT name FROM competitions ORDER BY name')},
      'eyeball': {'metrics':youth,'positions':values('SELECT DISTINCT primary_position FROM youth_player_rankings WHERE primary_position IS NOT NULL ORDER BY 1'),
        'competitions':values('SELECT DISTINCT championship FROM youth_player_rankings WHERE championship IS NOT NULL ORDER BY 1')},
      'common_fields':['score','age','minutes','fm_pa','fm_ca'],
      'wyscout_extra':['market_value_eur'],
      'rules':'Call query_players for combined numeric filters, FM matches, sorting or grouped averages. One observation per player/season/competition/club; grouped count counts observations, not unique people (unique_players is separate). FM is dated simulation, not measured performance. Unknowns are NULL and excluded from comparisons. Market values may be stale; display profile timestamp. Positions are provider codes, use catalog. Percent metrics use 0–100; scores 0–100; FM 0–200. Group order_by sorts the mean. No last-match or injury data.'}

def compile_query(args, youth_metrics=()):
    a=PlayerQuery.model_validate(args); pro=a.source=='wyscout'
    params={'season':a.season,'min_age':a.min_age,'max_age':a.max_age,'min_minutes':a.min_minutes,'limit':a.limit,'min_group_size':a.min_group_size}
    fields={'score':'ps.global_score_adjusted' if pro else 'ps.score','age':'pm.age' if pro else 'ps.age','minutes':'ps.minutes_played','fm_pa':'f.pa','fm_ca':'f.ca'}
    if pro:
        fields.update({k:'pm.'+k for k in METRICS});fields['market_value_eur']='ps.tm_market_value_eur'
    else:
        for i,k in enumerate(youth_metrics):
            if not re.fullmatch(r'[a-zA-Z][a-zA-Z0-9_]{0,79}',k): continue
            params['key_'+str(i)]=k
            val=f"(ps.metrics ->> :key_{i})"
            fields[k]=f"CASE WHEN {val} ~ '^-?[0-9]+([.][0-9]+)?$' THEN ({val})::double precision END"
    selected=list(dict.fromkeys(['score','age','minutes','fm_pa','fm_ca']+(['market_value_eur'] if pro else [])+a.metrics+[a.order_by]+[t.metric for t in a.filters]))
    unknown=set(selected)-fields.keys()
    if unknown: raise ValueError('Unsupported metrics: '+', '.join(sorted(unknown))+'. Consult query_catalog; do not silently drop constraints.')
    name='p.name' if pro else 'ps.display_name'; pid='p.id' if pro else 'ps.provider_player_id'
    club='ps.team_in_selected_period' if pro else 'ps.club_name'
    competition='c.name' if pro else 'ps.championship'
    position='ps.position' if pro else 'ps.primary_position'
    joins='JOIN players p ON p.id=ps.player_id JOIN competitions c ON c.id=ps.competition_id LEFT JOIN player_metrics pm ON pm.player_season_id=ps.id' if pro else ''
    table='player_seasons' if pro else 'youth_player_rankings'
    where=['ps.calendar=:season',f'{fields["age"]} BETWEEN :min_age AND :max_age','ps.minutes_played>=:min_minutes']
    for attr,expr in [('competitions',competition),('positions',position)]:
        values=getattr(a,attr)
        if values:
            keys=[]
            for i,v in enumerate(values): params[attr+str(i)]=v; keys.append(':'+attr+str(i))
            where.append(expr+' IN ('+','.join(keys)+')')
    if a.club:
        params['club']='%'+a.club.replace('\\','\\\\').replace('%','\\%').replace('_','\\_')+'%'
        where.append(club+' ILIKE :club')
    for i,t in enumerate(a.filters):
        params['v'+str(i)]=t.value
        op={'gte':'>=','lte':'<=','gt':'>','lt':'<','eq':'='}[t.operator]
        where.append(f'({fields[t.metric]}) {op} :v{i}')
    # The only interpolated identifiers are internal constants or catalog-validated metric keys.
    cols=', '.join(f'{fields[k]} AS "{k}"' for k in selected)
    url="'/report?player_id='||p.id||'&player_season_id='||ps.id" if pro else "'/youth-scouting/reports?youth_id='||ps.provider_player_id"
    sql=f'''WITH cohort AS (SELECT ps.id observation_id,{pid} player_id,{name} name,
      {club} club,{competition} competition,{position} position,ps.calendar season,
      {cols},ps.updated_at data_updated_at,i.game_date fm_game_date,i.known_build fm_known_build,
      {"ps.tm_profile_updated_at" if pro else "NULL::timestamptz"} market_profile_updated_at,
      {url} report_url
      FROM {table} ps {joins}
      LEFT JOIN fm_imports i ON i.id=(SELECT id FROM fm_imports ORDER BY game_date DESC,id DESC LIMIT 1)
      LEFT JOIN fm_player_matches m ON m.import_id=i.id AND m.provider='{a.source}' AND m.provider_id={pid}::text AND m.status='accepted'
      LEFT JOIN fm_players f ON f.import_id=m.import_id AND f.uid=m.fm_uid
      WHERE {' AND '.join(where)}) '''
    if a.group_by:
        averages=', '.join(f'AVG("{k}") AS "{k}", COUNT("{k}") AS "{k}_known"' for k in selected)
        sql+=f'SELECT {a.group_by},COUNT(*) observations,COUNT(DISTINCT player_id) unique_players,{averages} FROM cohort GROUP BY {a.group_by} HAVING COUNT(*)>=:min_group_size ORDER BY "{a.order_by}" {a.direction} NULLS LAST,{a.group_by} LIMIT :limit'
    else:
        sql+=f'SELECT * FROM cohort ORDER BY "{a.order_by}" {a.direction} NULLS LAST,observation_id LIMIT :limit'
    return sql,params

def query_players(session,args):
    a=PlayerQuery.model_validate(args)
    keys=query_catalog(session)['eyeball']['metrics'] if a.source=='eyeball' else ()
    sql,params=compile_query(args,keys)
    # Separate transaction: chat persistence can write, this connection cannot.
    with session.get_bind().connect() as c, c.begin():
        c.execute(text('SET TRANSACTION READ ONLY'))
        c.execute(text("SET LOCAL statement_timeout='8s'"))
        result=[dict(r) for r in c.execute(text(sql),params).mappings()]
    return {'rows':result,'returned_rows':len(result),'limit':a.limit,'sql':sql,'parameters':params,
      'unit':'player / season / competition / club observation',
      'notes':'Returned rows are bounded, not a total count. Group means exclude NULL per metric; *_known gives denominator. FM game date is simulation date. Market profile timestamp does not guarantee a fresh valuation. No unknown replaced by zero.'}
