import { useEffect, useState } from 'react';
import Link from 'next/link';
import { fetchJson } from '@/lib/api';

const number = (n) => n == null ? '—' : Number(n).toLocaleString('en-GB', { maximumFractionDigits: 1 });
export default function ScoutingBriefing({ season, onAsk }) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(false);
  const [tab, setTab] = useState('performers');
  const [reload, setReload] = useState(0);
  useEffect(() => {
    let active = true;
    setData(null); setError(false);
    fetchJson('/ai/briefing', { season: season || 'all' })
      .then(value => { if (active) setData(value); })
      .catch(() => { if (active) setError(true); });
    return () => { active = false; };
  }, [season, reload]);
  return <section className="overflow-hidden rounded-xl border border-border bg-card" aria-label="Scouting intelligence">
    <div className="border-b border-border p-4">
      <div className="flex items-center justify-between gap-2">
        <p className="nl-kicker">On your radar</p>
        <button className="text-xs text-primary hover:text-foreground" onClick={() => setReload(r => r + 1)} aria-label="Refresh scouting shortlist">Refresh ↻</button>
      </div>
      <h2 className="mt-1 text-base font-semibold text-foreground">Players worth a closer look</h2>
      <p className="mt-2 text-xs leading-5 text-muted-foreground">Season evidence to guide your next scouting conversation.</p>
    </div>
    <div className="flex flex-wrap gap-1 border-b border-border p-2" role="group" aria-label="Shortlist category">
      {[['performers', 'U23'], ['youth', 'Youth'], ['momentum', 'Progress'], ['potential', 'FM × form'], ['watchlist', 'Following']].map(([key,label]) => <button key={key} aria-pressed={tab===key} onClick={() => setTab(key)} className={`rounded-md px-2.5 py-2 text-xs transition ${tab===key ? 'bg-accent text-primary' : 'text-muted-foreground hover:bg-secondary hover:text-foreground'}`}>{label}</button>)}
    </div>
    {error ? <div className="p-4 text-sm text-muted-foreground">The shortlist could not be loaded. <button className="text-primary" onClick={() => setReload(r=>r+1)}>Retry</button></div> : !data ? <div className="space-y-3 p-4" role="status" aria-label="Loading shortlist">{[0,1,2].map(i=><div key={i} className="h-16 animate-pulse rounded-md bg-secondary motion-reduce:animate-none" />)}</div> : <>
      <div className="max-h-[390px] overflow-auto divide-y divide-border">
        {!(data[tab] || []).length ? <p className="p-4 text-xs leading-5 text-muted-foreground">No verified profiles meet this selection yet. Try another season or ask the assistant to broaden the search.</p> : (data[tab] || []).map((p,i)=><article key={`${p.report_url}-${i}`} className="px-4 py-3 hover:bg-secondary">
          <div className="flex items-start justify-between gap-3">
            <div className="min-w-0"><Link href={p.report_url || '#'} target="_blank" rel="noopener noreferrer" className="text-sm font-semibold text-foreground hover:text-primary">{p.name}</Link><p className="mt-1 break-words text-xs text-muted-foreground">{p.club || 'Club unavailable'}{p.position ? ` · ${p.position}` : ''}</p></div>
            <div className="shrink-0 text-right"><span className="text-base font-semibold tabular-nums text-primary">{number(p.score)}</span><p className="text-[10px] text-slate-500">Score</p></div>
          </div>
          <p className="mt-2 text-[11px] text-muted-foreground">{number(p.minutes_played)} min · {p.season || data.season}{p.pa != null ? ` · FM PA ${p.pa}/200` : ''}</p>
          {p.delta != null && <p className="mt-1 text-xs text-primary">+{number(p.delta)} points · {p.previous_date} → {p.snapshot_date}</p>}
          {p.data_updated_at && <p className="mt-1 text-[10px] text-slate-500">Updated {new Date(p.data_updated_at).toLocaleDateString('en-GB')}</p>}
          <button onClick={() => onAsk(`Assess ${p.name} (${p.club || ''}), ${p.season || data.season}. Verify performance, comparable score history and FM potential if matched. Explain uncertainty and the next scouting action.`)} className="mt-2 text-xs font-medium text-primary hover:text-foreground">Investigate profile ↗</button>
        </article>)}
      </div>
      <div className="border-t border-border px-4 py-3 text-[10px] leading-4 text-slate-500">
        {tab==='performers' ? 'U23 · 450+ minutes · ordered by season score.' : tab==='youth' ? 'U21 · 180+ minutes · ordered by youth season score.' : tab==='momentum' ? 'Comparable snapshots only · same scoring model and player-season. Cumulative score change, not recent match form.' : tab==='potential' ? 'Verified FM identity links · PA 140+ · 450+ real minutes. Simulation is not a prediction.' : 'Agency portfolio and prospect list. Scores describe the displayed season.'}
        {data.fm?.[0] && <p className="mt-2">FM save: {data.fm[0].game_date}. {data.fm[0].known_build ? '' : 'Reader build not fully validated.'}</p>}
      </div>
    </>}
  </section>;
}
