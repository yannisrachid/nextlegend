import { useEffect, useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/router';
import { useAuth } from '@/lib/auth';
import { fetchJson } from '@/lib/api';
import { DEFAULT_SCOUTING_SEASON, withDefaultSeason } from '@/lib/scoutingFilters';

const number = value => value == null ? '—' : Number(value).toLocaleString('en-GB',{maximumFractionDigits:1});
const date = value => value ? new Date(value).toLocaleDateString('en-GB',{day:'numeric',month:'short',year:'numeric'}) : 'Date unavailable';
const suggestions = [
  ['Find an opportunity','Find 5 U23 players in the selected season with at least 450 minutes, verified FM PA of 140 or more and known Transfermarkt value below €20M. Rank by NextLegend score. Explain freshness and uncertainty.'],
  ['Track development','Which players are improving across comparable score snapshots in the selected season? Explain the dates and sample sizes.'],
  ['Review our portfolio','Review the agency players and prospects we follow. Show the available performance evidence and which profiles need a closer look.'],
];
function Arrow({className='h-4 w-4'}) {return <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" aria-hidden="true"><path d="M5 12h14m-5-5 5 5-5 5"/></svg>;}
function PlayerRow({player,index,onAsk}) {
  return <li className="group flex items-center gap-3 border-t border-border py-4 first:border-0">
    <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-secondary text-xs font-semibold text-primary">{String(index+1).padStart(2,'0')}</span>
    <div className="min-w-0 flex-1"><Link href={player.report_url} className="block truncate text-sm font-semibold text-foreground hover:text-primary">{player.name}</Link><p className="mt-1 truncate text-xs text-muted-foreground">{player.club || 'Club unavailable'} · {number(player.minutes_played)} min · {player.season || 'Season unavailable'}</p></div>
    <div className="text-right"><p className="text-lg font-semibold tabular-nums text-primary">{number(player.score)}</p><p className="text-[10px] text-muted-foreground">{player.delta != null ? `+${number(player.delta)} points` : 'Season score'}</p></div>
    <button className="nl-icon-button border-transparent" onClick={()=>onAsk(`Investigate ${player.name} (${player.club || ''}) in ${player.season || 'the selected season'}. Check performance, verified FM potential and comparable score history. Give the next scouting action.`)} aria-label={`Investigate ${player.name}`}><Arrow/></button>
  </li>;
}
function SignalPanel({eyebrow,title,description,players,loading,onAsk,empty,href}) {
  return <section className="min-w-0 rounded-2xl border border-border bg-card p-5 md:p-6">
    <div className="flex items-center justify-between"><p className="nl-kicker">{eyebrow}</p><Link href={href} className="nl-icon-button border-transparent" aria-label={`Open ${title}`}><Arrow/></Link></div>
    <h2 className="mt-1 text-xl font-semibold tracking-tight">{title}</h2><p className="mt-2 min-h-10 text-xs leading-5 text-muted-foreground">{description}</p>
    {loading ? <div className="mt-4 space-y-4" role="status" aria-label={`Loading ${title}`}>{[1,2,3].map(i=><div key={i} className="nl-skeleton h-14"/>)}</div> : players?.length ? <ul className="mt-2">{players.slice(0,3).map((p,i)=><PlayerRow key={`${p.report_url}-${i}`} player={p} index={i} onAsk={onAsk}/>)}</ul> : <div className="mt-4 rounded-xl bg-secondary/60 p-5 text-sm leading-6 text-muted-foreground">{empty}</div>}
  </section>;
}
export default function HomePage() {
  const {me}=useAuth();const router=useRouter();
  const [season,setSeason]=useState(DEFAULT_SCOUTING_SEASON);const [seasons,setSeasons]=useState([DEFAULT_SCOUTING_SEASON]);const [data,setData]=useState(null);
  const [error,setError]=useState('');const [revision,setRevision]=useState(0);const [prompt,setPrompt]=useState('');
  useEffect(()=>{let active=true;fetchJson('/meta/seasons').then(r=>{if(active)setSeasons(withDefaultSeason(r || []));}).catch(()=>{});return()=>{active=false;};},[]);
  useEffect(()=>{let active=true;setData(null);setError('');fetchJson('/ai/briefing',{season}).then(r=>{if(active)setData(r);}).catch(()=>{if(active)setError('Your briefing could not be loaded. Please try again.');});return()=>{active=false;};},[season,revision]);
  function ask(text) {
    try { sessionStorage.setItem('nl_scout_draft',JSON.stringify({prompt:text,season}));router.push('/ai'); }
    catch {setError('Unable to open your draft. Please open AI Assistant from the navigation.');}
  }
  const loading=!data&&!error;const lead=data?.performers?.[0];
  const name=me?.full_name || me?.display_name || me?.username || 'there';
  return <main className="nl-page px-4 py-6 md:px-7 lg:py-8">
    <div className="mx-auto max-w-[1440px] space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-3"><p className="text-sm text-muted-foreground">Welcome back, <span className="font-medium text-foreground">{name.split(' ')[0]}</span>.</p><div className="flex items-center gap-2"><label htmlFor="home-season" className="text-xs text-muted-foreground">Season</label><select id="home-season" value={season} onChange={e=>setSeason(e.target.value)} className="nl-field w-auto">{seasons.map(s=><option key={s}>{s}</option>)}</select><button className="nl-button-secondary" onClick={()=>setRevision(v=>v+1)} disabled={loading}>Refresh</button></div></div>
      <section className="relative overflow-hidden rounded-3xl border border-border bg-card">
        <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(ellipse_at_top_right,rgba(37,99,235,0.09),transparent_65%)]" aria-hidden="true"/>
        <div className="relative grid gap-8 p-6 md:p-10 xl:grid-cols-[minmax(0,1fr)_320px] xl:items-center">
          <div className="min-w-0"><p className="nl-kicker">The HD Sports workspace</p><h1 className="mt-5 max-w-2xl text-4xl font-semibold leading-[1.12] tracking-[-0.045em] md:text-5xl">See the talent.<br/><span className="text-primary">Make the next move.</span></h1><p className="mt-5 max-w-xl text-base leading-7 text-muted-foreground">From the first signal to a confident shortlist. Bring player performance, scouting insight and your agency’s priorities into one conversation.</p>
            <form onSubmit={e=>{e.preventDefault();if(prompt.trim())ask(prompt.trim());}} className="mt-7 rounded-2xl border border-input bg-card p-3 shadow-sm"><label htmlFor="home-question" className="sr-only">Ask your scouting assistant</label><textarea id="home-question" value={prompt} maxLength={16000} onChange={e=>setPrompt(e.target.value)} rows={2} placeholder="Who should we be watching this season?" className="w-full resize-none rounded-lg bg-transparent px-2 py-2 text-sm leading-6 outline-none placeholder:text-muted-foreground focus-visible:ring-2 focus-visible:ring-primary/20"/><div className="mt-2 flex flex-wrap items-center justify-between gap-2"><span className="px-2 text-xs text-muted-foreground">Your question. Your data. Clear evidence.</span><button type="submit" disabled={!prompt.trim()} className="nl-button-primary">Explore with AI <Arrow/></button></div></form>
            <div className="mt-3 flex flex-wrap gap-2">{suggestions.map(([label,text])=><button key={label} onClick={()=>ask(text)} className="rounded-full border border-border bg-card px-3 py-2 text-xs font-medium text-muted-foreground transition hover:border-primary/30 hover:text-primary">{label} ↗</button>)}</div>
          </div>
          <aside className="rounded-2xl border border-primary/15 bg-secondary/60 p-6"><div className="flex items-center gap-2"><span className="h-2 w-2 rounded-full bg-primary"/><p className="nl-kicker">Performance spotlight</p></div>{loading ? <div className="mt-6 space-y-4"><div className="nl-skeleton h-8"/><div className="nl-skeleton h-24"/></div> : lead ? <><p className="mt-6 text-xs uppercase tracking-wider text-muted-foreground">U23 · 450+ minutes</p><Link href={lead.report_url} className="mt-2 block text-2xl font-semibold tracking-tight hover:text-primary">{lead.name}</Link><p className="mt-2 text-sm text-muted-foreground">{lead.club} · {lead.position}</p><div className="mt-7 flex items-end gap-3"><span className="text-6xl font-semibold tracking-[-0.06em] text-primary">{number(lead.score)}</span><span className="pb-2 text-xs leading-5 text-muted-foreground">NextLegend<br/>season score</span></div><p className="mt-5 border-t border-primary/10 pt-4 text-xs leading-5 text-muted-foreground">{number(lead.minutes_played)} minutes · {season}<br/>Data updated {date(lead.data_updated_at)}</p><button onClick={()=>ask(`Assess ${lead.name} in ${season}: performance evidence, sample size, potential and next scouting steps.`)} className="nl-button-secondary mt-5 w-full">Investigate profile <Arrow/></button></> : <p className="mt-6 text-sm leading-6 text-muted-foreground">{error ? 'Briefing unavailable. Retry below.' : 'No player meets this selection for this season.'}</p>}</aside>
        </div>
      </section>
      {error && <div role="alert" className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-destructive/20 bg-card p-4 text-sm text-destructive">{error}<button className="nl-button-secondary" onClick={()=>setRevision(v=>v+1)}>Retry</button></div>}
      {!error && <div className="grid gap-4 lg:grid-cols-3"><SignalPanel eyebrow="Discover" title="Emerging performers" description="U21 profiles with at least 180 minutes, ranked by youth season score." players={data?.youth} loading={loading} onAsk={ask} href="/youth-scouting/ranking" empty="No youth profiles meet this season’s selection yet."/><SignalPanel eyebrow="Develop" title="Moving forward" description="Score changes across comparable snapshots. Season progression, not last-match form." players={data?.momentum} loading={loading} onAsk={ask} href="/ai" empty="Not enough comparable snapshots yet. Progression will appear as the season develops."/><SignalPanel eyebrow="Follow" title="Your agency radar" description="Agency players and prospects. Each profile shows its latest available season." players={data?.watchlist} loading={loading} onAsk={ask} href="/hd-players" empty="Your agency watchlist is empty. Add players from their reports to start following them."/></div>}
      <section className="rounded-2xl border border-border bg-card p-5 md:p-6"><div className="flex flex-wrap items-end justify-between gap-3"><div><p className="nl-kicker">Keep things moving</p><h2 className="mt-2 text-xl font-semibold">One workspace. Every next step.</h2></div><Link href="/hq" className="text-sm font-medium text-primary hover:underline">Open team agenda ↗</Link></div><div className="mt-5 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">{[['/ranking','Find your next player','Build a focused scouting cohort.'],['/crm','Grow your network','Keep your club relationships close.'],['/hq','Plan with your team','Agenda, tasks and ownership.'],['/prospect','Revisit your shortlist','Turn promising signals into action.']].map(([href,title,desc])=><Link key={href} href={href} className="group rounded-xl border border-border p-4 transition hover:border-primary/30 hover:bg-secondary"><div className="flex items-center justify-between gap-2"><h3 className="text-sm font-semibold">{title}</h3><Arrow className="h-4 w-4 shrink-0 text-primary transition group-hover:translate-x-1"/></div><p className="mt-2 text-xs leading-5 text-muted-foreground">{desc}</p></Link>)}</div></section>
      <p className="text-xs leading-5 text-muted-foreground">{data ? `Briefing generated ${date(data.generated_at)}. Source updates may be older.` : 'Evidence-led scouting. Built for HD Sports.'} {data?.fm?.[0] ? `FM context is simulation from ${date(data.fm[0].game_date)}, not a prediction.` : ''}</p>
    </div>
  </main>;
}
