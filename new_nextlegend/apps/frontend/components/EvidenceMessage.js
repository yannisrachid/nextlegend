import React from 'react';

function allowed(url) { return /^https?:\/\//i.test(url) || /^\/(report\?|youth-scouting\/reports\?)/.test(url); }
function inline(text) {
  return String(text).split(/(\[[^\]]+\]\([^)]+\)|\*\*[^*]+\*\*)/g).map((part,i) => {
    const match=part.match(/^\[([^\]]+)\]\(([^)]+)\)$/);
    if (match && allowed(match[2])) return <a key={i} href={match[2]} target="_blank" rel="noopener noreferrer" className="break-words font-medium text-primary underline decoration-primary/30 underline-offset-4">{match[1]}</a>;
    if (part.startsWith('**/') && part.endsWith('**') && allowed(part.slice(2,-2))) return <a key={i} href={part.slice(2,-2)} target="_blank" rel="noopener noreferrer" className="text-primary underline">Open player report ↗</a>;
    if (part.startsWith('**') && part.endsWith('**')) return <strong key={i} className="font-semibold text-foreground">{part.slice(2,-2)}</strong>;
    return part;
  });
}
function blocks(content) {
  const lines=String(content || '').split('\n'), result=[];
  const cells=line=>line.trim().replace(/^\||\|$/g,'').split('|').map(c=>c.trim());
  for(let i=0;i<lines.length;i++) {
    const line=lines[i]; if(!line.trim()) continue;
    if(line.includes('|') && /^\s*\|?\s*:?-{3,}/.test(lines[i+1] || '')) {
      const head=cells(line), rows=[];i+=2;
      while(i<lines.length && lines[i].includes('|')) {rows.push(cells(lines[i]));i++;} i--;
      result.push(<div key={i} className="max-w-full overflow-x-auto rounded-lg border border-border" tabIndex={0} role="region" aria-label="Scouting results table"><table className="w-full text-left text-xs"><thead><tr>{head.map((c,j)=><th key={j} className="whitespace-nowrap px-3 py-2">{inline(c)}</th>)}</tr></thead><tbody>{rows.map((r,k)=><tr key={k} className="border-t border-border">{head.map((_,j)=><td key={j} className="min-w-[90px] px-3 py-2 align-top">{inline(r[j] || '—')}</td>)}</tr>)}</tbody></table></div>);
    } else if(/^#{1,4} /.test(line)) result.push(<h3 key={i} className="pt-3 text-base font-semibold text-foreground">{inline(line.replace(/^#+ /,''))}</h3>);
    else result.push(<p key={i} className="whitespace-pre-wrap break-words">{inline(line)}</p>);
  }
  return result;
}
export default function EvidenceMessage({ content, payload }) {
  return <div className="min-w-0 space-y-3 text-sm leading-7 text-foreground">
    {blocks(content)}
    {!!payload?.player_links?.length && <div className="flex flex-wrap gap-2 border-t border-border pt-3">{payload.player_links.map(p=><a key={p.url} href={allowed(p.url)?p.url:'#'} target="_blank" rel="noopener noreferrer" className="rounded-md border border-border px-2.5 py-1 text-xs font-medium text-primary hover:bg-accent">{p.name} ↗</a>)}</div>}
    {!!payload?.sources?.length && <div className="border-t border-border pt-3"><p className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Sources</p><ul className="mt-2 space-y-1">{payload.sources.filter(s=>allowed(s.url)).map((s,i)=><li key={`${s.url}-${i}`}><a href={s.url} target="_blank" rel="noopener noreferrer" className="text-xs text-primary underline">{s.title || s.url}</a></li>)}</ul></div>}
    {!!payload?.evidence?.length && <details className="rounded-lg border border-border bg-secondary/40 px-3 py-2 text-xs text-muted-foreground"><summary className="cursor-pointer font-medium">View evidence · {payload.evidence.length} research steps</summary><ol className="mt-3 space-y-3">{payload.evidence.map((e,i)=><li key={i} className="border-t border-border pt-2"><p className="font-semibold text-foreground">{e.tool.replaceAll('_',' ')}{e.status==='invalid' ? ' · invalid request' : e.status==='failed' ? ' · unavailable' : ''}</p><p>{new Date(e.checked_at).toLocaleString('en-GB')}{e.row_count != null ? ` · ${e.row_count} rows returned` : ''}</p><details className="mt-1"><summary className="cursor-pointer text-primary">{e.sql ? 'SQL & applied filters' : 'Applied filters'}</summary><pre className="mt-2 max-h-72 overflow-auto whitespace-pre-wrap break-all rounded-md bg-card p-3 text-[11px] leading-5">{JSON.stringify(e.parameters,null,2)}{e.sql ? `\n\n${e.sql}\n\nBindings\n${JSON.stringify(e.bindings,null,2)}` : ''}</pre>{e.unit && <p className="mt-1">Unit: {e.unit}</p>}</details></li>)}</ol></details>}
  </div>;
}
