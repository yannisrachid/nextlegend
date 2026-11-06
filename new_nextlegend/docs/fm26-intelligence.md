# FM26 and scouting intelligence

Implemented on `feature/fm26-scouting-intelligence`. DEV only; no VPS changes.

## Data and provenance

The copied save in `/Users/yannis/fm26-data/raw/exports_originaux/` is unchanged.
The complete extraction is in `processed/players.json`, `players.csv`, typed
`players.parquet` and `fm26.sqlite`. `players.jsonl` is the bounded-memory import input. It contains 64,985 players and 218 flat
columns, including CA/PA, attributes and the original nested record. This is a
saved-game population, not the complete original game database.

Game date: 2026-01-30. Build: 26.0.2+2228080. Reader: fmsave 0.9.0.
**The build is not fully supported by the reader.** Preserve `known_build=false`;
do not present the extracted ratings as independently validated. FM describes a
simulation, not current observed performance, real personality, or a probability
of future success. Raw financial units are not assumed to be euros or weekly pay.
Commercial reuse rights have not been established by buying the game.

Additive PostgreSQL schema lives in `apps/api/fm26.py` and is applied by the importer:

- `fm_imports`: save SHA-256, game date/build, reader, extraction date and coverage.
- `fm_players`: immutable `(import_id, uid)` records, database `unique_id`, identity,
  CA/PA and lossless original JSONB. No changes to Wyscout/Eyeball scoring tables.
- `fm_player_matches`: provider identity, accepted/review/rejected, method and evidence.
  Accepted links are unique in both directions within an import/provider.

Provider `wyscout` stores the **internal NextLegend `players.id`**, not the external
Wyscout number; `eyeball` stores `provider_player_id`. FM `uid` and `unique_id` are
separate namespaces. Never join equal numeric IDs across providers.

Names are accent/case-normalized. Full name (or first initial and exact surname) with DOB may be accepted if unique.
An existing Transfermarkt alias/DOB is used only when its name corroborates the
Wyscout name (including an unambiguous initial + surname). An exact full name + explicit birth year + exact club can also be accepted when
unique. Abbreviated names without DOB, age-derived birth-year intervals and
ambiguous identities stay in review. A club
change does not invalidate exact name + DOB. Dates disagreeing are rejected.
Age inferred from an import timestamp is approximate and never sufficient for
automatic acceptance. Review proposals are not used by the assistant.

Reimport is transactional and idempotent. Existing explicitly reviewed links are
preserved. Generated reports are local, not committed:
`/Users/yannis/fm26-data/reports/nextlegend_import.json`.

## Local commands

Use the existing isolated extraction environment to regenerate from the copied save:

```bash
/Users/yannis/fm26-data/.venv/bin/python /Users/yannis/fm26-data/scripts/export_full.py
cd /Users/yannis/ylfc/new_nextlegend
./scripts/import_fm26_dev.sh
```

The DEV DB is Docker `compose-db-1` (host port 15432); containers use `db:5432`.
The wrapper streams JSONL in batches (500 records), keeping only identities in memory.
It mounts inputs read-only, uses the existing API image/environment and
writes the audit report locally. Never copy the whole `.env` into documentation.

```bash
docker compose --env-file .env -f infra/compose/docker-compose.yml up -d --no-deps api
cd apps/frontend
PATH="$HOME/.nvm/versions/node/v22.17.0/bin:$PATH" npm run dev
```

Open http://localhost:3000. HQ agenda/Kanban remain at `/hq`; `/ai` remains supported.
There is no new scheduled FM import and no automatic production rollout.

## Assistant architecture

`apps/api/scouting_intelligence.py` extends the existing authenticated conversation
API. Conversations remain user-scoped in PostgreSQL; ownership is derived from
the authenticated session, never the supplied `user_id`.

Default model: `gpt-5.4`, configurable through `AI_SCOUT_MODEL`. Uses the existing
`OPENAI_API_KEY`, the Responses API, bounded function calls and low reasoning effort.
This is database-grounded inference, **not fine-tuning**. Only selected tool outputs
are sent to OpenAI; raw saves and whole tables are not uploaded. `store=False` is
set for Responses (this is not a claim of zero provider-side retention).

Tools: player cohort search, real-world dossier with accepted FM match, full FM
record lookup gated by a previously discovered FM UID, comparable
score momentum, agency/prospect watchlist, data catalogue, and public player news.
They cover Wyscout, Eyeball, metrics, score snapshots, Opta club strength,
Transfermarkt fields, human scout observations and FM attributes. No unrestricted
SQL, no writes, no credentials/session tables, medical documents, contact details,
or other users' conversations. The old standalone scout/report endpoints remain.

Web research uses a separate request containing only a verified public identity.
It must return dated claims and clickable citations. It never receives CRM notes
as search queries. Web content and scout text are untrusted evidence, not instructions.

Scores are cumulative season signals. Momentum compares the same player-season
(club and competition), scoring hash and distinct snapshot dates; it is not a
last-match form statistic. Missing evidence stays missing. FM cannot satisfy
minimum real-world minutes or current-season performance filters.

The homepage uses the existing chat and design system, with a responsive side
briefing: U23 (450+ min), youth U21 (180+ min), comparable progress, FM PA 140+
with observed minutes, and agency/prospect monitoring. These are deterministic
screening lists, not autonomous recruitment decisions. Displayed refresh dates
are database timestamps, not a promise of live match data.

Usage records preserve actual tokens including web subrequests. Cost is marked
unavailable; billing remains the source of truth. Loop budget: 7 responses,
bounded result sizes and outputs; failures return a retryable error.

## Validation

```bash
docker exec compose-api-1 python -m unittest discover -s tests -p test_fm26_intelligence.py -v
docker exec compose-api-1 python -m py_compile main.py models.py fm26.py scouting_intelligence.py scripts/import_fm26.py
cd apps/frontend
PATH="$HOME/.nvm/versions/node/v22.17.0/bin:$PATH" npm run build
```

Tests cover namespaces, conflicting DOBs, ambiguous mappings, corroborated aliases,
unique full-name/year/club matches and review-only estimated-age matches, SQL parameter binding, tool limits and unsafe citations.
Also verify authenticated chat persistence, unauthenticated 401, mobile overflow,
desktop sidebar, HQ access, source links and FM attribution before deployment.

Official API references:
- https://developers.openai.com/api/docs/guides/function-calling
- https://developers.openai.com/api/docs/guides/tools-web-search
- https://developers.openai.com/api/docs/models/gpt-5.4

## Verified DEV result — 2026-11-01

64,985 rows, one import, zero duplicate import/UID pairs. Idempotent wrapper rerun
returned the same counts: 6,271 accepted links (5,380 professional and 891 youth),
2,381 review proposals. 11 targeted tests passed; real authenticated chat persisted
across reload; live OpenAI/tool and web-citation calls verified. Next.js build passes.
Desktop and 390px mobile checked; no horizontal document overflow.

## Light UI and analytical query verification — 2026-11-06

Branch: `feature/light-blue-scouting`. No merge or PROD changes. The working tree includes the earlier FM import/assistant changes.

### Interface
White panels and blue accents replace the dark theme across scouting, youth, HQ, CRM, settings and authentication. Navigation and module tabs are more compact. The assistant prioritizes the conversation with the shortlist/history alongside it on desktop and below it on mobile. Responses support Markdown tables and expandable SQL evidence. The HQ calendar sizes weeks according to event lanes. Existing PNG/PDF export artwork remains intentionally separate from the app theme.

### Query semantics
`apps/api/scouting_queries.py`: validated plans, bound parameters, explicit position/competition filters, numeric comparisons, FM links, market values, ordering and grouped means. Wyscout metrics are allowlisted; youth JSON metric keys come from the catalog and are validated. A dedicated read-only PostgreSQL transaction enforces an 8s statement timeout. Maximum 30 returned rows. Unsupported metrics fail explicitly.

The unit is a player/season/competition/club observation. Group results include both observations and distinct players; each nullable mean has a non-null denominator. Ages are provider ages, not recomputed as of today. FM links must be accepted and from the latest saved-game import. Unknown market value/PA does not satisfy a numeric filter. Transfermarkt profile timestamps can be old and do not guarantee current valuations. No unrestricted SQL, arbitrary tables, match-by-match form or invented injury data.

### Executed checks
- Next.js production build and TypeScript passed (28 prerendered pages). No independent lint configuration; the repository's lint command aliases a build.
- 18 unit checks passed: 11 FM identity/tool checks and 7 analytical query/link checks.
- Four live French natural-language cases executed via the existing OpenAI configuration: (1) U23, 450+ minutes, PA >=140, value <=20M; (2) competition means on U23/450+ cohorts with >=20 observations and xG/90 known-value denominator; (3) intentionally impossible 15–16-year-old/10000-minute/PA190/xG90>2 cohort; (4) Eyeball ATT, age15–18, 180+ minutes, score>=50. Each used query_players and retained requested constraints.
- First case returned Valente, Carlos Forbs, Boteli, Gloukh, Goes in score order. Independent SQL matched the five observation IDs.
- Competition aggregates matched independent SQL including missing-value denominators: Eredivisie (44 observations, xG/90 n=40), Belgian Pro League (43, n=38), Swiss Super League (24, n=21), German 2. Bundesliga (21, n=17), Danish Superliga (44, n=38).
- Impossible cohort returned zero without relaxing constraints. Youth returned Elies Diallo, Andria Svanidze, Noah Adnane; independent SQL matched their three observation IDs.
- Fixed a generated `#/youth-scouting/reports` link prefix and added a regression test.
- PostgreSQL `transaction_read_only` verified as `on` for the analytical connection.

### Reproduce locally
```bash
docker start compose-db-1 compose-minio-1 compose-api-1
docker exec compose-api-1 python -m unittest discover -s tests -p 'test_fm26_intelligence.py' -v
docker exec compose-api-1 python -m unittest discover -s tests -p 'test_scouting_queries.py' -v
# Optional: makes paid API calls; results stored only inside container /tmp.
docker exec -e PYTHONPATH=/app compose-api-1 python scripts/evaluate_scouting_live.py
# From apps/frontend, with Node 22 available:
npm run build
npm run dev
```

The live evaluation script records model calls and query results; it is an exploratory evaluation, not a blanket accuracy guarantee for every natural-language question. Keep reviewing new query types against independent SQL.

### Final local UI check (2026-11-06)
Authenticated browser check covered the AI conversation (including a fifth real multi-constraint request, returned table, working report links and visible SQL/bindings), 390px mobile containment, HQ calendar and event form, Research (1,408 plotted players with median/Q3 highlighting), Youth Ranking and collapse/expand navigation. A mobile filter truncation and stale usage counter were corrected. The final build/TypeScript passed after these corrections. This is a targeted smoke test, not an exhaustive end-to-end check of every business workflow.

The interactive test conversation remains in the admin history, titled `Test de validation : en 2026/2027, les 2 meilleurs scores...`, so the user can inspect the evidence. Its numerical shortlist agrees with the SQL tests. The prompt was tightened afterwards to call the proprietary score NextLegend, not a Wyscout-issued rating. No additional paid run was needed for that wording-only correction.

## Dynamic landing and focused conversation UI — 2026-11-06
Home `/` is now a distinct authenticated landing, using the existing briefing endpoint for real U23 spotlight, youth performers, comparable progression and agency watchlist. `/ai` retains the conversation experience, personal history and SQL evidence. Home draft prompts transfer through one-time sessionStorage, preserve the chosen season and do not auto-send or create paid requests. No model/API changes for this UI iteration.

Validated: production build/TypeScript; 390px containment without horizontal page overflow; 2025/2026 selection returned the correct older player-season report and 2,486 minutes for Yamal instead of 643 for 2026/2027; Home suggestion opened a blank thread with editable prompt and correct season; history search empty state; conversation settings remain accessible. Watchlist is explicitly latest-available-season and can differ from selected scouting season.

Production compatibility: derive TM birth year from the available birth-date field. Optional normalized EUR value is read defensively from the row JSON; absent or invalid values remain unknown and cannot satisfy a budget filter. No obsolete Transfermarkt refresh columns are required by the FM importer.
