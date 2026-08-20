-- Idempotent cleanup for production agency/workbook duplicates.
-- Run only after a Postgres backup. This avoids deleting core Wyscout players by name.

BEGIN;

-- HD Players seed duplication cleanup. Keep the most useful active card:
-- linked player first, then most recently updated, then highest id.
WITH ranked_hd_players AS (
    SELECT
        id,
        ROW_NUMBER() OVER (
            PARTITION BY LOWER(BTRIM(display_name))
            ORDER BY
                CASE WHEN player_id IS NOT NULL THEN 0 ELSE 1 END,
                updated_at DESC NULLS LAST,
                id DESC
        ) AS rank
    FROM hd_players
    WHERE status <> 'archived'
),
duplicate_hd_players AS (
    SELECT id
    FROM ranked_hd_players
    WHERE rank > 1
)
DELETE FROM hd_players
WHERE id IN (SELECT id FROM duplicate_hd_players);

-- Active Mercato request duplicates. Keep the newest active request and archive older ones.
WITH ranked_requests AS (
    SELECT
        id,
        ROW_NUMBER() OVER (
            PARTITION BY club_id, LOWER(BTRIM(title)), COALESCE(assigned_agent_id, ''), season
            ORDER BY updated_at DESC NULLS LAST, id DESC
        ) AS rank
    FROM mercato_requests
    WHERE archived_at IS NULL
),
duplicate_requests AS (
    SELECT id
    FROM ranked_requests
    WHERE rank > 1
)
UPDATE mercato_requests
SET status = 'closed',
    archived_at = NOW(),
    updated_at = NOW()
WHERE id IN (SELECT id FROM duplicate_requests);

-- Duplicate needs within one request. Keep the newest one; dependent candidates cascade.
WITH ranked_needs AS (
    SELECT
        id,
        ROW_NUMBER() OVER (
            PARTITION BY
                mercato_request_id,
                COALESCE(LOWER(BTRIM(position)), ''),
                COALESCE(LOWER(BTRIM(role)), '')
            ORDER BY updated_at DESC NULLS LAST, id DESC
        ) AS rank
    FROM mercato_needs
),
duplicate_needs AS (
    SELECT id
    FROM ranked_needs
    WHERE rank > 1
)
DELETE FROM mercato_needs
WHERE id IN (SELECT id FROM duplicate_needs);

-- Candidate duplicates are normally protected by UNIQUE(mercato_need_id, player_id).
WITH ranked_candidates AS (
    SELECT
        id,
        ROW_NUMBER() OVER (
            PARTITION BY mercato_need_id, player_id
            ORDER BY match_score DESC NULLS LAST, updated_at DESC NULLS LAST, id DESC
        ) AS rank
    FROM mercato_candidates
),
duplicate_candidates AS (
    SELECT id
    FROM ranked_candidates
    WHERE rank > 1
)
DELETE FROM mercato_candidates
WHERE id IN (SELECT id FROM duplicate_candidates);

CREATE UNIQUE INDEX IF NOT EXISTS mercato_requests_active_dedupe_idx ON mercato_requests (
    club_id,
    LOWER(BTRIM(title)),
    COALESCE(assigned_agent_id, ''),
    season
) WHERE archived_at IS NULL;

CREATE UNIQUE INDEX IF NOT EXISTS mercato_needs_dedupe_idx ON mercato_needs (
    mercato_request_id,
    COALESCE(LOWER(BTRIM(position)), ''),
    COALESCE(LOWER(BTRIM(role)), '')
);

CREATE UNIQUE INDEX IF NOT EXISTS hd_players_active_player_unique_idx ON hd_players(player_id)
WHERE player_id IS NOT NULL AND status <> 'archived';

CREATE UNIQUE INDEX IF NOT EXISTS hd_players_active_name_unique_idx ON hd_players(LOWER(BTRIM(display_name)))
WHERE status <> 'archived';

COMMIT;

SELECT 'hd_players_active_name' AS check_name, COUNT(*) AS duplicate_groups
FROM (
    SELECT LOWER(BTRIM(display_name))
    FROM hd_players
    WHERE status <> 'archived'
    GROUP BY 1
    HAVING COUNT(*) > 1
) duplicates;

SELECT 'mercato_requests_active' AS check_name, COUNT(*) AS duplicate_groups
FROM (
    SELECT club_id, LOWER(BTRIM(title)), COALESCE(assigned_agent_id, ''), season
    FROM mercato_requests
    WHERE archived_at IS NULL
    GROUP BY 1, 2, 3, 4
    HAVING COUNT(*) > 1
) duplicates;

SELECT 'mercato_needs' AS check_name, COUNT(*) AS duplicate_groups
FROM (
    SELECT mercato_request_id, COALESCE(LOWER(BTRIM(position)), ''), COALESCE(LOWER(BTRIM(role)), '')
    FROM mercato_needs
    GROUP BY 1, 2, 3
    HAVING COUNT(*) > 1
) duplicates;

SELECT 'mercato_candidates' AS check_name, COUNT(*) AS duplicate_groups
FROM (
    SELECT mercato_need_id, player_id
    FROM mercato_candidates
    GROUP BY 1, 2
    HAVING COUNT(*) > 1
) duplicates;
