# data_collector.py
# Riot (LoL) data collector (SQLite):
# - Throttled requests (default ~0.8 rps) so you stay under 100 req / 120s
# - Seeds via Challenges-V1, League-Exp-V4, League-V4 entries (incl. EMERALD), or from file
# - Promote collected participants → seeds (snowball discovery)
# - Incremental harvest using startTime (DB dedupe)
# - Status / integrity readout
#
# CLI:
#   python data_collector.py init-db
#   python data_collector.py seed-challenges --platform na1 --max-players 1200
#   python data_collector.py seed-league-exp --platform na1 --tiers DIAMOND,EMERALD,PLATINUM --pages-per-div 10 --max-players 5000
#   python data_collector.py seed-league --platform na1 --tiers DIAMOND,EMERALD,PLATINUM --pages-per-div 20 --max-players 10000
#   python data_collector.py seed-from-file --platform na1 --path puuids.txt
#   python data_collector.py promote-participants --platform na1 --since-days 365 --reset-cursor
#   python data_collector.py harvest --platform na1 --since-days 60 --matches-per 100 --page-limit 1 --max-seeds 300 --min-interval 1.3 --queue 420
#   python data_collector.py status
#
# Env:
#   RIOT_API_KEY must be set. You can also set RIOT_MIN_INTERVAL (seconds) to override throttle default.

import os, time, random, argparse, sqlite3
from typing import List, Tuple, Optional
import requests
from tqdm import tqdm

RIOT_API_KEY = os.getenv("RIOT_API_KEY")
assert RIOT_API_KEY, "Set RIOT_API_KEY environment variable."
DB_PATH_DEFAULT = "riot_comp.db"

# ------------ rate limit throttle ------------
# ~100 requests / 120s => <= 0.83 rps => ~1.2–1.3s between calls
MIN_INTERVAL = float(os.getenv("RIOT_MIN_INTERVAL", "1.3"))
_LAST_CALL_TS = 0.0

# ------------ tiers/divisions ------------
APEX_TIERS = ("MASTER", "GRANDMASTER", "CHALLENGER")
DIVISIONED_TIERS = ("IRON","BRONZE","SILVER","GOLD","PLATINUM","EMERALD","DIAMOND")
ALL_TIERS = DIVISIONED_TIERS + APEX_TIERS
ALL_DIVISIONS = ("I","II","III","IV")

# -------------------------
# Riot API client
# -------------------------

S = requests.Session()
S.headers.update({"X-Riot-Token": RIOT_API_KEY})

def riot_get(url, params=None, retries=6):
    """Throttled GET with basic 429 handling."""
    global _LAST_CALL_TS
    for _ in range(retries):
        # throttle
        now = time.time()
        wait = MIN_INTERVAL - (now - _LAST_CALL_TS)
        if wait > 0:
            time.sleep(wait)
        # call
        r = S.get(url, params=params, timeout=20)
        _LAST_CALL_TS = time.time()

        if r.status_code == 429:
            retry_after = int(r.headers.get("Retry-After", "1"))
            time.sleep(retry_after + 1)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError("Too many rate-limit retries for URL: %s" % url)

def url(route: str, path: str) -> str:
    return f"https://{route}.api.riotgames.com{path}"

def platform_to_region(platform: str) -> str:
    p = platform.lower()
    if p in {"na1","br1","la1","la2","oc1"}: return "americas"
    if p in {"euw1","eun1","tr1","ru"}:       return "europe"
    if p in {"kr","jp1"}:                     return "asia"
    if p in {"ph2","sg2","th2","tw2","vn2"}:  return "sea"
    raise ValueError(f"Unknown platform: {platform}")

# --- Challenges-V1 (PUUIDs directly) ---
def challenges_leaderboard_puuids(platform: str, challenge_id: int, level: str, limit: int = 300) -> List[str]:
    # /lol/challenges/v1/challenges/{challenge_id}/leaderboards/by-level/{level}
    data = riot_get(url(platform, f"/lol/challenges/v1/challenges/{challenge_id}/leaderboards/by-level/{level}"))
    puuids = []
    for entry in data:
        p = entry.get("puuid")
        if p:
            puuids.append(p)
        if len(puuids) >= limit:
            break
    return puuids

# --- League-Exp-V4 + Summoner-V4 (convert to PUUIDs) ---
def league_exp_entries(platform: str, queue: str, tier: str, division: str, page: int = 1):
    # /lol/league-exp/v4/entries/{queue}/{tier}/{division}?page={page}
    path = f"/lol/league-exp/v4/entries/{queue}/{tier}/{division}"
    return riot_get(url(platform, path), params={"page": page})

def summoner_by_id(platform: str, encrypted_summoner_id: str):
    # /lol/summoner/v4/summoners/{encryptedSummonerId}
    return riot_get(url(platform, f"/lol/summoner/v4/summoners/{encrypted_summoner_id}"))


def summoner_by_name(platform: str, summoner_name: str):
    # Fallback (prefer by-id)
    from urllib.parse import quote
    return riot_get(url(platform, f"/lol/summoner/v4/summoners/by-name/{quote(summoner_name)}"))

# --- Classic League-V4 entries (incl. EMERALD) ---
def league_entries(platform: str, queue: str, tier: str, division: str, page: int = 1):
    # /lol/league/v4/entries/{queue}/{tier}/{division}?page={page}
    path = f"/lol/league/v4/entries/{queue}/{tier}/{division}"
    return riot_get(url(platform, path), params={"page": page})

# --- Match-V5 ---
def match_ids_by_puuid(region: str, puuid: str, start: int=0, count: int=20,
                       queue: Optional[int]=420, startTime: Optional[int]=None):
    params = {"start": start, "count": count}
    if queue is not None: params["queue"] = queue
    if startTime is not None: params["startTime"] = startTime
    return riot_get(url(region, f"/lol/match/v5/matches/by-puuid/{puuid}/ids"), params=params)

def get_match(region: str, match_id: str):
    return riot_get(url(region, f"/lol/match/v5/matches/{match_id}"))

# -------------------------
# SQLite schema/util
# -------------------------

DDL = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS seeds (
  platform TEXT NOT NULL,
  puuid    TEXT NOT NULL,
  last_start_time INTEGER,          -- unix seconds cursor for incremental fetch
  PRIMARY KEY (platform, puuid)
);

CREATE TABLE IF NOT EXISTS matches (
  match_id   TEXT PRIMARY KEY,
  platform   TEXT NOT NULL,
  region     TEXT NOT NULL,
  queue_id   INTEGER,
  patch      TEXT,
  game_start INTEGER,               -- unix ms if available; else 0
  game_end   INTEGER,
  duration_s INTEGER
);

CREATE TABLE IF NOT EXISTS teams (
  match_id TEXT NOT NULL,
  team_id  INTEGER NOT NULL,        -- 100 or 200
  win      INTEGER,                 -- 1 if win else 0
  PRIMARY KEY (match_id, team_id),
  FOREIGN KEY (match_id) REFERENCES matches(match_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS participants (
  match_id     TEXT NOT NULL,
  puuid        TEXT,
  team_id      INTEGER,
  champion_id  INTEGER,
  team_pos     TEXT,                -- teamPosition
  lane         TEXT,                -- legacy
  role         TEXT,                -- legacy
  PRIMARY KEY (match_id, puuid),
  FOREIGN KEY (match_id) REFERENCES matches(match_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_matches_queue ON matches(queue_id);
CREATE INDEX IF NOT EXISTS idx_matches_patch ON matches(patch);
CREATE INDEX IF NOT EXISTS idx_part_champ ON participants(champion_id);
CREATE INDEX IF NOT EXISTS idx_part_teamid ON participants(match_id, team_id);
"""

def db_connect(path: str):
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn

def db_init(conn):
    conn.executescript(DDL)
    conn.commit()

def seeds_add(conn, platform: str, puuids: List[str], last_start_time: Optional[int]=None):
    cur = conn.cursor()
    for p in puuids:
        cur.execute("INSERT OR IGNORE INTO seeds(platform, puuid, last_start_time) VALUES(?,?,?);",
                    (platform.lower(), p, last_start_time))
    conn.commit()

def seeds_list(conn, platform: Optional[str]=None) -> List[Tuple[str,str,int]]:
    cur = conn.cursor()
    if platform:
        cur.execute("SELECT platform, puuid, COALESCE(last_start_time,0) FROM seeds WHERE platform=?;", (platform.lower(),))
    else:
        cur.execute("SELECT platform, puuid, COALESCE(last_start_time,0) FROM seeds;")
    return cur.fetchall()

def seeds_update_start_time(conn, platform: str, puuid: str, start_time: int):
    conn.execute("UPDATE seeds SET last_start_time=? WHERE platform=? AND puuid=?;",
                 (start_time, platform.lower(), puuid))
    conn.commit()

def insert_match(conn, platform: str, region: str, m: dict) -> bool:
    mid = m.get("metadata", {}).get("matchId")
    info = m.get("info", {})
    if not mid: return False
    game_start = info.get("gameStartTimestamp") or info.get("gameCreation") or 0
    game_end   = info.get("gameEndTimestamp") or 0
    duration_s = info.get("gameDuration") or 0
    queue_id   = info.get("queueId")
    patch      = (info.get("gameVersion","").split(" ")[0]) if info.get("gameVersion") else None

    try:
        conn.execute("""
          INSERT OR IGNORE INTO matches(match_id, platform, region, queue_id, patch, game_start, game_end, duration_s)
          VALUES(?,?,?,?,?,?,?,?);
        """, (mid, platform.lower(), region, queue_id, patch, game_start, game_end, duration_s))
    except sqlite3.IntegrityError:
        return False

    for t in info.get("teams", []):
        conn.execute("""
          INSERT OR IGNORE INTO teams(match_id, team_id, win)
          VALUES(?,?,?);
        """, (mid, int(t.get("teamId",0)), 1 if t.get("win") else 0))

    for p in info.get("participants", []):
        conn.execute("""
          INSERT OR REPLACE INTO participants(match_id, puuid, team_id, champion_id, team_pos, lane, role)
          VALUES(?,?,?,?,?,?,?);
        """, (mid, p.get("puuid"), int(p.get("teamId",0)), int(p.get("championId",0)),
              p.get("teamPosition"), p.get("lane"), p.get("role")))
    conn.commit()
    return True

# -------------------------
# Seeding methods
# -------------------------

def seed_challenges(db_path: str, platform: str,
                    challenge_id: int = 0,
                    levels: List[str] = ["CHALLENGER","GRANDMASTER","MASTER"],
                    per_level: int = 300,
                    max_players: int = 1200,
                    rng_seed: int = 42):
    """Seed PUUIDs directly from Challenges leaderboards (apex tiers only)."""
    conn = db_connect(db_path); db_init(conn)
    random.seed(rng_seed)
    all_p: List[str] = []
    for lvl in levels:
        try:
            got = challenges_leaderboard_puuids(platform, challenge_id, lvl, per_level)
            if not isinstance(got, list) or not got:
                print(f"[challenges] empty level: {platform} {lvl}")
                continue
            all_p.extend(got)
        except Exception as e:
            print(f"[challenges] error {platform} {lvl}: {e}")
            continue
    random.shuffle(all_p)
    uniq, seen = [], set()
    for p in all_p:
        if p not in seen:
            uniq.append(p); seen.add(p)
        if len(uniq) >= max_players: break
    seeds_add(conn, platform, uniq, last_start_time=None)
    conn.close()
    print(f"Seeded {len(uniq)} PUUIDs for {platform} (Challenges)")

def seed_league_exp(db_path: str,
                    platform: str,
                    queue: str = "RANKED_SOLO_5x5",
                    tiers: List[str] = ["DIAMOND","EMERALD","PLATINUM"],
                    divisions: List[str] = ["I","II","III","IV"],
                    pages_per_div: int = 5,
                    max_players: int = 5000,
                    rng_seed: int = 42):
    """
    Seed via league-exp ladders; converts encryptedSummonerId → PUUID (by-id).
    Good for DIAMOND/EMERALD/PLATINUM volume.
    """
    conn = db_connect(db_path); db_init(conn)
    random.seed(rng_seed)
    puuids, seen = [], set()

    for tier in tiers:
        divs = ["I"] if tier.upper() in APEX_TIERS else divisions
        for div in divs:
            for page in range(1, pages_per_div+1):
                try:
                    entries = league_exp_entries(platform, queue, tier, div, page)
                    if not isinstance(entries, list) or len(entries) == 0:
                        print(f"[league-exp] empty: {platform} {tier} {div} page={page}")
                        break
                except Exception as e:
                    print(f"[league-exp] error {platform} {tier} {div} page={page}: {e}")
                    break
                for e in entries:
                    sid = e.get("summonerId")
                    sname = e.get("summonerName")
                    if not sid and not sname:
                        continue
                    try:
                        summ = summoner_by_id(platform, sid) if sid else summoner_by_name(platform, sname)
                    except Exception as ex:
                        print(f"[league-exp→summoner] fail sid={bool(sid)} name={sname!r}: {ex}")
                        continue
                    p = summ.get("puuid")
                    if p and p not in seen:
                        puuids.append(p); seen.add(p)
                        if len(puuids) >= max_players:
                            seeds_add(conn, platform, puuids, last_start_time=None)
                            conn.close()
                            print(f"Seeded {len(puuids)} PUUIDs for {platform} (League-Exp)")
                            return

    seeds_add(conn, platform, puuids, last_start_time=None)
    conn.close()
    print(f"Seeded {len(puuids)} PUUIDs for {platform} (League-Exp)")

def seed_league(db_path: str,
                platform: str,
                queue: str = "RANKED_SOLO_5x5",
                tiers: List[str] = ["DIAMOND","EMERALD","PLATINUM"],
                divisions: List[str] = ["I","II","III","IV"],
                pages_per_div: int = 20,
                max_players: int = 10000,
                rng_seed: int = 42):
    """
    Seed via classic league-v4 entries (reliable; supports EMERALD/PLAT/DIAMOND).
    Converts encryptedSummonerId → PUUID (by-id).
    """
    conn = db_connect(db_path); db_init(conn)
    random.seed(rng_seed)
    puuids, seen = [], set()

    for tier in tiers:
        if tier.upper() in APEX_TIERS:
            print(f"[league] skipping apex tier for entries: {tier}")
            continue
        for div in divisions:
            for page in range(1, pages_per_div+1):
                try:
                    entries = league_entries(platform, queue, tier, div, page)
                    if not isinstance(entries, list) or len(entries) == 0:
                        print(f"[league] empty: {platform} {tier} {div} page={page}")
                        break
                except Exception as e:
                    print(f"[league] error {platform} {tier} {div} page={page}: {e}")
                    break
                for e in entries:
                    sid = e.get("summonerId")
                    sname = e.get("summonerName")
                    if not sid and not sname:
                        continue
                    try:
                        summ = summoner_by_id(platform, sid) if sid else summoner_by_name(platform, sname)
                    except Exception as ex:
                        print(f"[league→summoner] fail sid={bool(sid)} name={sname!r}: {ex}")
                        continue
                    p = summ.get("puuid")
                    if p and p not in seen:
                        puuids.append(p); seen.add(p)
                        if len(puuids) >= max_players:
                            seeds_add(conn, platform, puuids, last_start_time=None)
                            conn.close()
                            print(f"Seeded {len(puuids)} PUUIDs for {platform} (League-V4 entries)")
                            return

    seeds_add(conn, platform, puuids, last_start_time=None)
    conn.close()
    print(f"Seeded {len(puuids)} PUUIDs for {platform} (League-V4 entries)")

def seed_from_file(db_path: str, platform: str, path: str):
    with open(path, "r", encoding="utf-8") as f:
        puuids = [line.strip() for line in f if line.strip()]
    conn = db_connect(db_path); db_init(conn)
    seeds_add(conn, platform, puuids, last_start_time=None)
    conn.close()
    print(f"Seeded {len(puuids)} PUUIDs from {path} for {platform}")

def promote_participants_to_seeds(db_path: str, platform: str, since_days: int = 90, reset_cursor: bool = False):
    """Promote distinct participant PUUIDs (seen in existing matches) into seeds."""
    conn = db_connect(db_path); db_init(conn)
    cutoff_ms = int((time.time() - since_days*24*3600) * 1000)
    q = """
    SELECT DISTINCT p.puuid
    FROM participants p
    JOIN matches m ON m.match_id = p.match_id
    WHERE m.platform = ? AND p.puuid IS NOT NULL AND m.game_start >= ?
    """
    puuids = [row[0] for row in conn.execute(q, (platform.lower(), cutoff_ms))]
    if not puuids:
        print("No participants found to promote."); conn.close(); return
    last = None if reset_cursor else int(time.time()) - 3600  # start near-now unless backfilling
    seeds_add(conn, platform, puuids, last_start_time=last)
    conn.close()
    print(f"Promoted {len(puuids)} participant PUUIDs into seeds for {platform} (reset_cursor={reset_cursor})")

# -------------------------
# Harvest (incremental + dedupe)
# -------------------------

def harvest(db_path: str, platform: str, matches_per: int, queue: int,
            since_days: int, page_limit: int = 6, seed_shuffle: bool=True,
            max_seeds: Optional[int]=None):
    """
    For each seed, fetch new match IDs using startTime >= last_start_time (or now - since_days if unset).
    Paginates with `start` up to `page_limit`. DB primary keys prevent duplicates.
    You can limit how many seeds to process via --max-seeds.
    """
    region = platform_to_region(platform)
    conn = db_connect(db_path); db_init(conn)
    seeds = seeds_list(conn, platform)
    if not seeds:
        print("No seeds in DB for this platform. Run a seed command first.")
        conn.close()
        return

    if seed_shuffle:
        random.shuffle(seeds)
    if max_seeds is not None and max_seeds > 0:
        seeds = seeds[:max_seeds]

    now_sec = int(time.time())
    default_start = now_sec - since_days*24*3600

    all_ids = set()
    for _, puuid, last_st in tqdm(seeds, desc="fetch match ids"):
        start_time = last_st or default_start
        start = 0
        pages = 0
        try:
            while pages < page_limit:
                ids = match_ids_by_puuid(region, puuid, start=start, count=matches_per, queue=queue, startTime=start_time)
                if not ids:
                    break
                all_ids.update(ids)
                start += len(ids)
                pages += 1
                if len(ids) < matches_per:
                    break
            # conservative: nudge cursor slightly behind now to avoid missing ultra-recent games
            seeds_update_start_time(conn, platform, puuid, max(start_time, now_sec - 300))
        except Exception:
            continue

    print(f"Unique match IDs to download: {len(all_ids)}")

    inserted = 0
    for mid in tqdm(list(all_ids), desc="download matches"):
        try:
            m = get_match(region, mid)
            if m.get("info", {}).get("queueId") != queue:
                continue
            if insert_match(conn, platform, region, m):
                inserted += 1
        except Exception:
            continue
    print(f"Inserted matches: {inserted}")
    conn.close()

# -------------------------
# Status / integrity
# -------------------------

def status(db_path: str):
    conn = db_connect(db_path)
    total_matches, distinct_matches = conn.execute("SELECT COUNT(*), COUNT(DISTINCT match_id) FROM matches;").fetchone()
    parts, dup_parts = conn.execute("""
        SELECT COUNT(*),
               COUNT(*) - COUNT(DISTINCT match_id || ':' || COALESCE(puuid,'noid'))
        FROM participants;""").fetchone()
    by_platform = list(conn.execute("SELECT platform, COUNT(*) FROM matches GROUP BY platform ORDER BY 2 DESC;"))
    latest_patch = conn.execute("SELECT patch, COUNT(*) AS c FROM matches WHERE patch IS NOT NULL GROUP BY patch ORDER BY patch DESC LIMIT 10;").fetchall()
    seeds_total = conn.execute("SELECT COUNT(*) FROM seeds;").fetchone()[0]
    conn.close()

    print(f"seeds: {seeds_total}")
    print(f"matches: {distinct_matches} distinct (rows: {total_matches})")
    print(f"participants: {parts} | duplicate (match_id, puuid): {dup_parts}")
    print("by platform:")
    for p,c in by_platform:
        print(f"  {p}: {c}")
    print("top recent patches:")
    for p,c in latest_patch:
        print(f"  {p}: {c}")

# -------------------------
# CLI
# -------------------------

def main():
    ap = argparse.ArgumentParser(description="Riot LoL Data Collector (SQLite, throttled)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init-db")
    p_init.add_argument("--db", default=DB_PATH_DEFAULT)

    p_seed = sub.add_parser("seed-challenges")
    p_seed.add_argument("--db", default=DB_PATH_DEFAULT)
    p_seed.add_argument("--platform", default="na1")
    p_seed.add_argument("--challenge-id", type=int, default=0)  # 0 = total challenge points
    p_seed.add_argument("--levels", default="CHALLENGER,GRANDMASTER,MASTER")
    p_seed.add_argument("--per-level", type=int, default=300)
    p_seed.add_argument("--max-players", type=int, default=1200)
    p_seed.add_argument("--seed", type=int, default=42)

    p_seedlexp = sub.add_parser("seed-league-exp")
    p_seedlexp.add_argument("--db", default=DB_PATH_DEFAULT)
    p_seedlexp.add_argument("--platform", default="na1")
    p_seedlexp.add_argument("--queue", default="RANKED_SOLO_5x5")
    p_seedlexp.add_argument("--tiers", default="DIAMOND,EMERALD,PLATINUM")
    p_seedlexp.add_argument("--divisions", default="I,II,III,IV")
    p_seedlexp.add_argument("--pages-per-div", type=int, default=5)
    p_seedlexp.add_argument("--max-players", type=int, default=5000)
    p_seedlexp.add_argument("--seed", type=int, default=42)

    p_seedl = sub.add_parser("seed-league")
    p_seedl.add_argument("--db", default=DB_PATH_DEFAULT)
    p_seedl.add_argument("--platform", default="na1")
    p_seedl.add_argument("--queue", default="RANKED_SOLO_5x5")
    p_seedl.add_argument("--tiers", default="DIAMOND,EMERALD,PLATINUM")
    p_seedl.add_argument("--divisions", default="I,II,III,IV")
    p_seedl.add_argument("--pages-per-div", type=int, default=20)
    p_seedl.add_argument("--max-players", type=int, default=10000)
    p_seedl.add_argument("--seed", type=int, default=42)

    p_seedf = sub.add_parser("seed-from-file")
    p_seedf.add_argument("--db", default=DB_PATH_DEFAULT)
    p_seedf.add_argument("--platform", default="na1")
    p_seedf.add_argument("--path", required=True, help="file with one PUUID per line")

    p_prom = sub.add_parser("promote-participants")
    p_prom.add_argument("--db", default=DB_PATH_DEFAULT)
    p_prom.add_argument("--platform", default="na1")
    p_prom.add_argument("--since-days", type=int, default=90)
    p_prom.add_argument("--reset-cursor", action="store_true")

    p_harv = sub.add_parser("harvest")
    p_harv.add_argument("--db", default=DB_PATH_DEFAULT)
    p_harv.add_argument("--platform", default="na1")
    p_harv.add_argument("--matches-per", type=int, default=50)
    p_harv.add_argument("--queue", type=int, default=420)
    p_harv.add_argument("--since-days", type=int, default=30)
    p_harv.add_argument("--page-limit", type=int, default=6)
    p_harv.add_argument("--max-seeds", type=int, default=None, help="only process this many seeds in this run")
    p_harv.add_argument("--min-interval", type=float, default=1.3, help="seconds between requests (throttle)")

    p_stat = sub.add_parser("status")
    p_stat.add_argument("--db", default=DB_PATH_DEFAULT)

    args = ap.parse_args()

    if args.cmd == "init-db":
        conn = db_connect(args.db); db_init(conn); conn.close()
        print(f"Initialized DB at {args.db}")

    elif args.cmd == "seed-challenges":
        levels = [x.strip().upper() for x in args.levels.split(",") if x.strip()]
        seed_challenges(args.db, args.platform, args.challenge_id, levels,
                        per_level=args.per_level, max_players=args.max_players, rng_seed=args.seed)

    elif args.cmd == "seed-league-exp":
        tiers = [t.strip().upper() for t in args.tiers.split(",") if t.strip()]
        divs  = [d.strip().upper() for d in args.divisions.split(",") if d.strip()]
        seed_league_exp(args.db, args.platform, args.queue, tiers, divs,
                        pages_per_div=args.pages_per_div, max_players=args.max_players, rng_seed=args.seed)

    elif args.cmd == "seed-league":
        tiers = [t.strip().upper() for t in args.tiers.split(",") if t.strip()]
        divs  = [d.strip().upper() for d in args.divisions.split(",") if d.strip()]
        seed_league(args.db, args.platform, args.queue, tiers, divs,
                    pages_per_div=args.pages_per_div, max_players=args.max_players, rng_seed=args.seed)

    elif args.cmd == "seed-from-file":
        seed_from_file(args.db, args.platform, args.path)

    elif args.cmd == "promote-participants":
        promote_participants_to_seeds(args.db, args.platform, since_days=args.since_days, reset_cursor=bool(args.reset_cursor))

    elif args.cmd == "harvest":
        # set global throttle for this run
        global MIN_INTERVAL
        MIN_INTERVAL = float(args.min_interval)
        harvest(args.db, args.platform, args.matches_per, args.queue, args.since_days,
                args.page_limit, max_seeds=args.max_seeds)

    elif args.cmd == "status":
        status(args.db)

if __name__ == "__main__":
    main()
