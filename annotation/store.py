"""SQLite persistence and tournament logic. Every decision is committed before the call returns.

Design notes
* Candidates are identified by content hash, never by screen position. A judgment stores the canonical
  slot order (candidate_a/candidate_b) *and* which candidate was on the left.
* Undo never deletes: judgments/resolutions are voided with a reason and stay auditable. Exports skip
  voided rows unless asked.
* A non-decisive response (TIE, BOTH_BAD, SKIP) is saved as the label and leaves the match unresolved.
  Advancing needs a separate explicit `resolve_match`, stored apart from the label.
"""

from __future__ import annotations

import json
import random
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from annotation import bracket

SCHEMA_VERSION = "annotation_db.v1"
UI_VERSION = "tournament_ui.v1"
RESPONSES = ("A_BETTER", "B_BETTER", "TIE", "BOTH_BAD", "SKIP")
DECISIVE = ("A_BETTER", "B_BETTER")
CHOICES = ("LEFT", "RIGHT", "TIE", "BOTH_BAD", "SKIP")
REASON_TAGS = ("text_too_small", "text_overflow", "unwanted_overlap", "poor_spacing",
               "bad_alignment", "broken_grouping", "weak_hierarchy", "other")
DEFAULT_DB = Path("data/annotations.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS candidates(
  candidate_id TEXT PRIMARY KEY, source_design_id TEXT NOT NULL, template_family_id TEXT NOT NULL,
  target_width INTEGER NOT NULL, target_height INTEGER NOT NULL,
  source_path TEXT NOT NULL, source_preview_path TEXT, candidate_path TEXT NOT NULL, preview_path TEXT NOT NULL,
  generator_id TEXT NOT NULL, generator_version TEXT NOT NULL, renderer_version TEXT NOT NULL,
  provenance_json TEXT NOT NULL, set_id TEXT NOT NULL, imported_at_utc TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS tournaments(
  tournament_id TEXT PRIMARY KEY, source_design_id TEXT NOT NULL, template_family_id TEXT NOT NULL,
  target_width INTEGER NOT NULL, target_height INTEGER NOT NULL, size INTEGER NOT NULL, seed INTEGER NOT NULL,
  annotator_id TEXT NOT NULL, source_review_status TEXT NOT NULL, created_at_utc TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS entrants(
  tournament_id TEXT NOT NULL REFERENCES tournaments, slot INTEGER NOT NULL,
  candidate_id TEXT NOT NULL REFERENCES candidates, PRIMARY KEY(tournament_id, slot),
  UNIQUE(tournament_id, candidate_id));
CREATE TABLE IF NOT EXISTS matches(
  match_id TEXT PRIMARY KEY, tournament_id TEXT NOT NULL REFERENCES tournaments,
  kind TEXT NOT NULL CHECK(kind IN ('bracket','extra','repeat')), round INTEGER NOT NULL, position INTEGER NOT NULL,
  candidate_a_id TEXT, candidate_b_id TEXT, repeat_of TEXT, forced_left_id TEXT);
CREATE TABLE IF NOT EXISTS judgments(
  judgment_id INTEGER PRIMARY KEY AUTOINCREMENT, comparison_id TEXT NOT NULL UNIQUE,
  presentation_token TEXT NOT NULL UNIQUE, tournament_id TEXT NOT NULL, match_id TEXT NOT NULL REFERENCES matches,
  candidate_a_id TEXT NOT NULL, candidate_b_id TEXT NOT NULL, screen_left_id TEXT NOT NULL,
  label TEXT NOT NULL CHECK(label IN ('A_BETTER','B_BETTER','TIE','BOTH_BAD','SKIP')),
  reason_tags TEXT NOT NULL, note TEXT NOT NULL, annotator_id TEXT NOT NULL, ui_version TEXT NOT NULL,
  created_at_utc TEXT NOT NULL, voided_at_utc TEXT, void_reason TEXT);
CREATE UNIQUE INDEX IF NOT EXISTS one_active_judgment ON judgments(match_id) WHERE voided_at_utc IS NULL;
CREATE TABLE IF NOT EXISTS resolutions(
  resolution_id INTEGER PRIMARY KEY AUTOINCREMENT, tournament_id TEXT NOT NULL, match_id TEXT NOT NULL REFERENCES matches,
  judgment_id INTEGER NOT NULL REFERENCES judgments, winner_candidate_id TEXT NOT NULL, method TEXT NOT NULL,
  annotator_id TEXT NOT NULL, created_at_utc TEXT NOT NULL, voided_at_utc TEXT, void_reason TEXT);
CREATE UNIQUE INDEX IF NOT EXISTS one_active_resolution ON resolutions(match_id) WHERE voided_at_utc IS NULL;
"""


class StaleMatch(RuntimeError):
    """The presented match is no longer the current, unjudged one (rerun, second tab, undo)."""


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def connect(path: Path | str = DEFAULT_DB) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10, isolation_level=None)  # explicit transactions
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA)
    conn.execute("INSERT OR IGNORE INTO meta VALUES('schema_version', ?)", (SCHEMA_VERSION,))
    return conn


class _Tx:
    def __init__(self, conn):
        self.conn = conn

    def __enter__(self):
        self.conn.execute("BEGIN IMMEDIATE")
        return self.conn

    def __exit__(self, exc_type, *_):
        self.conn.execute("ROLLBACK" if exc_type else "COMMIT")
        return False


def tx(conn) -> _Tx:
    return _Tx(conn)


def set_meta(conn, key: str, value: str) -> None:
    conn.execute("INSERT INTO meta VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))


def get_meta(conn, key: str, default: str | None = None) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return row["value"] if row else default


# ---------------------------------------------------------------- tournaments

def create_tournament(conn, candidate_ids: list[str], *, size: int = bracket.DEFAULT_SIZE, seed: int = 0,
                      annotator_id: str = "local_annotator", source_review_status: str = "unknown",
                      tournament_id: str | None = None) -> str:
    """Same source and same target are enforced here, not just in the UI."""
    bracket.round_count(size)
    ids = sorted(set(candidate_ids))
    rows = [conn.execute("SELECT * FROM candidates WHERE candidate_id=?", (c,)).fetchone() for c in ids]
    if any(r is None for r in rows):
        raise ValueError("unknown candidate ID")
    if len({(r["source_design_id"], r["target_width"], r["target_height"]) for r in rows}) != 1:
        raise ValueError("a tournament needs candidates of one source design and one target canvas")
    entrants = bracket.seed_entrants(ids, size, seed)
    first = rows[0]
    tid = tournament_id or "t" + uuid.uuid4().hex[:10]
    with tx(conn):
        conn.execute("INSERT INTO tournaments VALUES(?,?,?,?,?,?,?,?,?,?)",
                     (tid, first["source_design_id"], first["template_family_id"], first["target_width"],
                      first["target_height"], size, seed, annotator_id, source_review_status, now_utc()))
        conn.executemany("INSERT INTO entrants VALUES(?,?,?)", [(tid, i, c) for i, c in enumerate(entrants)])
        conn.executemany("INSERT INTO matches VALUES(?,?,?,?,?,NULL,NULL,NULL,NULL)",
                         [(f"{tid}:r{r}m{p}", tid, "bracket", r, p) for r, p in bracket.bracket_slots(size)])
        set_meta(conn, "last_tournament", tid)
    return tid


@dataclass(frozen=True)
class MatchView:
    match_id: str
    kind: str
    round: int
    position: int
    a: str | None
    b: str | None
    status: str  # waiting | ready | decided | unresolved | resolved
    winner: str | None
    left: str | None
    revision: int
    token: str
    judgment: sqlite3.Row | None
    resolution: sqlite3.Row | None
    repeat_of: str | None

    @property
    def right(self) -> str | None:
        if self.left is None:
            return None
        return self.b if self.left == self.a else self.a


@dataclass(frozen=True)
class TournamentState:
    tournament: sqlite3.Row
    entrants: tuple[str, ...]
    matches: dict[str, MatchView]  # bracket matches in play order, then extras

    def bracket(self) -> list[MatchView]:
        return [m for m in self.matches.values() if m.kind == "bracket"]

    def extras(self) -> list[MatchView]:
        return [m for m in self.matches.values() if m.kind != "bracket"]

    def by_slot(self, round_no: int, position: int) -> MatchView:
        return self.matches[f"{self.tournament['tournament_id']}:r{round_no}m{position}"]

    @property
    def champion(self) -> str | None:
        return self.by_slot(bracket.round_count(self.tournament["size"]), 0).winner

    @property
    def complete(self) -> bool:
        return self.champion is not None

    def eliminated(self) -> list[str]:
        return [m.b if m.winner == m.a else m.a for m in self.bracket() if m.winner and m.a and m.b]

    def next_match(self, kind: str = "bracket") -> MatchView | None:
        pool = self.extras() if kind == "extra" else self.bracket()
        return next((m for m in pool if m.status == "ready"), None)

    def unresolved(self) -> list[MatchView]:
        return [m for m in self.bracket() if m.status == "unresolved"]


def load_state(conn, tournament_id: str) -> TournamentState:
    t = conn.execute("SELECT * FROM tournaments WHERE tournament_id=?", (tournament_id,)).fetchone()
    if t is None:
        raise KeyError(f"unknown tournament {tournament_id}")
    entrants = tuple(r["candidate_id"] for r in conn.execute(
        "SELECT candidate_id FROM entrants WHERE tournament_id=? ORDER BY slot", (tournament_id,)))
    rows = conn.execute("SELECT * FROM matches WHERE tournament_id=? ORDER BY CASE kind WHEN 'bracket' THEN 0 ELSE 1 END, "
                        "round, position", (tournament_id,)).fetchall()
    judged = {r["match_id"]: r for r in conn.execute(
        "SELECT * FROM judgments WHERE tournament_id=? AND voided_at_utc IS NULL", (tournament_id,))}
    revisions = {r["match_id"]: r["n"] for r in conn.execute(
        "SELECT match_id, COUNT(*) n FROM judgments WHERE tournament_id=? GROUP BY match_id", (tournament_id,))}
    resolved = {r["match_id"]: r for r in conn.execute(
        "SELECT * FROM resolutions WHERE tournament_id=? AND voided_at_utc IS NULL", (tournament_id,))}
    views: dict[str, MatchView] = {}
    for row in rows:
        mid = row["match_id"]
        if row["kind"] == "bracket":
            if row["round"] == 1:
                a, b = entrants[2 * row["position"]], entrants[2 * row["position"] + 1]
            else:
                fa, fb = bracket.feeders(row["round"], row["position"])
                a = views[f"{tournament_id}:r{fa[0]}m{fa[1]}"].winner
                b = views[f"{tournament_id}:r{fb[0]}m{fb[1]}"].winner
        else:
            a, b = row["candidate_a_id"], row["candidate_b_id"]
        j, res, revision = judged.get(mid), resolved.get(mid), revisions.get(mid, 0)
        winner = None
        if j is not None:
            winner = {"A_BETTER": a, "B_BETTER": b}.get(j["label"])
            if winner is None and res is not None and row["kind"] == "bracket":
                winner = res["winner_candidate_id"]
        if a is None or b is None:
            status = "waiting"
        elif j is None:
            status = "ready"
        elif j["label"] in DECISIVE:
            status = "decided"
        else:
            status = "resolved" if res is not None else "unresolved"
        left = None
        if a and b:
            # A recorded judgment fixes what was actually shown; only unjudged matches derive it.
            left = (j["screen_left_id"] if j is not None else
                    row["forced_left_id"] or bracket.left_candidate(t["seed"], mid, a, b, revision))
        views[mid] = MatchView(mid, row["kind"], row["round"], row["position"], a, b, status, winner, left,
                               revision, f"{mid}#{revision}", j, res, row["repeat_of"])
    return TournamentState(t, entrants, views)


# ---------------------------------------------------------------- judgments

def record_judgment(conn, tournament_id: str, match_id: str, choice: str, *, presented_left: str,
                    presented_right: str, token: str, tags=(), note: str = "",
                    annotator_id: str = "local_annotator") -> tuple[sqlite3.Row, bool]:
    """Save one response. Returns (row, duplicate). Idempotent per presentation token."""
    if choice not in CHOICES:
        raise ValueError(f"choice must be one of {CHOICES}")
    tags = sorted(set(tags))
    if any(t not in REASON_TAGS for t in tags):
        raise ValueError("unknown reason tag")
    with tx(conn):
        existing = conn.execute("SELECT * FROM judgments WHERE presentation_token=?", (token,)).fetchone()
        if existing is not None:
            return existing, True
        match = load_state(conn, tournament_id).matches.get(match_id)
        if match is None or match.status != "ready" or match.token != token:
            raise StaleMatch("This match is no longer waiting for a decision.")
        if (presented_left, presented_right) != (match.left, match.right):
            raise StaleMatch("Displayed candidates do not match the stored presentation.")
        if choice == "LEFT":
            label = "A_BETTER" if match.left == match.a else "B_BETTER"
        elif choice == "RIGHT":
            label = "A_BETTER" if match.right == match.a else "B_BETTER"
        else:
            label = choice
        conn.execute(
            "INSERT INTO judgments(comparison_id,presentation_token,tournament_id,match_id,candidate_a_id,candidate_b_id,"
            "screen_left_id,label,reason_tags,note,annotator_id,ui_version,created_at_utc) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (uuid.uuid4().hex, token, tournament_id, match_id, match.a, match.b, match.left, label,
             json.dumps(tags), note, annotator_id, UI_VERSION, now_utc()))
        set_meta(conn, "last_tournament", tournament_id)
        return conn.execute("SELECT * FROM judgments WHERE presentation_token=?", (token,)).fetchone(), False


def resolve_match(conn, tournament_id: str, match_id: str, winner_id: str, annotator_id: str = "local_annotator") -> None:
    """Explicitly advance a candidate after TIE/BOTH_BAD/SKIP. The original label is untouched."""
    with tx(conn):
        match = load_state(conn, tournament_id).matches.get(match_id)
        if match is None or match.kind != "bracket" or match.status != "unresolved":
            raise StaleMatch("Only an unresolved bracket match can be resolved.")
        if winner_id not in (match.a, match.b):
            raise ValueError("winner must be one of the match's candidates")
        conn.execute("INSERT INTO resolutions(tournament_id,match_id,judgment_id,winner_candidate_id,method,annotator_id,"
                     "created_at_utc) VALUES(?,?,?,?,?,?,?)",
                     (tournament_id, match_id, match.judgment["judgment_id"], winner_id, "manual_resolution",
                      annotator_id, now_utc()))


def _void(conn, match_id: str, reason: str, *, judgment: bool = True) -> None:
    stamp = now_utc()
    conn.execute("UPDATE resolutions SET voided_at_utc=?, void_reason=? WHERE match_id=? AND voided_at_utc IS NULL",
                 (stamp, reason, match_id))
    if judgment:
        conn.execute("UPDATE judgments SET voided_at_utc=?, void_reason=? WHERE match_id=? AND voided_at_utc IS NULL",
                     (stamp, reason, match_id))


def _cascade(conn, state: TournamentState, match: MatchView, source_id: str) -> None:
    tid = state.tournament["tournament_id"]
    for r, p in bracket.downstream(match.round, match.position, state.tournament["size"]):
        _void(conn, f"{tid}:r{r}m{p}", f"upstream_changed:{source_id}")


def undo_match(conn, tournament_id: str, match_id: str, annotator_id: str = "local_annotator") -> None:
    """Void a match's judgment and resolution, and (bracket matches) every dependent later match."""
    with tx(conn):
        state = load_state(conn, tournament_id)
        match = state.matches.get(match_id)
        if match is None or match.judgment is None:
            raise StaleMatch("Nothing to undo for that match.")
        _void(conn, match_id, f"undone_by:{annotator_id}")
        if match.kind == "bracket":
            _cascade(conn, state, match, match_id)


def undo_last(conn, tournament_id: str, annotator_id: str = "local_annotator") -> str | None:
    """Undo the newest active action: a manual resolution first, otherwise the newest judgment."""
    with tx(conn):
        state = load_state(conn, tournament_id)
        res = conn.execute("SELECT * FROM resolutions WHERE tournament_id=? AND voided_at_utc IS NULL "
                           "ORDER BY resolution_id DESC LIMIT 1", (tournament_id,)).fetchone()
        jud = conn.execute("SELECT * FROM judgments WHERE tournament_id=? AND voided_at_utc IS NULL "
                           "ORDER BY judgment_id DESC LIMIT 1", (tournament_id,)).fetchone()
        if jud is None:
            return None
        if res is not None and res["created_at_utc"] >= jud["created_at_utc"]:
            match_id, judgment_too = res["match_id"], False
        else:
            match_id, judgment_too = jud["match_id"], True
        match = state.matches[match_id]
        _void(conn, match_id, f"undone_by:{annotator_id}", judgment=judgment_too)
        if match.kind == "bracket":
            _cascade(conn, state, match, match_id)
        return match_id


# ---------------------------------------------------------------- extra comparisons

def add_extra_matches(conn, tournament_id: str, mode: str, count: int, seed: int = 0) -> list[str]:
    """Optional extra labels. `eliminated_pairs`: never-compared pairs of eliminated candidates.
    `repeats`: re-present judged pairs with reversed screen order to measure consistency.
    Neither kind affects bracket advancement."""
    if mode not in ("eliminated_pairs", "repeats") or count < 1:
        raise ValueError("mode must be eliminated_pairs or repeats, count >= 1")
    with tx(conn):
        state = load_state(conn, tournament_id)
        existing = state.extras()
        rng = random.Random(f"extra:{state.tournament['seed']}:{seed}:{len(existing)}")
        if mode == "eliminated_pairs":
            seen = {frozenset((m.a, m.b)) for m in state.matches.values() if m.a and m.b and m.kind != "repeat"}
            pool = sorted(set(state.eliminated()))
            pairs = [frozenset((x, y)) for i, x in enumerate(pool) for y in pool[i + 1:]]
            pairs = sorted((p for p in pairs if p not in seen), key=sorted)
            rng.shuffle(pairs)
            plans = [(sorted(p), None, None) for p in pairs[:count]]
        else:
            done = {m.repeat_of for m in existing if m.kind == "repeat"}
            judged = [m for m in state.matches.values()
                      if m.kind != "repeat" and m.judgment is not None and m.judgment["label"] in DECISIVE
                      and m.match_id not in done]
            rng.shuffle(judged)
            plans = [([m.a, m.b], m.match_id, m.judgment["screen_left_id"]) for m in judged[:count]]
        made, n = [], len(existing)
        for (a, b), repeat_of, original_left in plans:
            n += 1
            mid = f"{tournament_id}:x{n}"
            forced = (b if original_left == a else a) if repeat_of else None
            conn.execute("INSERT INTO matches VALUES(?,?,?,?,?,?,?,?,?)",
                         (mid, tournament_id, "repeat" if repeat_of else "extra", 0, n, a, b, repeat_of, forced))
            made.append(mid)
        return made


# ---------------------------------------------------------------- queries

def list_tournaments(conn) -> list[sqlite3.Row]:
    return conn.execute("SELECT * FROM tournaments ORDER BY created_at_utc DESC").fetchall()


def get_candidate(conn, candidate_id: str) -> sqlite3.Row:
    row = conn.execute("SELECT * FROM candidates WHERE candidate_id=?", (candidate_id,)).fetchone()
    if row is None:
        raise KeyError(candidate_id)
    return row


def progress(state: TournamentState) -> tuple[int, int]:
    bracket_matches = state.bracket()
    return sum(m.status in ("decided", "resolved") for m in bracket_matches), len(bracket_matches)
