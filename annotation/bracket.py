"""Pure single-elimination bracket structure, seeding and left/right presentation (no I/O)."""

from __future__ import annotations

import hashlib
import random

SIZES = (4, 8, 16)
DEFAULT_SIZE = 8


def round_count(size: int) -> int:
    if not isinstance(size, int) or not 2 <= size <= 16:
        raise ValueError("tournament size must be between 2 and 16")
    return (size - 1).bit_length()


def round_name(round_no: int, size: int) -> str:
    remaining = (1 << round_count(size)) >> (round_no - 1)
    return {2: "Final", 4: "Semifinals", 8: "Quarterfinals", 16: "Round of 16"}[remaining]


def bracket_slots(size: int) -> list[tuple[int, int]]:
    """(round, position) for every bracket match, in play order."""
    capacity = 1 << round_count(size)
    return [(r, p) for r in range(1, round_count(size) + 1) for p in range(capacity >> r)]


def feeders(round_no: int, position: int) -> tuple[tuple[int, int], tuple[int, int]]:
    if round_no < 2:
        raise ValueError("round 1 matches are fed by entrants")
    return (round_no - 1, 2 * position), (round_no - 1, 2 * position + 1)


def downstream(round_no: int, position: int, size: int) -> list[tuple[int, int]]:
    """Every later match whose participants depend on this one."""
    chain = []
    while round_no < round_count(size):
        round_no, position = round_no + 1, position // 2
        chain.append((round_no, position))
    return chain


def seed_entrants(candidate_ids: list[str], size: int, seed: int) -> list[str]:
    """Reproducible selection and bracket order; independent of the input ordering."""
    round_count(size)
    unique = sorted(set(candidate_ids))
    if len(unique) < size:
        raise ValueError(f"need at least {size} distinct candidates, got {len(unique)}")
    rng = random.Random(f"bracket:{seed}")
    chosen = rng.sample(unique, size)
    rng.shuffle(chosen)
    return chosen


def left_candidate(seed: int, match_id: str, a_id: str, b_id: str, revision: int) -> str:
    """Reproducible pseudo-random screen-left candidate; the stored mapping is the record of truth."""
    digest = hashlib.sha256(f"{seed}|{match_id}|{a_id}|{b_id}|{revision}".encode()).digest()
    return a_id if digest[0] % 2 == 0 else b_id
