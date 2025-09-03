# file: numpy_optimized_simulation.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Mapping, Optional, Tuple, Iterable

import numpy as np
import pandas as pd
import heapq

# -------------------------------------------------
# Types & Data Classes
# -------------------------------------------------


@dataclass(frozen=True)
class Calendar:
    work_dates: pd.DatetimeIndex
    n_days: int
    iso_weeks_per_day: np.ndarray  # (n_days,) int
    dow_idx: np.ndarray  # (n_days,) int (0..4)
    weekday_codes_per_day: np.ndarray  # (n_days,) object/str
    slot_cat_idx_half: np.ndarray  # (2*n_days,) int (map to halfdays)
    slot_weeknums_half: np.ndarray  # (2*n_days,) int (ISO week)
    halfday_one_day: np.ndarray  # ["am"*9 + "pm"*9]
    times_day: np.ndarray  # 18 time floats


# -------------------------------------------------
# Global constants (slots/time grid)
# -------------------------------------------------

SLOT_LEN_HOURS: float = 0.5
SLOTS_PER_HALFDAY: int = 9
SLOTS_PER_DAY: int = 18

DAY_CODES = np.array(["mon", "tue", "wed", "thu", "fri"], dtype=object)

SLOTS_AM = np.arange(8.0, 12.5, 0.5)
SLOTS_PM = np.arange(12.5, 17.0, 0.5)
TIMES_DAY = np.concatenate([SLOTS_AM, SLOTS_PM])


# -------------------------------------------------
# Calendar & setup
# -------------------------------------------------


def prepare_calendar(
    start: date,
    end: date,
    week_factor: Dict[str, float],
) -> Tuple[Calendar, Dict[str, int], np.ndarray, np.ndarray, List[str], np.ndarray]:
    """Build calendar arrays and halfday category mapping."""
    all_dates = pd.date_range(start=start, end=end, freq="D")
    work_dates = all_dates[all_dates.weekday < 5]
    n_days = len(work_dates)

    iso_weeks_per_day = work_dates.isocalendar().week.to_numpy().astype(int)
    dow_idx = work_dates.dayofweek.to_numpy()
    weekday_codes_per_day = DAY_CODES[dow_idx]

    week_factor_halfdays = {d: {"am": 0.5, "pm": 0.5} for d in week_factor}
    halfday_distribution = {
        f"{d}_{p}": week_factor[d] * pf
        for d in week_factor
        for p, pf in week_factor_halfdays[d].items()
    }
    halfdays = list(halfday_distribution.keys())
    halfday_probs = np.array(list(halfday_distribution.values()), dtype=float)
    halfday_probs /= halfday_probs.sum()

    halfday_to_idx = {hd: i for i, hd in enumerate(halfdays)}
    am_idx_map = np.array([halfday_to_idx[f"{d}_am"] for d in DAY_CODES])
    pm_idx_map = np.array([halfday_to_idx[f"{d}_pm"] for d in DAY_CODES])

    slot_cat_idx_half = np.empty(2 * n_days, dtype=int)
    slot_cat_idx_half[0::2] = am_idx_map[dow_idx]
    slot_cat_idx_half[1::2] = pm_idx_map[dow_idx]

    slot_weeknums_half = np.repeat(iso_weeks_per_day, 2)

    halfday_one_day = np.array(
        [*(["am"] * SLOTS_PER_HALFDAY), *(["pm"] * SLOTS_PER_HALFDAY)], dtype=object
    )

    cal = Calendar(
        work_dates=work_dates,
        n_days=n_days,
        iso_weeks_per_day=iso_weeks_per_day,
        dow_idx=dow_idx,
        weekday_codes_per_day=weekday_codes_per_day,
        slot_cat_idx_half=slot_cat_idx_half,
        slot_weeknums_half=slot_weeknums_half,
        halfday_one_day=halfday_one_day,
        times_day=TIMES_DAY,
    )

    return cal, halfday_to_idx, am_idx_map, pm_idx_map, halfdays, halfday_probs


def build_week_weighting_from_weeks(
    weeks: Iterable[int], weight: float, *, base: Optional[Mapping[int, float]] = None
) -> Dict[int, float]:
    """Create/extend a week_weighting mapping (ISO 1..53)."""
    out: Dict[int, float] = dict(base) if base else {}
    w = float(weight)
    for kw in weeks:
        if 1 <= int(kw) <= 53:
            out[int(kw)] = w
    return out


def scale_week_weighting(
    base: Optional[Mapping[int, float]], factors: Mapping[int, float]
) -> Dict[int, float]:
    """Multiply weights per specified ISO week (default baseline 1.0)."""
    out: Dict[int, float] = dict(base) if base else {}
    for week, factor in factors.items():
        if 1 <= int(week) <= 53:
            out[int(week)] = float(out.get(int(week), 1.0)) * float(factor)
    return out


# -------------------------------------------------
# BG drawing (quantized)
# -------------------------------------------------


def draw_bgs(
    rng: np.random.Generator,
    num_employees: int,
    target_rate: float,
    min_bg: float,
    max_bg: float,
    step_bg: float,
    tolerance: float,
    max_trials: int = 1000,
) -> np.ndarray:
    """Draw BGs (employment rates) on a quantized grid close to the target mean."""
    if num_employees == 0:
        return np.empty(0, dtype=float)

    values = np.arange(min_bg, max_bg + step_bg / 2, step_bg)
    probs = np.maximum(0.0, 1.0 - np.abs(values - target_rate) / (max_bg - min_bg))
    if probs.sum() == 0:
        probs = np.ones_like(values, dtype=float)
    probs = probs / probs.sum()

    bgs: Optional[np.ndarray] = None
    for _ in range(max_trials):
        bgs = rng.choice(values, size=num_employees, replace=True, p=probs)
        if abs(bgs.mean() - target_rate) <= tolerance:
            return bgs

    if bgs is None or bgs.size == 0:
        return np.empty(0, dtype=float)

    # Nudge mean minimally within grid
    diff = target_rate - bgs.mean()
    steps_needed = int(round(diff / step_bg * num_employees))
    if steps_needed > 0:
        idxs = np.where(bgs < max_bg)[0]
        if idxs.size:
            chosen = rng.choice(idxs, size=min(steps_needed, idxs.size), replace=False)
            bgs[chosen] = np.clip(bgs[chosen] + step_bg, min_bg, max_bg)
    elif steps_needed < 0:
        idxs = np.where(bgs > min_bg)[0]
        if idxs.size:
            chosen = rng.choice(idxs, size=min(-steps_needed, idxs.size), replace=False)
            bgs[chosen] = np.clip(bgs[chosen] - step_bg, min_bg, max_bg)
    return bgs


# -------------------------------------------------
# Vectorized sampling helpers
# -------------------------------------------------


def _topk_mask_per_row(scores: np.ndarray, k_per_row: np.ndarray) -> np.ndarray:
    """Row-wise top-k selection mask via thresholding."""
    n_rows, n_cols = scores.shape
    k = np.clip(k_per_row.astype(int), 0, n_cols)
    if n_cols == 0 or n_rows == 0:
        return np.zeros_like(scores, dtype=bool)
    sort_scores = np.sort(scores, axis=1)
    idx = n_cols - k
    thresh = np.full(n_rows, np.inf, dtype=scores.dtype)
    take = idx < n_cols
    thresh[take] = sort_scores[np.arange(n_rows)[take], idx[take]]
    return scores >= thresh[:, None]


def sample_working_weeks_vectorized(
    rng: np.random.Generator,
    iso_weeks: np.ndarray,
    week_weights: np.ndarray,
    num_employees: int,
    weeks_not_working: int,
) -> np.ndarray:
    """Weighted sampling without replacement of working ISO weeks per employee."""
    total_weeks = int(iso_weeks.size)
    weeks_working = max(0, total_weeks - int(weeks_not_working))

    week_map = np.zeros((num_employees, 54), dtype=bool)
    if weeks_working == 0 or total_weeks == 0 or num_employees == 0:
        return week_map

    logp = np.full(total_weeks, -np.inf, dtype=float)
    pos = week_weights > 0
    logp[pos] = np.log(week_weights[pos])

    g = rng.gumbel(size=(num_employees, total_weeks))
    scores = logp[None, :] + g

    if weeks_working >= total_weeks:
        chosen_cols = [np.arange(total_weeks)] * num_employees
    else:
        k_vec = np.full(num_employees, weeks_working, dtype=int)
        mask = _topk_mask_per_row(scores, k_vec)
        chosen_cols = [np.flatnonzero(mask[i]) for i in range(num_employees)]

    for i, cols in enumerate(chosen_cols):
        week_map[i, iso_weeks[cols]] = True
    return week_map


def build_category_mask_vectorized(
    rng: np.random.Generator,
    num_employees: int,
    bgs: np.ndarray,
    n_cats: int,
    halfday_probs: np.ndarray,
) -> np.ndarray:
    """Pick distinct halfday categories per employee: k_i ≈ round(bg_i*10)."""
    if num_employees == 0 or n_cats == 0:
        return np.zeros((num_employees, n_cats), dtype=bool)
    k_per_row = np.rint(bgs * 10).astype(int)
    logp = np.full(n_cats, -np.inf, dtype=float)
    pos = halfday_probs > 0
    logp[pos] = np.log(halfday_probs[pos])
    scores = logp[None, :] + rng.gumbel(size=(num_employees, n_cats))
    return _topk_mask_per_row(scores, k_per_row)


# -------------------------------------------------
# Meetings (distributions passed in)
# -------------------------------------------------


def generate_meetings(
    rng: np.random.Generator,
    total_hours: float,
    size_vals: np.ndarray,
    size_probs: np.ndarray,
    dur_vals: np.ndarray,
    dur_probs: np.ndarray,
    start_vals: np.ndarray,
    start_probs: np.ndarray,
    *,
    max_draws: int = 100,
) -> np.ndarray:
    """Draw meetings until person-hour budget exhausted; keep first exceeding one."""
    if total_hours <= 0:
        return np.empty((0, 5), dtype=float)

    sizes = rng.choice(size_vals, size=max_draws, p=size_probs)
    durations = rng.choice(dur_vals, size=max_draws, p=dur_probs)
    starts = rng.choice(start_vals, size=max_draws, p=start_probs)

    ends = starts + durations
    valid = ends <= 17.0
    sizes, durations, starts, ends = (
        sizes[valid],
        durations[valid],
        starts[valid],
        ends[valid],
    )
    if sizes.size == 0:
        return np.empty((0, 5), dtype=float)

    person_hours = sizes * durations
    cum = np.cumsum(person_hours)

    keep = cum <= total_hours
    if np.any(~keep):
        keep[np.argmax(~keep)] = True
    if not np.any(keep):
        return np.empty((0, 5), dtype=float)

    return np.stack(
        [sizes[keep], durations[keep], person_hours[keep], starts[keep], ends[keep]],
        axis=1,
    )


# -------------------------------------------------
# Meeting assignment & cleardesk (NumPy-only)
# -------------------------------------------------


def _assign_meetings_numpy(
    rng: np.random.Generator,
    it: int,
    present: np.ndarray,  # (E,D,S) int8/bool
    meeting: np.ndarray,  # (E,D,S) int8
    meeting_id: np.ndarray,  # (E,D,S) int32
    times_day: np.ndarray,  # (S,) float
    unit: str,
    day_idx: int,
    date_ts: pd.Timestamp,
    next_meeting_id: int,
    meeting_records: List[List[object]],
    meetings: np.ndarray,  # (M,5)
) -> Tuple[int, List[List[object]]]:
    if meetings.size == 0:
        return next_meeting_id, meeting_records

    time_to_idx = {float(t): i for i, t in enumerate(times_day)}

    for size, dur, ph, start, end in meetings:
        s_idx = time_to_idx.get(float(start))
        e_idx = time_to_idx.get(float(end))
        if s_idx is None or e_idx is None:
            continue
        need_slots = e_idx - s_idx
        if need_slots <= 0:
            continue

        window = present[:, day_idx, s_idx:e_idx].astype(bool) & (
            meeting[:, day_idx, s_idx:e_idx] == 0
        )
        if window.size == 0:
            continue
        avail_full = window.sum(axis=1) == need_slots
        emp_candidates = np.flatnonzero(avail_full)
        if emp_candidates.size == 0:
            continue

        choose_n = min(int(size), emp_candidates.size)
        chosen = rng.choice(emp_candidates, size=choose_n, replace=False)

        meeting[chosen, day_idx, s_idx:e_idx] = 1
        meeting_id[chosen, day_idx, s_idx:e_idx] = next_meeting_id

        meeting_records.append(
            [
                it,
                int(next_meeting_id),
                unit,
                date_ts,
                int(date_ts.isocalendar().week),
                int(size),
                float(dur),
                float(ph),
                float(start),
                float(end),
            ]
        )
        next_meeting_id += 1

    return next_meeting_id, meeting_records


def _run_lengths_meeting_segments(m: np.ndarray) -> np.ndarray:
    """Length of consecutive 1s per slot for 3D (E,D,S)."""
    m_bool = m.astype(bool)
    c1 = np.cumsum(m_bool, axis=2)
    z1 = np.maximum.accumulate(np.where(~m_bool, c1, 0), axis=2)
    left = c1 - z1

    mr = m_bool[..., ::-1]
    c2 = np.cumsum(mr, axis=2)
    z2 = np.maximum.accumulate(np.where(~mr, c2, 0), axis=2)
    right = (c2 - z2)[..., ::-1]

    return left + right - 1


# -------------------------------------------------
# Post helpers
# -------------------------------------------------


def compute_meeting_room_size(
    all_meetings: pd.DataFrame,
    room_cap: Mapping[str, float],
    *,
    overflow_label: str = "unbekannt",
    categories: Optional[List[str]] = None,
) -> pd.Categorical:
    """Map participant size -> room label using `room_cap` thresholds (klein <=4, mittel <=6, gross >6)."""
    size = np.asarray(all_meetings["size"], dtype=float)

    # Sort thresholds ascending by cap
    items = sorted(
        ((str(k), float(v)) for k, v in room_cap.items()), key=lambda kv: kv[1]
    )
    if not items:
        # no thresholds → everything overflow
        out = np.full(size.shape, overflow_label, dtype=object)
        cats = [overflow_label] if categories is None else categories
        return pd.Categorical(out, categories=cats)

    labels = np.array([k for k, _ in items], dtype=object)
    caps = np.array([v for _, v in items], dtype=float)

    # Find first cap >= size
    idx = np.searchsorted(caps, size, side="left")

    out = np.full(size.shape, overflow_label, dtype=object)
    in_range = idx < caps.size
    if np.any(in_range):
        out[in_range] = labels[idx[in_range]]

    if categories is None:
        cats = list(labels.tolist()) + [overflow_label]
    else:
        cats = categories

    return pd.Categorical(out, categories=cats)


def compute_required_rooms(
    all_meetings: pd.DataFrame,
    *,
    slot_times: np.ndarray,
    by: Tuple[str, ...] = ("replication", "unit", "date", "meeting_room_size"),
    return_peak_time: bool = False,
) -> pd.DataFrame:
    """Compute required meeting rooms as the **max concurrent** meetings per group.

    Vectorized line-sweep using a difference array per group; no Python loops over days.

    Parameters
    ----------
    all_meetings : DataFrame
        Requires columns in `by` plus 'start_time' and 'end_time'.
    slot_times : np.ndarray
        Sorted array of slot *start* times (e.g. np.array([8.0, 8.5, ..., 16.5])).
        Slot length is inferred from the spacing.
    by : tuple[str, ...], default ("replication","unit","date","meeting_room_size")
        Grouping dimensions for which to compute room requirements.
    return_peak_time : bool, default False
        If True, add column 'peak_time' with the slot start time where the peak occurs.

    Returns
    -------
    DataFrame with columns: *by*, ['peak_time',] 'required_rooms'.
    """
    if all_meetings.empty:
        cols = (
            list(by) + (["peak_time"] if return_peak_time else []) + ["required_rooms"]
        )
        return pd.DataFrame(columns=cols)

    # Validate required columns
    for col in ("start_time", "end_time", *by):
        if col not in all_meetings.columns:
            raise KeyError(f"all_meetings missing required column '{col}'")

    df = all_meetings.loc[:, list(by) + ["start_time", "end_time"]].copy()

    # Factorize tuple key for vectorized accumulation
    grp_vals = [df[k].to_numpy() for k in by]
    grp_codes, grp_uniques = pd.factorize(list(zip(*grp_vals)), sort=False)
    G = int(grp_uniques.size)

    # Slot grid & length
    slot_times = np.asarray(slot_times, dtype=float)
    if slot_times.ndim != 1 or slot_times.size == 0:
        raise ValueError("slot_times must be a 1D non-empty array")
    n_slots = int(slot_times.size)
    slot_len = float(np.median(np.diff(slot_times))) if n_slots > 1 else 0.5

    # Convert to [start_idx, end_idx) on the grid
    start_arr = df["start_time"].to_numpy(float)
    end_arr = df["end_time"].to_numpy(float)
    s_idx = np.rint((start_arr - slot_times[0]) / slot_len).astype(int)
    dur_slots = np.rint((end_arr - start_arr) / slot_len).astype(int)
    e_idx = s_idx + dur_slots

    # Clip to grid bounds
    s_idx = np.clip(s_idx, 0, n_slots)
    e_idx = np.clip(e_idx, 0, n_slots)

    # Difference-array approach across all groups at once
    stride = n_slots + 1
    starts_lin = grp_codes * stride + s_idx
    ends_lin = grp_codes * stride + e_idx

    # Accumulate +1 at starts, -1 at ends
    diff = np.bincount(starts_lin, minlength=G * stride) - np.bincount(
        ends_lin, minlength=G * stride
    )
    diff = diff.reshape(G, stride)

    occ = np.cumsum(diff, axis=1)[:, :n_slots]
    required = occ.max(axis=1)

    if return_peak_time:
        peak_idx = occ.argmax(axis=1)
        peak_time = slot_times[peak_idx]

    # Build result frame
    out = pd.DataFrame(list(grp_uniques), columns=list(by))
    out["required_rooms"] = required.astype(int)
    if return_peak_time:
        out["peak_time"] = peak_time.astype(float)

    cols = list(by) + (["peak_time"] if return_peak_time else []) + ["required_rooms"]
    return out[cols]


# -------------------------------------------------
# Room occupancy (slot grid per room)
# -------------------------------------------------


def build_room_occupancy_slots(
    all_meetings: pd.DataFrame,
    *,
    slot_times: np.ndarray = TIMES_DAY,
    by: Tuple[str, ...] = ("replication", "unit", "date", "meeting_room_size"),
    room_col: str = "room_id",
    include_idle: bool = False,
) -> pd.DataFrame:
    """Create a 30-min room occupancy grid from `all_meetings`.

    The function assigns a minimal number of rooms per group (interval coloring)
    and expands meetings into 30-min slots.

    Parameters
    ----------
    all_meetings : DataFrame
        Must contain columns: *by*, 'meeting_id', 'start_time', 'end_time'.
    slot_times : np.ndarray
        Start times of slots for a working day (e.g., TIMES_DAY). Slot length is
        inferred from spacing (default 0.5h).
    by : tuple[str,...]
        Grouping key for independent room pools (default: replication/unit/date/size).
    room_col : str
        Output column name for room index per group (0..R-1).
    include_idle : bool
        If True, also output empty slots (busy=0) for every room and slot.

    Returns
    -------
    DataFrame with columns: *by*, room_col, 'time_idx', 'time_float',
    'meeting_id', 'busy' (int8). Busy rows have busy=1; if include_idle, idle rows
    have busy=0 and meeting_id=-1.
    """
    if all_meetings.empty:
        cols = list(by) + [room_col, "time_idx", "time_float", "meeting_id", "busy"]
        return pd.DataFrame(columns=cols)

    required = {c for c in (*by, "meeting_id", "start_time", "end_time")}
    missing = required - set(all_meetings.columns)
    if missing:
        raise KeyError(f"all_meetings missing required columns: {sorted(missing)}")

    df = all_meetings.loc[:, list(by) + ["meeting_id", "start_time", "end_time"]].copy()

    # Slot grid and slot length
    slot_times = np.asarray(slot_times, dtype=float)
    if slot_times.ndim != 1 or slot_times.size == 0:
        raise ValueError("slot_times must be a 1D non-empty array")
    n_slots = int(slot_times.size)
    slot_len = float(np.median(np.diff(slot_times))) if n_slots > 1 else 0.5

    # Group factorization to integer codes
    grp_vals = [df[k].to_numpy() for k in by]
    grp_codes, grp_uniques = pd.factorize(list(zip(*grp_vals)), sort=False)
    G = int(grp_uniques.size)

    # Convert times to slot indices
    start_arr = df["start_time"].to_numpy(float)
    end_arr = df["end_time"].to_numpy(float)
    s_idx = np.rint((start_arr - slot_times[0]) / slot_len).astype(int)
    e_idx = np.rint((end_arr - slot_times[0]) / slot_len).astype(int)  # half-open
    s_idx = np.clip(s_idx, 0, n_slots)
    e_idx = np.clip(e_idx, 0, n_slots)

    # Greedy interval coloring per group (minimal rooms)
    room_ids = np.full(df.shape[0], -1, dtype=np.int32)

    order = np.lexsort((s_idx, grp_codes))  # group then start
    grp_sorted = grp_codes[order]
    s_sorted = s_idx[order]
    e_sorted = e_idx[order]
    mid_sorted = df["meeting_id"].to_numpy()[order]

    # boundaries of groups in sorted order
    bounds = np.flatnonzero(np.r_[True, grp_sorted[1:] != grp_sorted[:-1], True])

    for g_start, g_end in zip(bounds[:-1], bounds[1:]):
        # min-heap of (end_idx, room_id) for active meetings
        active: list[tuple[int, int]] = []
        free: list[int] = []
        next_room = 0
        for k in range(g_start, g_end):
            s, e = int(s_sorted[k]), int(e_sorted[k])
            # free rooms that ended before current start
            while active and active[0][0] <= s:
                _, rid = heapq.heappop(active)
                heapq.heappush(free, rid)
            if free:
                rid = heapq.heappop(free)
            else:
                rid = next_room
                next_room += 1
            heapq.heappush(active, (e, rid))
            room_ids[order[k]] = rid

    df[room_col] = room_ids

    # Expand meetings to slot rows (busy slots only)
    lengths = (e_idx - s_idx).astype(int)
    keep = lengths > 0
    if not np.any(keep):
        cols = list(by) + [room_col, "time_idx", "time_float", "meeting_id", "busy"]
        return pd.DataFrame(columns=cols)

    df_nonzero = df.loc[keep].reset_index(drop=True)
    s_nz = s_idx[keep]
    e_nz = e_idx[keep]
    len_nz = lengths[keep]

    # Build ranges per meeting
    # For performance, use Python-level loop to create ranges; typically fast enough.
    time_idx_blocks = [np.arange(s, e, dtype=np.int16) for s, e in zip(s_nz, e_nz)]
    time_idx = np.concatenate(time_idx_blocks)

    rep_blocks = [np.repeat(df_nonzero[k].to_numpy(), len_nz) for k in by]
    room_block = np.repeat(df_nonzero[room_col].to_numpy(), len_nz).astype(np.int32)
    mid_block = np.repeat(df_nonzero["meeting_id"].to_numpy(), len_nz).astype(np.int64)

    out = pd.DataFrame(
        {
            **{k: v for k, v in zip(by, rep_blocks)},
            room_col: room_block,
            "time_idx": time_idx,
        }
    )

    out["time_float"] = slot_times[out["time_idx"].to_numpy()].astype(np.float32)
    out["meeting_id"] = mid_block
    out["busy"] = np.ones(out.shape[0], dtype=np.int8)

    if include_idle:
        # Create full grid per (group, room_id) x time slot and left-join
        # Compute rooms per group
        grp_key_df = pd.DataFrame(list(grp_uniques), columns=list(by))
        grp_key_df["__grp_id__"] = np.arange(G, dtype=np.int32)
        df = df.merge(grp_key_df, left_on=list(by), right_on=list(by), how="left")
        rooms_per_grp = pd.DataFrame(
            {
                "__grp_id__": df.groupby("__grp_id__")[room_col]
                .max()
                .fillna(-1)
                .astype(int)
                .index,
                "max_room": df.groupby("__grp_id__")[room_col]
                .max()
                .fillna(-1)
                .astype(int)
                .values,
            }
        )
        # Build grid
        rows = []
        for gid, max_room in zip(
            rooms_per_grp["__grp_id__"].to_numpy(), rooms_per_grp["max_room"].to_numpy()
        ):
            if max_room < 0:
                continue
            # group key values
            gvals = grp_key_df.loc[grp_key_df["__grp_id__"] == gid, list(by)].iloc[0]
            rr, tt = np.meshgrid(
                np.arange(max_room + 1, dtype=np.int32),
                np.arange(n_slots, dtype=np.int16),
                indexing="ij",
            )
            base = pd.DataFrame(
                {
                    **{k: np.repeat(gvals[k], rr.size) for k in by},
                    room_col: rr.ravel(),
                    "time_idx": tt.ravel(),
                }
            )
            rows.append(base)
        if rows:
            full_grid = pd.concat(rows, ignore_index=True)
            full_grid["time_float"] = slot_times[
                full_grid["time_idx"].to_numpy()
            ].astype(np.float32)
            occ = out.merge(
                full_grid,
                on=list(by) + [room_col, "time_idx", "time_float"],
                how="right",
            )
            occ["busy"] = occ["busy"].fillna(0).astype(np.int8)
            occ["meeting_id"] = occ["meeting_id"].fillna(-1).astype(np.int64)
            return occ.sort_values(list(by) + [room_col, "time_idx"]).reset_index(
                drop=True
            )

    return out.sort_values(list(by) + [room_col, "time_idx"]).reset_index(drop=True)


# -------------------------------------------------
# Public API: run_simulation (NumPy-first; build DFs at the end)
# -------------------------------------------------


def run_simulation(
    *,
    start_date: date,
    end_date: date,
    week_factor: Dict[str, float],
    profiles: Dict[str, Dict[str, float]],
    min_bg: float = 0.4,
    max_bg: float = 1.0,
    step_bg: float = 0.1,
    employment_rate_variability: float = 0.01,
    weeks_not_working: int = 7,
    iterations: int = 1,
    seed: int | None = 42,
    min_cleardesk_hours: float = 1.5,
    meeting_room_max_size: Mapping[str, int] = None,
    week_weighting: Optional[Mapping[int, float]] = None,
    meeting_size_dist: Mapping[int, float] = None,
    meeting_duration_dist: Mapping[float, float] = None,
    meeting_start_time_dist: Mapping[float, float] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run full simulation in NumPy; create pandas DataFrames once at the end."""
    rng = np.random.default_rng(seed)

    # Calendar & categories
    cal, _, _, _, halfdays, halfday_probs = prepare_calendar(
        start_date, end_date, week_factor
    )

    # Week weights from mapping (normalize over weeks present)
    weeks = pd.date_range(start=start_date, end=end_date, freq="W-MON")
    iso_weeks = weeks.isocalendar().week.to_numpy().astype(int)
    if week_weighting is None:
        week_weights = np.ones(len(weeks), dtype=float)
    else:
        week_weights = np.array(
            [float(week_weighting.get(int(w), 1.0)) for w in iso_weeks], dtype=float
        )
    week_weights[~np.isfinite(week_weights) | (week_weights < 0)] = 0.0
    if week_weights.sum() <= 0:
        week_weights = np.ones(len(weeks), dtype=float)
    week_weights = week_weights / week_weights.sum()

    def _prep_dist(
        d: Mapping[float, float], dtype=float
    ) -> Tuple[np.ndarray, np.ndarray]:
        vals = np.array(list(d.keys()), dtype=dtype)
        probs = np.array(list(d.values()), dtype=float)
        s = probs.sum()
        probs = probs / s if s > 0 else np.ones_like(probs) / len(probs)
        return vals, probs

    size_vals, size_probs = _prep_dist(meeting_size_dist, dtype=int)
    dur_vals, dur_probs = _prep_dist(meeting_duration_dist, dtype=float)
    start_vals, start_probs = _prep_dist(meeting_start_time_dist, dtype=float)

    all_meeting_records: List[List[object]] = []
    next_meeting_id = 0

    min_cleardesk_slots = int(round(min_cleardesk_hours / SLOT_LEN_HOURS))

    final_cols: Dict[str, List[np.ndarray]] = {
        "replication": [],
        "date_idx": [],
        "time_idx": [],
        "unit": [],
        "employee_id": [],
        "bg": [],
        "present": [],
        "meeting": [],
        "einzel_ap": [],
        "cleardesk": [],
        "meeting_id": [],
        "consec_meeting_slots": [],
    }

    emp_id_start_global = 0

    for it in range(iterations):
        for unit, profile in profiles.items():
            num_employees = int(profile["num_employees"])  # type: ignore[index]
            target_rate = float(profile["employment_rate"])  # type: ignore[index]
            office_share = float(profile["office"])  # type: ignore[index]
            meeting_ratio_center = float(profile["meeting"])  # type: ignore[index]

            bgs = draw_bgs(
                rng,
                num_employees=num_employees,
                target_rate=target_rate,
                min_bg=min_bg,
                max_bg=max_bg,
                step_bg=step_bg,
                tolerance=employment_rate_variability,
            )

            week_map = sample_working_weeks_vectorized(
                rng,
                iso_weeks=iso_weeks,
                week_weights=week_weights,
                num_employees=num_employees,
                weeks_not_working=weeks_not_working,
            )
            weekmask_half = week_map[:, cal.slot_weeknums_half]

            n_cats = len(halfdays)
            chosen_cat_mask = build_category_mask_vectorized(
                rng,
                num_employees=num_employees,
                bgs=bgs,
                n_cats=n_cats,
                halfday_probs=halfday_probs,
            )
            catmask_half = chosen_cat_mask[:, cal.slot_cat_idx_half]

            present_half = catmask_half & weekmask_half
            weeks_working = max(0, len(iso_weeks) - weeks_not_working)

            total_slots_unit = int(np.sum(bgs * 10)) * max(weeks_working, 0)
            targets = np.rint(halfday_probs * total_slots_unit * office_share).astype(
                int
            )

            present_sum_per_half = present_half.sum(axis=0).astype(int)
            counts_by_cat = np.bincount(
                cal.slot_cat_idx_half, weights=present_sum_per_half, minlength=n_cats
            ).astype(int)

            for c in range(n_cats):
                extra = counts_by_cat[c] - targets[c]
                if extra <= 0:
                    continue
                half_idxs_c = np.flatnonzero(cal.slot_cat_idx_half == c)
                sub = present_half[:, half_idxs_c]
                emp_rel_idx = np.flatnonzero(sub)
                if emp_rel_idx.size == 0:
                    continue
                n_remove = int(min(extra, emp_rel_idx.size))
                pick = rng.choice(emp_rel_idx.size, size=n_remove, replace=False)
                emp_idx, half_rel = np.unravel_index(emp_rel_idx[pick], sub.shape)
                present_half[emp_idx, half_idxs_c[half_rel]] = False

            E, twoD = present_half.shape
            D = cal.n_days
            assert twoD == 2 * D
            present_slots = (
                present_half.reshape(E, D, 2)[:, :, :, None]
                .repeat(SLOTS_PER_HALFDAY, axis=3)
                .reshape(E, D, SLOTS_PER_DAY)
            ).astype(np.int8)

            meeting = np.zeros_like(present_slots, dtype=np.int8)
            meeting_id_arr = np.full_like(present_slots, -1, dtype=np.int32)
            einzel_ap = present_slots.copy()

            present_half_sum_by_day = present_half.sum(axis=0).reshape(D, 2)
            present_total_per_day = present_half_sum_by_day.sum(axis=1).astype(int)

            for day_idx, dts in enumerate(cal.work_dates):
                present_total = int(present_total_per_day[day_idx])
                if present_total == 0:
                    continue
                meet_ratio = rng.triangular(
                    meeting_ratio_center - 0.03,
                    meeting_ratio_center,
                    meeting_ratio_center + 0.07,
                )
                total_meetingtime = 4.5 * present_total * meet_ratio
                meetings = generate_meetings(
                    rng,
                    total_meetingtime,
                    size_vals,
                    size_probs,
                    dur_vals,
                    dur_probs,
                    start_vals,
                    start_probs,
                    max_draws=100,
                )
                if meetings.size == 0:
                    continue

                next_meeting_id, all_meeting_records = _assign_meetings_numpy(
                    rng,
                    it,
                    present_slots,
                    meeting,
                    meeting_id_arr,
                    cal.times_day,
                    unit,
                    day_idx,
                    dts,
                    next_meeting_id,
                    all_meeting_records,
                    meetings,
                )

            consec = _run_lengths_meeting_segments(meeting)
            cleardesk = ((meeting == 1) & (consec >= min_cleardesk_slots)).astype(
                np.int8
            )
            einzel_ap = (einzel_ap & (1 - cleardesk)).astype(np.int8)

            n_rows_unit = num_employees * D * SLOTS_PER_DAY
            date_idx_flat = np.tile(
                np.repeat(np.arange(D, dtype=np.int32), SLOTS_PER_DAY), num_employees
            )
            time_idx_flat = np.tile(
                np.tile(np.arange(SLOTS_PER_DAY, dtype=np.int16), D), num_employees
            )

            emp_ids = np.arange(
                emp_id_start_global + 1,
                emp_id_start_global + 1 + num_employees,
                dtype=np.int32,
            )
            emp_id_flat = np.repeat(emp_ids, D * SLOTS_PER_DAY)

            present_flat = present_slots.reshape(-1)
            meeting_flat = meeting.reshape(-1)
            einzel_flat = einzel_ap.reshape(-1)
            cleardesk_flat = cleardesk.reshape(-1)
            meeting_id_flat = meeting_id_arr.reshape(-1)
            consec_flat = consec.reshape(-1).astype(np.int16)
            bg_flat = np.repeat(bgs.astype(np.float32), D * SLOTS_PER_DAY)

            final_cols["replication"].append(np.full(n_rows_unit, it, dtype=np.int16))
            final_cols["date_idx"].append(date_idx_flat)
            final_cols["time_idx"].append(time_idx_flat)
            final_cols["unit"].append(
                np.repeat(np.array(unit, dtype=object), n_rows_unit)
            )
            final_cols["employee_id"].append(emp_id_flat)
            final_cols["bg"].append(bg_flat)
            final_cols["present"].append(present_flat.astype(np.int8))
            final_cols["meeting"].append(meeting_flat.astype(np.int8))
            final_cols["einzel_ap"].append(einzel_flat.astype(np.int8))
            final_cols["cleardesk"].append(cleardesk_flat.astype(np.int8))
            final_cols["meeting_id"].append(meeting_id_flat.astype(np.int32))
            final_cols["consec_meeting_slots"].append(consec_flat)

            emp_id_start_global += num_employees

    def _cat(arrs: List[np.ndarray], dtype=None) -> np.ndarray:
        out = np.concatenate(arrs)
        return out.astype(dtype) if dtype is not None else out

    replication = _cat(final_cols["replication"], dtype=np.int16)
    date_idx = _cat(final_cols["date_idx"], dtype=np.int32)
    time_idx = _cat(final_cols["time_idx"], dtype=np.int16)
    unit_col = _cat(final_cols["unit"])
    employee_id = _cat(final_cols["employee_id"], dtype=np.int32)
    bg = _cat(final_cols["bg"], dtype=np.float32)
    present = _cat(final_cols["present"], dtype=np.int8)
    meeting = _cat(final_cols["meeting"], dtype=np.int8)
    einzel_ap = _cat(final_cols["einzel_ap"], dtype=np.int8)
    cleardesk = _cat(final_cols["cleardesk"], dtype=np.int8)
    meeting_id_series = _cat(final_cols["meeting_id"], dtype=np.int32)
    consec_meeting_slots = _cat(final_cols["consec_meeting_slots"], dtype=np.int16)

    dates = cal.work_dates[date_idx]
    week_numbers = cal.iso_weeks_per_day[date_idx].astype(np.int16)
    weekday_codes = cal.weekday_codes_per_day[date_idx]
    time_float = cal.times_day[time_idx].astype(np.float32)
    halfday_codes = np.where(time_idx < SLOTS_PER_HALFDAY, "am", "pm")

    all_data = pd.DataFrame(
        {
            "replication": replication,
            "date": dates,
            "halfday": pd.Categorical(halfday_codes, ordered=False),
            "weekday": pd.Categorical(
                weekday_codes, ordered=True, categories=DAY_CODES
            ),
            "weekNumber": week_numbers,
            "unit": pd.Categorical(unit_col),
            "employee_id": employee_id,
            "bg": bg,
            "present": present,
            "meeting": meeting,
            "einzel_ap": einzel_ap,
            "time_float": time_float,
            "meeting_id": meeting_id_series,
            "cleardesk": cleardesk,
            "consec_meeting_slots": consec_meeting_slots,
        }
    )

    all_meetings = pd.DataFrame(
        all_meeting_records,
        columns=[
            "replication",
            "meeting_id",
            "unit",
            "date",
            "weekNumber",
            "size",
            "duration",
            "person_hours",
            "start_time",
            "end_time",
        ],
    )

    if not all_meetings.empty:
        all_meetings["meeting_room_size"] = compute_meeting_room_size(
            all_meetings, meeting_room_max_size
        )

    return all_data, all_meetings


# -------------------------------------------------
# Plots
# -------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_tagespeak(all_data: pd.DataFrame):
    """Visualisierung wie im Beispielbild: Anzahl APs für verschiedene Quantile."""

    # group = (
    #     all_data.groupby(["replication", "date", "time_float"])["einzel_ap"]
    #     .sum()
    #     .groupby(["replication", "date"])
    # )
    # daily_peaks = group.max()  # Tagespeak je Replication & Datum
    # avg_peaks_per_rep = group.mean()  # Durchschnittlicher Tagespeak je Replication
    group = (
        all_data.groupby(["replication", "weekNumber", "date", "time_float"])[
            "einzel_ap"
        ]
        .sum()
        .groupby(["replication", "weekNumber", "date"])
    )

    daily_peaks = group.max()  # Tagespeak je Replication & Datum
    avg_peaks_per_rep = daily_peaks.groupby(
        ["replication", "weekNumber"]
    ).mean()  # Durchschnittlicher Tagespeak je Replication

    # 3. Quantile (100% .. 5%)
    quantile_levels = np.arange(0.0, 1.0, 0.05) + 0.05
    quantiles = np.quantile(avg_peaks_per_rep, q=quantile_levels)

    # 4. Zusatzlinien
    max_daily_peak = daily_peaks.max()
    max_avg_peak = avg_peaks_per_rep.max()
    mean_avg_peak = avg_peaks_per_rep.median()

    # 5. Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(
        [f"{int(round(q*100,0))}%" for q in quantile_levels],
        quantiles,
        color=["green" if q in (0.85, 0.5) else "#2a5783" for q in quantile_levels],
    )

    for bar, val in zip(bars, quantiles):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 3,
            f"{int(val)}",
            ha="center",
            va="bottom",
        )

    ax.axhline(max_daily_peak, color="orange", linewidth=2, label="Maximaler Tagespeak")
    ax.axhline(
        max_avg_peak,
        color="darkgreen",
        linewidth=2,
        label="Max. durchschnittlicher Tagespeak",
    )
    ax.axhline(
        mean_avg_peak,
        color="green",
        linewidth=2,
        label="Median durchschnittlicher Tagespeak",
    )

    ax.set_ylabel("Anzahl benötigter Einzel-APs")
    ax.set_xlabel("Abdeckungsgrad der Simulationen")
    ax.set_title(
        "Anzahl der benötigten Einzelarbeitsplätze für ein Büro,\n"
        "die in X% der Simulationsdurchläufe den durchschnittlichen Tagespeak abdecken"
    )
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig


def plot_meetingrooms(all_meetingrooms: pd.DataFrame, size: str):
    """Visualisierung wie im Beispielbild: Anzahl Meetingräume (klein/mittel/gross)."""
    df = all_meetingrooms[all_meetingrooms["meeting_room_size"] == size]
    if df.empty:
        print(f"Keine Daten für {size} Meetingräume.")
        return

    # group = (
    #     df.groupby(["replication", "date", "time_float"])["busy"]
    #     .sum()
    #     .groupby(["replication", "date"])
    # )

    # # 1. Tagespeak je Replication & Datum & Slot
    # daily_peaks = group.max()  # Tagespeak je Replication & Datum

    # # 2. Durchschnittlicher Tagespeak pro Simulation
    # avg_peaks_per_rep = group.mean()  # Durchschnittlicher Tagespeak je Replication
    group = (
        df.groupby(["replication", "weekNumber", "date", "time_float"])["busy"]
        .sum()
        .groupby(["replication", "weekNumber", "date"])
    )

    # 1. Tagespeak je Replication & Datum & Slot
    daily_peaks = group.max()  # Tagespeak je Replication & Datum

    # 2. Durchschnittlicher Tagespeak pro Simulation
    avg_peaks_per_rep = daily_peaks.groupby(
        ["replication", "weekNumber"]
    ).mean()  # Durchschnittlicher Tagespeak je Replication

    # 3. Quantile (100% .. 5%) runden auf 2 stellige Zahl
    quantile_levels = np.arange(0.0, 1.0, 0.05) + 0.05
    quantiles = np.quantile(avg_peaks_per_rep, q=quantile_levels)

    # 4. Zusatzlinien
    max_daily_peak = daily_peaks.max()
    max_avg_peak = avg_peaks_per_rep.max()
    mean_avg_peak = avg_peaks_per_rep.mean()

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(
        [f"{int(round(q*100,0))}%" for q in quantile_levels],
        quantiles,
        color=["green" if q in (0.85, 0.5) else "#2a5783" for q in quantile_levels],
    )
    for bar, val in zip(bars, quantiles):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.5,
            f"{int(val)}",
            ha="center",
            va="bottom",
        )

    ax.axhline(max_daily_peak, color="orange", linewidth=2, label="Maximaler Tagespeak")
    ax.axhline(
        max_avg_peak,
        color="darkgreen",
        linewidth=2,
        label="Max. durchschnittlicher Tagespeak",
    )
    ax.axhline(
        mean_avg_peak,
        color="green",
        linewidth=2,
        label="Median durchschnittlicher Tagespeak",
    )

    ax.set_ylabel(f"Anzahl benötigter {size}-Meetingräume")
    ax.set_xlabel("Abdeckungsgrad der Simulationen")
    ax.set_title(
        f"Anzahl der benötigten {size}-Meetingräume für ein Büro,\n"
        f"die in X% der Simulationsdurchläufe den durchschnittlichen Tagespeak abdecken"
    )
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig


# -------------------------------------------------
# Example usage
# -------------------------------------------------
if __name__ == "__main__":
    np.random.seed(42)

    start_date = date(2025, 1, 1)
    end_date = date(2025, 12, 31)

    week_factor = {"mon": 0.175, "tue": 0.245, "wed": 0.23, "thu": 0.23, "fri": 0.12}

    # Build base week weighting for all weeks in range, then scale special weeks
    weeks_in_range = (
        pd.date_range(start=start_date, end=end_date, freq="W-MON")
        .isocalendar()
        .week.unique()
    )
    week_weighting = build_week_weighting_from_weeks(weeks=weeks_in_range, weight=1.0)
    week_weighting = scale_week_weighting(
        week_weighting,
        {
            1: 33,
            2: 33,
            3: 33,
            4: 33,
            5: 37,
            6: 37,
            7: 37,
            8: 37,
            9: 35,
            10: 35,
            11: 35,
            12: 35,
            13: 35,
            14: 33,
            15: 33,
            16: 33,
            17: 33,
            18: 31,
            19: 31,
            20: 31,
            21: 31,
            22: 37,
            23: 37,
            24: 37,
            25: 37,
            26: 37,
            27: 33,
            28: 33,
            29: 33,
            30: 33,
            31: 27,
            32: 27,
            33: 27,
            34: 27,
            35: 27,
            36: 37,
            37: 37,
            38: 37,
            39: 37,
            40: 37,
            41: 37,
            42: 37,
            43: 37,
            44: 39,
            45: 39,
            46: 39,
            47: 39,
            48: 39,
            49: 27,
            50: 27,
            51: 27,
            52: 27,
        },
    )

    # Normalize for readability (run_simulation normalizes again for the slice)
    sw = sum(week_weighting.values())
    if sw > 0:
        week_weighting = {k: v / sw for k, v in week_weighting.items()}

    profiles = {
        "Abteilung_A": {
            "num_employees": 40,
            "employment_rate": 0.8,
            "office": 0.7,
            "meeting": 0.3,
            "not_office": 0.3,
        },
        "Team_B": {
            "num_employees": 30,
            "employment_rate": 0.75,
            "office": 0.6,
            "meeting": 0.25,
            "not_office": 0.4,
        },
        "Funktion_C": {
            "num_employees": 30,
            "employment_rate": 0.85,
            "office": 0.8,
            "meeting": 0.4,
            "not_office": 0.2,
        },
    }

    meeting_size_dist = {
        2: 0.45,
        3: 0.25,
        4: 0.15,
        5: 0.04,
        6: 0.03,
        7: 0.02,
        8: 0.01,
        9: 0.01,
        10: 0.01,
        11: 0.01,
        12: 0.01,
        13: 0.01,
    }
    meeting_duration_dist = {
        0.5: 0.1,
        1.0: 0.6,
        1.5: 0.1,
        2.0: 0.15,
        2.5: 0.01,
        3.0: 0.01,
        3.5: 0.01,
        4.0: 0.02,
    }
    meeting_start_time_dist = {
        8: 0.07,
        8.5: 0.05,
        9: 0.09,
        9.5: 0.05,
        10: 0.11,
        10.5: 0.06,
        11: 0.07,
        11.5: 0.03,
        12: 0.02,
        12.5: 0.01,
        13: 0.08,
        13.5: 0.07,
        14: 0.08,
        14.5: 0.04,
        15: 0.06,
        15.5: 0.03,
        16: 0.05,
        16.5: 0.03,
    }
    meeting_room_max_size = {"klein": 4, "mittel": 10, "gross": 20}

    all_data, all_meetings = run_simulation(
        start_date=start_date,
        end_date=end_date,
        week_factor=week_factor,
        profiles=profiles,
        min_bg=0.4,
        max_bg=1.0,
        step_bg=0.1,
        employment_rate_variability=0.05,
        weeks_not_working=7,
        iterations=20,
        seed=42,
        min_cleardesk_hours=1.5,
        meeting_room_max_size=meeting_room_max_size,
        week_weighting=week_weighting,
        meeting_size_dist=meeting_size_dist,
        meeting_duration_dist=meeting_duration_dist,
        meeting_start_time_dist=meeting_start_time_dist,
    )

    all_meetingrooms = build_room_occupancy_slots(
        all_meetings,
        slot_times=TIMES_DAY,
        by=("replication", "weekNumber", "date", "meeting_room_size"),
        room_col="room_id",
        include_idle=True,
    )

    plot_tagespeak(all_data).show()
    plot_meetingrooms(all_meetingrooms, "klein").show()
    plot_meetingrooms(all_meetingrooms, "mittel").show()
    plot_meetingrooms(all_meetingrooms, "gross")

    print("\nBG pro Einheit:")
    bg_rep_mean = all_data.groupby(["replication", "unit"], observed=False)["bg"].mean()
    bg_mean = bg_rep_mean.groupby("unit", observed=False).mean()
    bg_std = bg_rep_mean.groupby("unit", observed=False).std()
    for unit, profile in profiles.items():
        target_rate = float(profile["employment_rate"])
        print(
            f"{unit}: {bg_mean[unit]:.3f} (+/-{bg_std[unit]:.3f}) (target: {target_rate:.3f})"
        )

    print("\nAnteil Office pro Einheit:")
    office_sum = all_data.groupby(["replication", "unit"], observed=False)[
        "present"
    ].sum()
    office_share_mean = office_sum.groupby("unit", observed=False).mean()
    office_share_std = office_sum.groupby("unit", observed=False).std()
    for unit, profile in profiles.items():
        num_employees = int(profile["num_employees"])
        total_slots = (
            num_employees * 5 * 45 * SLOTS_PER_DAY * (profile["employment_rate"])
        )  # approx
        share = office_share_mean[unit] / total_slots if total_slots > 0 else 0.0
        std = office_share_std[unit] / total_slots if total_slots > 0 else 0.0
        print(f"{unit}: {share:.3f} (+/-{std:.3f}) (target: {profile['office']:.3f})")

    print("\nMeeting Rate pro Einheit:")
    meeting_sum = all_data.groupby(["replication", "unit"], observed=False)[
        "meeting"
    ].sum()
    meeting_sum_mean = meeting_sum.groupby("unit", observed=False).mean()
    meeting_sum_std = meeting_sum.groupby("unit", observed=False).std()
    for unit, profile in profiles.items():
        denom = office_share_mean[unit] if office_share_mean[unit] > 0 else 1.0
        share_mean = meeting_sum_mean[unit] / denom
        share_std = meeting_sum_std[unit] / denom
        print(
            f"{unit}: {share_mean:.3f} (+/-{share_std:.3f}) (target: {profile['meeting']:.3f})"
        )

    print("\nWochentagsverteilung Präsenz (in %):")
    present_by_dow = all_data.groupby("weekday", observed=False)["present"].sum()
    total_present = present_by_dow.sum()
    present_pct = (present_by_dow / total_present * 100).round(2)
    for d in DAY_CODES:
        target_pct = week_factor[d] * 100
        actual_pct = present_pct.get(d, 0.0)
        print(f"{d}: {actual_pct:.2f}% (target: {target_pct:.2f}%)")

    print("\nVerteilung auf Wochen (in %):")
    present_by_week = all_data.groupby("weekNumber", observed=False)["present"].sum()
    total_present = present_by_week.sum()
    present_pct = (present_by_week / total_present * 100).round(2)
    for week, weight in sorted(week_weighting.items()):
        w = int(week)
        target_pct = weight / sum(week_weighting.values()) * 100
        actual_pct = present_pct.get(w, 0.0)
        print(f"Woche {w}: {actual_pct:.2f}% (target: {target_pct:.2f}%)")

    # input("\nFertig. Enter zum Beenden...")
