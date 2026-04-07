#!/usr/bin/env python3
"""
preprocess_h1.py
================
Stack Exchange H1 feature extraction pipeline.

PURPOSE
-------
Produces a per-user panel dataset (h1_panel.parquet) for H1 analysis of the
impact of ChatGPT on knowledge dynamics across Stack Exchange communities.

Intended for use on BOTH:
  - Mathematics Stack Exchange  (high domain complexity)
  - Stack Overflow               (lower domain complexity)

Run once per community by pointing DATA_DIR / OUT_DIR at the correct dump.

STUDY DESIGN
------------
Treatment event: ChatGPT 3.5 release — 2022-11-30

Sampling frame (who is in scope):
  Users active during the PRE-TREATMENT WINDOW via any contribution
  (answer, question, or comment).  Post-treatment columns are computed for
  this same cohort regardless of whether a user was active post-treatment
  (inactive users receive zeros for post-treatment columns).

  PRE_START   = 2022-04-01
  PRE_END     = 2022-09-30
  POST_START  = 2023-04-01
  POST_END    = 2023-09-30

  Both windows are 6 months and symmetric around the treatment event,
  separated by a 6-month buffer (Oct 2022 – Mar 2023) that contains the
  treatment itself and allows an adaptation period.

SCORE FEATURES
--------------
Score columns in Posts.xml are cumulative dump-date totals and cannot be
used directly.  Pre- and post-treatment scores are reconstructed from
Votes.xml (VoteTypeId 2 = upvote, 3 = downvote) by filtering to votes cast
within each window.  This is the same approach used in patch_score_features.py.

  mean net score = (upvotes - downvotes) averaged over the user's posts
                   in the respective window.

Note: comment scores are NOT recoverable (PostFeedback table not in dump).

OUTPUT SCHEMA (h1_panel.parquet)
---------------------------------
All values are raw, unscaled, untransformed counts/rates.

  user_id                    int   — Stack Exchange user ID
  pre_num_questions          int   — questions posted in pre-treatment window
  pre_num_answers            int   — answers posted in pre-treatment window
  pre_score_answers          float — total net score of answers in pre window
  pre_score_questions        float — total net score of questions in pre window
  post_score_answers         float — total net score of answers in post window
  post_score_questions       float — total net score of questions in post window
  post_num_questions         int   — questions posted in post-treatment window
  post_num_answers           int   — answers posted in post-treatment window
  pre_aar                    float — answer acceptance rate in pre window
                                     (accepted_answers / total_answers; 0 if no answers)
  post_aar                   float — answer acceptance rate in post window
  num_badges                 int   — total badges earned up to PRE_END
  total_tenure                    float — days from account creation to CHATGPT_RELEASE
  pre_weekly_activity_regularity  float — unique active weeks / career weeks in pre window
                                          (capped at 1.0)
  post_weekly_activity_regularity float — unique active weeks / career weeks in post window
                                          (capped at 1.0)

DATA SOURCE
-----------
Stack Exchange Data Dump — September 2025 (data through 2025-09-30).
Archive: https://archive.org/details/stackexchange
Files used:
  Users.xml    — account creation date, for tenure computation
  Posts.xml    — questions (PostTypeId=1) and answers (PostTypeId=2)
  Badges.xml   — badge count up to PRE_END
  Votes.xml    — upvotes/downvotes (VoteTypeId 2/3) for score reconstruction

REPRODUCIBILITY
---------------
Python >= 3.10 required.
Dependencies: polars, pandas, numpy, pyarrow, lxml (optional).
No stochastic operations.  Output is fully deterministic given identical input.

REFERENCES
----------
Burtch, G., Lee, D., & Chen, Z. (2024). The consequences of generative AI
  for online knowledge communities. Scientific Reports, 14(1), 10413.
Li, X., & Kim, K. (2024). Impacts of generative AI on user contributions:
  evidence from a coding Q&A platform. Marketing Letters, 36(3), 577-591.
Zeng, S. (2025). The Impact of GenAI on Knowledge Dynamics in Online
  Communities: Domain Complexity and Expert Retention.
  Bachelor Thesis, University of Zurich.
"""

import gc, json, time, warnings, tempfile
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

try:
    from lxml import etree as ET
    LXML = True
except ImportError:
    import xml.etree.ElementTree as ET
    LXML = False

warnings.filterwarnings('ignore')
t_global = time.time()


# ============================================================
# SET YOUR PATHS HERE BEFORE RUNNING
# ============================================================
# Point these at one community at a time.
# Example paths shown for Stack Overflow; adjust for Math SE.
#DATA_DIR = Path(r'E:\Stack Exchange Data Dump\stackoverflow.com')
#OUT_DIR  = Path(r'E:\Stack Exchange Data Dump\stackoverflow.com\output_h1')

# HPC Pathing
DATA_DIR = Path('/scratch/szeng/stackoverflow.com')
OUT_DIR  = Path('/scratch/szeng/stackoverflow_output_h1')
# ============================================================


# ── Treatment anchor and study windows ───────────────────────────────────────
CHATGPT_RELEASE = pd.Timestamp('2022-11-30', tz='UTC')

PRE_START  = pd.Timestamp('2022-04-01', tz='UTC')
PRE_END    = pd.Timestamp('2022-09-30', tz='UTC')

POST_START = pd.Timestamp('2023-04-01', tz='UTC')
POST_END   = pd.Timestamp('2023-09-30', tz='UTC')

# ── Parser memory configuration ──────────────────────────────────────────────
# Increase POSTS_CHUNK_SIZE for Stack Overflow (large Posts.xml).
# Increase VOTES_CHUNK_SIZE for the much larger Votes.xml on Stack Overflow.
POSTS_CHUNK_SIZE  = 10_000
VOTES_CHUNK_SIZE  = 10_000


# =============================================================================
# XML PARSING INFRASTRUCTURE
# (carried over from preprocess.py with minor renaming for clarity)
# =============================================================================

def parse_xml(filepath, wanted_cols, date_cols=None,
              filter_col=None, ws=None, we=None, posttypes=None,
              chunk_size=None):
    """Stream-parse a Stack Exchange XML dump file into a temporary Parquet file.

    Parameters
    ----------
    filepath    : Path
    wanted_cols : list[str]
    date_cols   : list[str], optional
    filter_col  : str, optional — date column to filter on; None = no filter
    ws, we      : pd.Timestamp, optional — window start / end (UTC-aware)
    posttypes   : set[int], optional — keep only these PostTypeId values
    chunk_size  : int, optional — rows per flush; defaults to POSTS_CHUNK_SIZE

    Returns
    -------
    Path or None — temp Parquet file path, or None if no rows matched.
    """
    if chunk_size is None:
        chunk_size = POSTS_CHUNK_SIZE

    date_cols = date_cols or []
    buf = []
    rows_written = 0

    tmp = tempfile.NamedTemporaryFile(suffix='.parquet', delete=False, dir=OUT_DIR)
    tmp_path = Path(tmp.name)
    tmp.close()
    writer = None

    # Fixed PyArrow schema prevents null-type inference errors across chunks.
    pa_fields = []
    for col in wanted_cols:
        if col in date_cols:
            pa_fields.append(pa.field(col, pa.timestamp('ns', tz='UTC')))
        elif col == 'PostTypeId':
            pa_fields.append(pa.field(col, pa.int64()))
        else:
            pa_fields.append(pa.field(col, pa.string()))
    fixed_schema = pa.schema(pa_fields)

    def _flush(buf):
        nonlocal writer, rows_written
        chunk = _process_chunk(buf, date_cols, filter_col, ws, we, posttypes)
        if chunk is None:
            return
        table = pa.Table.from_pandas(chunk, schema=fixed_schema,
                                     preserve_index=False, safe=False)
        if writer is None:
            writer = pq.ParquetWriter(tmp_path, fixed_schema, compression='snappy')
        writer.write_table(table)
        rows_written += len(chunk)
        del chunk, table
        gc.collect()

    for _, elem in ET.iterparse(str(filepath), events=('end',)):
        if elem.tag != 'row':
            elem.clear()
            continue
        buf.append({c: elem.attrib.get(c) for c in wanted_cols})
        elem.clear()

        if len(buf) >= chunk_size:
            _flush(buf)
            buf = []

    if buf:
        _flush(buf)

    if writer is not None:
        writer.close()

    if rows_written == 0:
        tmp_path.unlink(missing_ok=True)
        return None

    return tmp_path


def _process_chunk(buf, date_cols, filter_col, ws, we, posttypes):
    """Convert a raw attribute-dict buffer into a filtered DataFrame chunk."""
    df = pd.DataFrame(buf)
    buf.clear()

    for dc in date_cols:
        if dc in df.columns:
            df[dc] = pd.to_datetime(df[dc], errors='coerce', utc=True)

    if filter_col and ws is not None:
        df = df[df[filter_col].between(ws, we)]
        if df.empty:
            return None

    if posttypes:
        df['PostTypeId'] = pd.to_numeric(df['PostTypeId'], errors='coerce')
        df = df[df['PostTypeId'].isin(posttypes)]

    return df if not df.empty else None


def _read_parquet_filtered(tmp_path, active_ids, id_col, numeric_cols,
                            owner_col=None):
    """Read a temp Parquet file, keeping only active-user rows.

    Reads row-group by row-group (peak RAM = one row-group at a time).
    Explicitly closes the ParquetFile before unlinking to avoid Windows
    file-locking errors (WinError 32).
    """
    pf = pq.ParquetFile(tmp_path)
    kept = []
    active_ids_set = set(active_ids)

    try:
        for batch in pf.iter_batches():
            tbl = pa.Table.from_batches([batch])
            df_chunk = tbl.to_pandas()
            del tbl, batch

            df_chunk[id_col] = pd.to_numeric(df_chunk[id_col], errors='coerce')
            df_chunk = df_chunk[df_chunk[id_col].isin(active_ids_set)]

            if not df_chunk.empty:
                kept.append(df_chunk)
            else:
                del df_chunk
            gc.collect()
    finally:
        pf.close()

    tmp_path.unlink(missing_ok=True)

    if not kept:
        return pd.DataFrame()

    result = pd.concat(kept, ignore_index=True)
    del kept; gc.collect()

    for c in numeric_cols:
        if c in result.columns:
            result[c] = pd.to_numeric(result[c], errors='coerce')

    if owner_col and owner_col in result.columns:
        result.dropna(subset=[owner_col], inplace=True)
        result[owner_col] = result[owner_col].astype(int)

    return result


# =============================================================================
# FILE LOADERS
# =============================================================================

def load_users():
    """Load all user records from Users.xml.

    total_tenure is anchored to CHATGPT_RELEASE (days since account creation).

    Returns
    -------
    pd.DataFrame
        Index: UserId (int).
        Columns: CreationDate (UTC datetime), total_tenure (float, days).
    """
    print('  [Users] parsing Users.xml...', flush=True)
    t0 = time.time()
    cols = ['Id', 'CreationDate']

    tmp_path = parse_xml(DATA_DIR / 'Users.xml', cols,
                         date_cols=['CreationDate'])
    if tmp_path is None:
        raise RuntimeError('Users.xml produced no rows — check DATA_DIR path')

    df = pd.read_parquet(tmp_path)
    tmp_path.unlink(missing_ok=True)

    df['Id'] = pd.to_numeric(df['Id'], errors='coerce')
    df.dropna(subset=['Id'], inplace=True)
    df['Id'] = df['Id'].astype(int)
    df.set_index('Id', inplace=True)

    df['total_tenure'] = (
        CHATGPT_RELEASE - df['CreationDate']
    ).dt.total_seconds().div(86400).clip(lower=0)

    print(f'  [Users]    {len(df):,} total registered users  ({time.time()-t0:.1f}s)',
          flush=True)
    return df


def load_posts_window(ws, we, label):
    """Stream Posts.xml for a given time window.

    Loads questions and answers with CreationDate in [ws, we].
    Columns returned: Id, PostTypeId, OwnerUserId, CreationDate,
                      AcceptedAnswerId, ParentId.

    Parameters
    ----------
    ws, we : pd.Timestamp — window start and end (UTC-aware)
    label  : str — descriptive label for progress output (e.g. 'pre', 'post')

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (questions, answers) filtered to the window, all registered users.
    """
    print(f'  [Posts/{label}] parsing Posts.xml '
          f'({ws.date()} -> {we.date()})...', flush=True)
    t0 = time.time()
    cols = ['Id', 'PostTypeId', 'OwnerUserId', 'CreationDate',
            'AcceptedAnswerId', 'ParentId']

    tmp_path = parse_xml(DATA_DIR / 'Posts.xml', cols,
                         date_cols=['CreationDate'],
                         filter_col='CreationDate',
                         ws=ws, we=we,
                         posttypes={1, 2})

    if tmp_path is None:
        print(f'  [Posts/{label}] WARNING: no posts found in window', flush=True)
        empty = pd.DataFrame(columns=cols)
        return empty, empty

    numeric_cols = ['Id', 'PostTypeId', 'OwnerUserId', 'AcceptedAnswerId', 'ParentId']
    # No active-user filter here — we want ALL users in the window to build
    # the sampling frame.  Filtering to active_user_ids happens in main().
    pf = pq.ParquetFile(tmp_path)
    kept = []
    try:
        for batch in pf.iter_batches():
            tbl = pa.Table.from_batches([batch])
            df_chunk = tbl.to_pandas()
            del tbl, batch
            for c in numeric_cols:
                if c in df_chunk.columns:
                    df_chunk[c] = pd.to_numeric(df_chunk[c], errors='coerce')
            df_chunk.dropna(subset=['OwnerUserId'], inplace=True)
            df_chunk['OwnerUserId'] = df_chunk['OwnerUserId'].astype(int)
            if not df_chunk.empty:
                kept.append(df_chunk)
            gc.collect()
    finally:
        pf.close()
    tmp_path.unlink(missing_ok=True)

    if not kept:
        empty = pd.DataFrame(columns=cols)
        return empty, empty

    df = pd.concat(kept, ignore_index=True)
    del kept; gc.collect()

    questions = df[df['PostTypeId'] == 1].reset_index(drop=True)
    answers   = df[df['PostTypeId'] == 2].reset_index(drop=True)
    del df; gc.collect()

    print(f'  [Posts/{label}]  Q={len(questions):,}  A={len(answers):,}'
          f'  ({time.time()-t0:.1f}s)', flush=True)
    return questions, answers


def load_badges_pre(active_user_ids):
    """Stream Badges.xml and return all badges earned up to PRE_END.

    Badge count is a simple total (all badge types combined) as a raw integer.
    The upper bound is PRE_END so the count reflects the pre-treatment state
    of the user, comparable across the cohort.

    Parameters
    ----------
    active_user_ids : set[int]

    Returns
    -------
    pd.Series
        Index: UserId (int), values: num_badges (int).
    """
    print(f'  [Badges] parsing Badges.xml (Date <= {PRE_END.date()})...', flush=True)
    t0 = time.time()
    cols = ['UserId', 'Date']

    tmp_path = parse_xml(DATA_DIR / 'Badges.xml', cols,
                         date_cols=['Date'],
                         filter_col='Date',
                         ws=pd.Timestamp('2000-01-01', tz='UTC'),
                         we=PRE_END)

    if tmp_path is None:
        print('  [Badges] WARNING: no badges found', flush=True)
        return pd.Series(dtype=int, name='num_badges')

    df = _read_parquet_filtered(tmp_path, active_user_ids,
                                id_col='UserId',
                                numeric_cols=['UserId'],
                                owner_col='UserId')

    badge_counts = df.groupby('UserId').size().rename('num_badges')
    print(f'  [Badges]  {len(df):,} badge rows for {len(badge_counts):,} users'
          f'  ({time.time()-t0:.1f}s)', flush=True)
    return badge_counts


def compute_post_scores_window(ws, we, label):
    """Stream Votes.xml and compute per-post net scores within a time window.

    Only VoteTypeId 2 (upvote) and 3 (downvote) are used.
    Only votes with CreationDate in [ws, we] are counted.

    This reconstructs an uncontaminated windowed net score without relying
    on the cumulative Posts.xml Score column.

    Parameters
    ----------
    ws, we : pd.Timestamp — window start and end (UTC-aware)
    label  : str — 'pre' or 'post'

    Returns
    -------
    pd.DataFrame
        Columns: PostId (int), net_score (int).
    """
    votes_path = DATA_DIR / 'Votes.xml'
    print(f'  [Votes/{label}] streaming Votes.xml '
          f'({ws.date()} -> {we.date()})...', flush=True)
    t0 = time.time()

    # String-compare bounds (ISO 8601 lexicographic order == chronological)
    ws_str = ws.strftime('%Y-%m-%dT%H:%M:%S')
    we_str = we.strftime('%Y-%m-%dT%H:%M:%S')

    post_counts: dict[int, list[int]] = {}
    rows_read = rows_kept = 0

    for _, elem in ET.iterparse(str(votes_path), events=('end',)):
        if elem.tag != 'row':
            elem.clear()
            continue

        rows_read += 1
        vtype = elem.attrib.get('VoteTypeId')
        if vtype not in ('2', '3'):
            elem.clear()
            continue

        created = elem.attrib.get('CreationDate', '')
        if created < ws_str or created > we_str:
            elem.clear()
            continue

        pid = elem.attrib.get('PostId')
        if pid is None:
            elem.clear()
            continue

        pid_int = int(pid)
        if pid_int not in post_counts:
            post_counts[pid_int] = [0, 0]

        if vtype == '2':
            post_counts[pid_int][0] += 1
        else:
            post_counts[pid_int][1] += 1

        rows_kept += 1
        elem.clear()

        if rows_read % 1_000_000 == 0:
            print(f'    ...{rows_read:,} rows read  {rows_kept:,} kept  '
                  f'{len(post_counts):,} unique PostIds  '
                  f'{time.time()-t0:.0f}s', flush=True)

    print(f'  [Votes/{label}] {rows_read:,} rows read, {rows_kept:,} kept, '
          f'{len(post_counts):,} unique PostIds  ({time.time()-t0:.1f}s)', flush=True)

    post_ids = list(post_counts.keys())
    ups   = [post_counts[p][0] for p in post_ids]
    downs = [post_counts[p][1] for p in post_ids]
    df = pd.DataFrame({
        'PostId':     post_ids,
        'net_score':  [u - d for u, d in zip(ups, downs)],
    })
    del post_counts; gc.collect()
    return df


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def compute_window_features(questions, answers, scores_df,
                             active_user_ids, label):
    """Aggregate per-user features for a single time window.

    Parameters
    ----------
    questions      : pd.DataFrame — questions in window (all users, not filtered)
    answers        : pd.DataFrame — answers in window (all users, not filtered)
    scores_df      : pd.DataFrame — per-PostId net scores (from Votes.xml)
    active_user_ids: set[int] — cohort (defines output rows)
    label          : str — 'pre' or 'post' (for column naming)

    Returns
    -------
    pd.DataFrame
        Indexed by UserId.  Columns prefixed with label:
          {label}_num_questions, {label}_num_answers,
          {label}_score_answers, {label}_score_questions,
          {label}_aar
        Plus for pre only: weekly_activity_regularity
    """
    print(f'  [Features/{label}] engineering window features...', flush=True)
    t0 = time.time()

    # Restrict to cohort users
    q_coh = questions[questions['OwnerUserId'].isin(active_user_ids)].copy()
    a_coh = answers[answers['OwnerUserId'].isin(active_user_ids)].copy()

    all_users = pd.DataFrame({'UserId': sorted(active_user_ids)})

    # ── Post counts ───────────────────────────────────────────────────────────
    q_counts = (q_coh.groupby('OwnerUserId').size()
                .rename(f'{label}_num_questions').reset_index()
                .rename(columns={'OwnerUserId': 'UserId'}))
    a_counts = (a_coh.groupby('OwnerUserId').size()
                .rename(f'{label}_num_answers').reset_index()
                .rename(columns={'OwnerUserId': 'UserId'}))

    feat = (all_users
            .merge(q_counts, on='UserId', how='left')
            .merge(a_counts, on='UserId', how='left'))
    feat[f'{label}_num_questions'] = feat[f'{label}_num_questions'].fillna(0).astype(int)
    feat[f'{label}_num_answers']   = feat[f'{label}_num_answers'].fillna(0).astype(int)

    # ── Answer acceptance rate ────────────────────────────────────────────────
    # AcceptedAnswerId on a question row identifies the accepted answer's Post Id.
    accepted_ids = set(
        q_coh['AcceptedAnswerId'].dropna().astype(int).tolist()
    )
    a_coh['is_accepted'] = a_coh['Id'].isin(accepted_ids).astype(int)

    aar_agg = (a_coh.groupby('OwnerUserId')
               .agg(total_ans=('Id', 'count'),
                    accepted_ans=('is_accepted', 'sum'))
               .reset_index()
               .rename(columns={'OwnerUserId': 'UserId'}))
    aar_agg[f'{label}_aar'] = (
        aar_agg['accepted_ans'] / aar_agg['total_ans'].clip(lower=1)
    )
    feat = feat.merge(aar_agg[['UserId', f'{label}_aar']], on='UserId', how='left')
    feat[f'{label}_aar'] = feat[f'{label}_aar'].fillna(0.0)

    # ── Score features (from Votes.xml reconstruction) ───────────────────────
    # Join per-post net scores onto answers and questions, then aggregate.
    # Posts that received no votes get net_score = 0.

    # Answer scores
    a_scored = a_coh[['Id', 'OwnerUserId']].merge(
        scores_df.rename(columns={'PostId': 'Id'}), on='Id', how='left'
    )
    a_scored['net_score'] = a_scored['net_score'].fillna(0).astype(float)

    ans_score_agg = (a_scored.groupby('OwnerUserId')['net_score']
                     .sum()
                     .rename(f'{label}_score_answers')
                     .reset_index()
                     .rename(columns={'OwnerUserId': 'UserId'}))
    feat = feat.merge(ans_score_agg, on='UserId', how='left')
    feat[f'{label}_score_answers'] = feat[f'{label}_score_answers'].fillna(0.0)

    # Question scores
    q_scored = q_coh[['Id', 'OwnerUserId']].merge(
        scores_df.rename(columns={'PostId': 'Id'}), on='Id', how='left'
    )
    q_scored['net_score'] = q_scored['net_score'].fillna(0).astype(float)

    q_score_agg = (q_scored.groupby('OwnerUserId')['net_score']
                   .sum()
                   .rename(f'{label}_score_questions')
                   .reset_index()
                   .rename(columns={'OwnerUserId': 'UserId'}))
    feat = feat.merge(q_score_agg, on='UserId', how='left')
    feat[f'{label}_score_questions'] = feat[f'{label}_score_questions'].fillna(0.0)

    # ── Weekly activity regularity ────────────────────────────────────────────
    # Defined as: unique active weeks / career weeks within the window,
    # capped at 1.0.  "Active" = any answer or question posted.
    # Computed for both pre and post windows.
    ev = pd.concat([
        a_coh[['OwnerUserId', 'CreationDate']].rename(
            columns={'OwnerUserId': 'UserId'}),
        q_coh[['OwnerUserId', 'CreationDate']].rename(
            columns={'OwnerUserId': 'UserId'}),
    ], ignore_index=True).dropna(subset=['CreationDate'])

    ev['week_str'] = (
        ev['CreationDate'].dt.isocalendar()
                          .apply(lambda r: f'{r.year}-{r.week:02d}', axis=1)
    )

    week_agg = (ev.groupby('UserId')
                .agg(unique_weeks=('week_str', 'nunique'),
                     first_ev=('CreationDate', 'min'),
                     last_ev=('CreationDate', 'max'))
                .reset_index())

    span_days = (
        (week_agg['last_ev'] - week_agg['first_ev'])
        .dt.total_seconds().div(86400).clip(lower=0)
    )
    career_weeks = np.maximum(span_days.values / 7.0, 1.0)
    week_agg[f'{label}_weekly_activity_regularity'] = np.clip(
        week_agg['unique_weeks'].values / career_weeks, 0, 1
    )

    feat = feat.merge(
        week_agg[['UserId', f'{label}_weekly_activity_regularity']],
        on='UserId', how='left'
    )
    feat[f'{label}_weekly_activity_regularity'] = (
        feat[f'{label}_weekly_activity_regularity'].fillna(0.0)
    )

    feat = feat.set_index('UserId')
    print(f'  [Features/{label}] done  shape={feat.shape}  '
          f'({time.time()-t0:.1f}s)', flush=True)
    return feat


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    """Execute the H1 panel construction pipeline.

    Phase 1 — Sampling frame:
      Identify users active during the pre-treatment window (any post or comment).

    Phase 2 — Pre-treatment features:
      Compute counts, scores, AAR, regularity over PRE_START -> PRE_END.

    Phase 3 — Post-treatment features:
      Compute counts, scores, AAR over POST_START -> POST_END for the same cohort.

    Phase 4 — Cross-window fields:
      Badges (up to PRE_END), tenure (up to CHATGPT_RELEASE).

    Phase 5 — Export.
    """
    print('=' * 65)
    print('Stack Exchange H1 Feature Extraction')
    print('Pre/Post Treatment Panel  (Zeng, 2025)')
    print('=' * 65)
    print(f'lxml             : {LXML}')
    print(f'polars           : {pl.__version__}')
    print(f'ChatGPT release  : {CHATGPT_RELEASE.date()}')
    print(f'Pre  window      : {PRE_START.date()} -> {PRE_END.date()}')
    print(f'Post window      : {POST_START.date()} -> {POST_END.date()}')
    print(f'DATA_DIR         : {DATA_DIR}')
    print(f'OUT_DIR          : {OUT_DIR}')
    print()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Validate required input files up front
    required_files = ['Users.xml', 'Posts.xml', 'Badges.xml', 'Votes.xml']
    for fname in required_files:
        fpath = DATA_DIR / fname
        if not fpath.exists():
            raise FileNotFoundError(
                f'Required file not found: {fpath}\n'
                f'Check DATA_DIR and ensure the dump is extracted.'
            )
        size_gb = fpath.stat().st_size / 1e9
        print(f'  {fname:15s}: {size_gb:.2f} GB')
    print()

    # ── Phase 1: Sampling frame ───────────────────────────────────────────────
    print('─' * 65)
    print('PHASE 1 — Sampling frame: users active in pre-treatment window')
    print('─' * 65)

    questions_pre, answers_pre = load_posts_window(PRE_START, PRE_END, 'pre')

    active_user_ids = (
        set(answers_pre['OwnerUserId'].unique().tolist())
        | set(questions_pre['OwnerUserId'].unique().tolist())
    )
    print(f'\n  Active-in-window users : {len(active_user_ids):,}')
    print(f'    via answers          : '
          f'{answers_pre["OwnerUserId"].nunique():,}')
    print(f'    via questions        : '
          f'{questions_pre["OwnerUserId"].nunique():,}')
    print()

    # ── Phase 2: Pre-treatment Votes ──────────────────────────────────────────
    print('─' * 65)
    print('PHASE 2 — Reconstruct pre-treatment post scores from Votes.xml')
    print('─' * 65)

    scores_pre = compute_post_scores_window(PRE_START, PRE_END, 'pre')
    print()

    # ── Phase 3: Pre-treatment features ───────────────────────────────────────
    print('─' * 65)
    print('PHASE 3 — Pre-treatment features')
    print('─' * 65)

    feat_pre = compute_window_features(
        questions_pre, answers_pre, scores_pre,
        active_user_ids, 'pre'
    )
    del questions_pre, answers_pre, scores_pre; gc.collect()
    print()

    # ── Phase 4: Post-treatment data ──────────────────────────────────────────
    print('─' * 65)
    print('PHASE 4 — Post-treatment data')
    print('─' * 65)

    questions_post, answers_post = load_posts_window(POST_START, POST_END, 'post')
    scores_post = compute_post_scores_window(POST_START, POST_END, 'post')

    feat_post = compute_window_features(
        questions_post, answers_post, scores_post,
        active_user_ids, 'post'
    )
    del questions_post, answers_post, scores_post; gc.collect()
    print()

    # ── Phase 5: Cross-window fields ──────────────────────────────────────────
    print('─' * 65)
    print('PHASE 5 — Cross-window fields (badges, tenure)')
    print('─' * 65)

    users      = load_users()
    badge_counts = load_badges_pre(active_user_ids)
    print()

    # ── Assemble panel ────────────────────────────────────────────────────────
    print('─' * 65)
    print('ASSEMBLING PANEL')
    print('─' * 65)

    panel = feat_pre.join(feat_post, how='left')

    # Tenure from Users.xml — restrict to cohort
    tenure_ser = (
        users.loc[users.index.isin(active_user_ids), 'total_tenure']
        .rename('total_tenure')
    )
    panel = panel.join(tenure_ser, how='left')
    panel['total_tenure'] = panel['total_tenure'].fillna(0.0)

    # Badge count — users not in badge_counts had 0 badges
    panel = panel.join(badge_counts, how='left')
    panel['num_badges'] = panel['num_badges'].fillna(0).astype(int)

    # Fill any remaining NaN for post columns (users not active post-treatment)
    post_cols = [c for c in panel.columns if c.startswith('post_')]
    panel[post_cols] = panel[post_cols].fillna(0)
    for c in post_cols:
        if 'num' in c:
            panel[c] = panel[c].astype(int)

    # Reset index to expose user_id as a column
    panel.index.name = 'user_id'
    panel = panel.reset_index()

    # Enforce canonical column order
    col_order = [
        'user_id',
        'pre_num_questions', 'pre_num_answers',
        'pre_score_answers', 'pre_score_questions',
        'post_score_answers', 'post_score_questions',
        'post_num_questions', 'post_num_answers',
        'pre_aar', 'post_aar',
        'pre_weekly_activity_regularity', 'post_weekly_activity_regularity',
        'num_badges', 'total_tenure',
    ]
    # Only keep columns that were produced (safety guard)
    col_order = [c for c in col_order if c in panel.columns]
    panel = panel[col_order]

    print(f'  Panel shape: {panel.shape[0]:,} users × {panel.shape[1]} columns')
    print()

    # ── Export ────────────────────────────────────────────────────────────────
    print('─' * 65)
    print('EXPORT')
    print('─' * 65)

    parquet_path = OUT_DIR / 'h1_panel.parquet'
    panel.to_parquet(parquet_path, engine='pyarrow', compression='snappy', index=False)
    print(f'  h1_panel.parquet  -> {parquet_path}')
    print(f'  File size         :  {parquet_path.stat().st_size / 1e6:.1f} MB')

    out_path = OUT_DIR / 'h1_panel.csv'
    panel.to_csv(out_path, index=False)
    print(f'  h1_panel.csv      -> {out_path}')
    print(f'  File size         :  {out_path.stat().st_size / 1e6:.1f} MB')

    # Descriptive stats
    desc_path = OUT_DIR / 'h1_panel_stats.csv'
    panel.describe().round(4).to_csv(desc_path)
    print(f'  h1_panel_stats.csv -> {desc_path}')

    # Zero rates
    zero_rates = (panel.set_index('user_id') == 0).mean().sort_values(ascending=False)
    zero_path  = OUT_DIR / 'h1_zero_rates.csv'
    zero_rates.to_csv(zero_path, header=['zero_rate'])
    print(f'  h1_zero_rates.csv  -> {zero_path}')

    high_zero = zero_rates[zero_rates > 0.80]
    if len(high_zero):
        print('\n  Columns with >80% zero values:')
        for fname, rate in high_zero.items():
            print(f'    {fname:<35s} {rate:.1%}')

    # Metadata
    meta = {
        'pipeline_version'   : '1.0',
        'study'              : (
            'Zeng (2025) — The Impact of GenAI on Knowledge Dynamics '
            'in Online Communities: Domain Complexity and Expert Retention. '
            'Bachelor Thesis, University of Zurich.'
        ),
        'analysis'           : 'H1 — Math SE vs Stack Overflow panel',
        'chatgpt_release'    : str(CHATGPT_RELEASE.date()),
        'pre_window'         : f'{PRE_START.date()} / {PRE_END.date()}',
        'post_window'        : f'{POST_START.date()} / {POST_END.date()}',
        'sampling_frame'     : 'users active (any post) in pre-treatment window',
        'score_method'       : (
            'Per-post net score reconstructed from Votes.xml '
            'VoteTypeId IN (2,3) filtered to each window. '
            'User score = SUM of net scores over all posts in window. '
            'Posts.xml Score column NOT used (cumulative dump-date total). '
            'Comment scores NOT recoverable (PostFeedback not in public dump).'
        ),
        'badge_upper_bound'  : str(PRE_END.date()),
        'tenure_anchor'      : str(CHATGPT_RELEASE.date()),
        'n_users'            : int(len(panel)),
        'n_columns'          : int(panel.shape[1]),
        'columns'            : panel.columns.tolist(),
        'zero_rates'         : zero_rates.round(4).to_dict(),
        'data_dir'           : str(DATA_DIR),
        'lxml'               : LXML,
        'polars_version'     : pl.__version__,
        'total_time_s'       : round(time.time() - t_global, 1),
    }
    meta_path = OUT_DIR / 'h1_panel_meta.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'  h1_panel_meta.json -> {meta_path}')

    print()
    print('=' * 65)
    print(f'DONE  —  {meta["total_time_s"]}s')
    print(f'  {len(panel):,} users  ×  {panel.shape[1]} columns')
    print('=' * 65)


if __name__ == '__main__':
    main()
