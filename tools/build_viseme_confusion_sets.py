"""Build the viseme-confusion training corpus for personalized LoRA.

Reads:
  - app/src/main/assets/viseme_map_fine.txt  (12-class ARPABET -> viseme map
                                              tuned for LoRA training; see
                                              file header for why this is
                                              different from viseme_map.txt)
  - cmudict.dict                              (CMUSphinx — word -> ARPABET)
  - Norvig count_1w.txt                       (Google web-corpus unigram
                                               counts; used to filter rare /
                                               weird words)

Writes:
  - app/src/main/assets/viseme_confusion_sets.json

What a "viseme confusion set" is
--------------------------------
A group of English words that map to the EXACT SAME viseme sequence under
Liperty's 9-class viseme map but to DIFFERENT ARPABET phoneme sequences.

  e.g. {pat, bat, mat} all map to "V2 V1 V4" (bilabial-vowel-alveolar)
       but their phoneme sequences differ on the onset: P/B/M.

The model literally cannot distinguish them from coarse-grained mouth
motion alone — these are the words VisemeRescorer's LM is most likely
to swap, and the words where a generic LoRA pass gives no gradient
because there's no visual signal to push the loss either way.

The point of this file is to flip that around: collect minimal-pair
recordings of the user actually saying each member of every confusion
set, and use the resulting (video, label) pairs as the LoRA training
distribution. Now the model is being asked to find the user-specific
visual micro-features (degree of jaw drop, tongue visibility through
the teeth gap, asymmetric lip motion, breath plosion, ...) that DO
distinguish the members of each set for THIS user.

Output schema
-------------
JSON object with fields:
  "metadata": {
    "viseme_count": int,
    "total_sets": int,
    "common_word_threshold_rank": int,   # words below this freq rank kept
    "version": "v1",
    "source": "...",
  },
  "sets": [
    {
      "viseme_seq": "V2 V1 V4",
      "words": ["bat", "pat", "mat", "fat"],   # sorted by freq, descending
      "phoneme_seqs": [["B","AE","T"], ["P","AE","T"], ...],
      "score": float,    # avg log-freq across members; higher = more useful
    },
    ...
  ]   # sorted by score descending — top-K is what the calibration UI shows
"""

import json
import math
import os
import re
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VISEME_MAP_PATH = os.path.join(ROOT, "app", "src", "main", "assets", "viseme_map_fine.txt")
OUTPUT_PATH = os.path.join(ROOT, "app", "src", "main", "assets", "viseme_confusion_sets.json")
CMUDICT_URL = "https://raw.githubusercontent.com/cmusphinx/cmudict/master/cmudict.dict"
# Peter Norvig's word-frequency table derived from Google's web n-grams
# corpus. ~5 MB, 333k entries. Cached after first run.
FREQ_URL = "https://norvig.com/ngrams/count_1w.txt"
CACHE_DIR = os.path.join(ROOT, "tools", ".cache")

# Sets with fewer than this many distinct phoneme sequences are dropped
# (singletons aren't "confusion" sets — there's nothing to confuse).
MIN_DISTINCT_PHONEME_SEQS = 2

# Drop words below this corpus-rank cutoff. Norvig's count_1w.txt is
# sorted by frequency, so rank == line number (0-indexed). Top 30k
# captures most everyday vocabulary; below that we hit specialised
# nouns and inflections that aren't useful for guided recording.
COMMON_WORD_RANK_CUTOFF = 30000

# Cap on words per set in the output JSON. Most confusion sets explode
# at the V1-dominated middle (vowels collapse into one viseme), so the
# top-N filter keeps the output JSON tractable AND keeps the user's
# calibration session focused on the most-frequent members.
MAX_WORDS_PER_SET = 6

# Skip very long words. Long words have many viseme positions so the
# probability of a same-viseme-different-phoneme collision drops fast,
# AND they're more painful to read aloud one at a time. Sweet spot
# for guided calibration is short, frequent words.
MAX_WORD_LEN_CHARS = 7


def _ensure_cache(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def _fetch_cached(url, cache_filename):
    """Download `url` once and cache it under tools/.cache/. Returns the
    decoded UTF-8 string."""
    cache_path = os.path.join(CACHE_DIR, cache_filename)
    _ensure_cache(cache_path)
    if os.path.exists(cache_path) and os.path.getsize(cache_path) > 0:
        print(f"  cache hit: {cache_path}")
        with open(cache_path, "r", encoding="utf-8") as f:
            return f.read()
    print(f"  fetching {url} -> {cache_path}")
    with urllib.request.urlopen(url) as r:
        text = r.read().decode("utf-8", errors="replace")
    with open(cache_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"    {len(text):,} bytes")
    return text


def load_viseme_map(path):
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                continue
            out[parts[0]] = parts[1]
    return out


def load_word_frequencies():
    """Return (word_lower -> rank). Rank starts at 0 for the most common
    word. Used both for filtering (drop everything below the cutoff) and
    for scoring (sets with frequent words rank higher).
    """
    text = _fetch_cached(FREQ_URL, "count_1w.txt")
    ranks = {}
    for i, raw in enumerate(text.splitlines()):
        line = raw.strip()
        if not line:
            continue
        # Format: "WORD\tCOUNT"
        parts = line.split()
        if len(parts) < 1:
            continue
        word = parts[0].lower()
        if word not in ranks:
            ranks[word] = i
    print(f"  loaded {len(ranks):,} ranked words")
    return ranks


def parse_cmudict(text, viseme_map):
    """Yield (word_lower, phoneme_tuple, viseme_seq_str) for every cmudict
    entry whose phonemes are fully covered by the viseme map. Filters out
    pronunciation variants (only the canonical form is kept) and obvious
    junk (proper nouns starting uppercase in the original cmudict, abbrev
    words with dots, etc.).
    """
    yielded = 0
    skipped_unk = 0
    skipped_variant = 0
    seen_canonical = set()
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        wordkey, phones_str = parts
        # Drop pronunciation-variant suffixes "(2)", "(3)" — keep only
        # the canonical pronunciation. (cmudict lists the canonical
        # without a suffix first, then alternates with (2), (3)... .)
        canonical_match = re.match(r"^(.+?)(\(\d+\))?$", wordkey)
        if canonical_match is None:
            continue
        canonical = canonical_match.group(1).lower()
        if canonical_match.group(2):
            # Variant — skip; we already saw the canonical.
            skipped_variant += 1
            continue
        if canonical in seen_canonical:
            continue
        # Junk filters: no dots (abbreviations), no apostrophes-without-
        # context (cmudict has both), no internal whitespace.
        if "." in canonical or "'" in canonical:
            continue
        if not re.fullmatch(r"[a-z\-]+", canonical):
            continue
        # Length filter
        if len(canonical) > MAX_WORD_LEN_CHARS or len(canonical) < 2:
            continue
        seen_canonical.add(canonical)
        # Strip stress digits (0/1/2 trailing on each vowel)
        phones = tuple(re.sub(r"\d+$", "", p) for p in phones_str.split())
        if not phones:
            continue
        visemes = []
        had_unknown = False
        for p in phones:
            v = viseme_map.get(p)
            if v is None:
                had_unknown = True
                break
            visemes.append(v)
        if had_unknown:
            skipped_unk += 1
            continue
        yielded += 1
        yield canonical, phones, " ".join(visemes)
    print(f"  yielded={yielded:,} skipped_unknown_phone={skipped_unk:,} "
          f"skipped_variant={skipped_variant:,}")


def build_confusion_sets(cmudict_text, viseme_map, word_ranks):
    """Group cmudict entries by viseme sequence. Return list of dicts:
        {viseme_seq, words, phoneme_seqs, score}
    sorted by score desc.

    Filtering steps:
      1. Word must be in word_ranks AND rank < COMMON_WORD_RANK_CUTOFF.
         (drops rare cmudict tail: proper-name-derived, jargon, etc.)
      2. Bucket must have >= MIN_DISTINCT_PHONEME_SEQS distinct phoneme
         sequences (true homophones don't count toward the threshold —
         they're indistinguishable BY DEFINITION, not just visually).
      3. Cap each bucket at MAX_WORDS_PER_SET, keeping the most common
         members.
    """
    # viseme_seq -> { phoneme_tuple: [words sorted by rank asc] }
    grouped = {}
    kept = 0
    dropped_rare = 0
    for word, phones, vseq in parse_cmudict(cmudict_text, viseme_map):
        rank = word_ranks.get(word)
        if rank is None or rank >= COMMON_WORD_RANK_CUTOFF:
            dropped_rare += 1
            continue
        kept += 1
        bucket = grouped.setdefault(vseq, {})
        bucket.setdefault(phones, []).append((word, rank))
    print(f"  bucketed {kept:,} common cmudict words "
          f"({dropped_rare:,} dropped as rare)")

    sets = []
    for vseq, phones_to_words in grouped.items():
        if len(phones_to_words) < MIN_DISTINCT_PHONEME_SEQS:
            continue
        # Pick one canonical (rank-best) word per phoneme sequence — true
        # homophones add nothing because the user can't disambiguate
        # them by speaking differently.
        canonical_per_phones = []
        for phones, entries in phones_to_words.items():
            entries.sort(key=lambda e: e[1])
            word, rank = entries[0]
            canonical_per_phones.append({
                "word": word,
                "rank": rank,
                "phones": phones,
            })
        # Order set members by frequency (rank asc).
        canonical_per_phones.sort(key=lambda e: e["rank"])
        kept_members = canonical_per_phones[:MAX_WORDS_PER_SET]
        if len(kept_members) < MIN_DISTINCT_PHONEME_SEQS:
            continue
        # Score = mean log-inverse-rank across kept members. Higher
        # means the set is composed of more-frequent words, hence more
        # likely to actually come up in real speech and a higher
        # priority for the calibration UI.
        score = sum(
            math.log(max(1, COMMON_WORD_RANK_CUTOFF - m["rank"]))
            for m in kept_members
        ) / len(kept_members)
        sets.append({
            "viseme_seq": vseq,
            "words": [m["word"] for m in kept_members],
            "phoneme_seqs": [list(m["phones"]) for m in kept_members],
            "score": round(score, 3),
        })
    sets.sort(key=lambda s: s["score"], reverse=True)
    return sets


def main():
    if not os.path.exists(VISEME_MAP_PATH):
        sys.exit(f"missing {VISEME_MAP_PATH}")
    print(f"Loading viseme map: {VISEME_MAP_PATH}")
    vmap = load_viseme_map(VISEME_MAP_PATH)
    print(f"  {len(vmap)} phoneme -> viseme entries")

    print("Loading word frequencies...")
    word_ranks = load_word_frequencies()

    print("Fetching cmudict...")
    cmu_text = _fetch_cached(CMUDICT_URL, "cmudict.dict")

    print("Building confusion sets...")
    sets = build_confusion_sets(cmu_text, vmap, word_ranks)
    print(f"  {len(sets):,} confusion sets")
    if sets:
        sizes = sorted(len(s["words"]) for s in sets)
        print(f"  members-per-set: min={sizes[0]} "
              f"median={sizes[len(sizes)//2]} "
              f"p95={sizes[int(len(sizes)*0.95)]} "
              f"max={sizes[-1]}")
        print(f"  top-10 sets by score:")
        for s in sets[:10]:
            print(f"    [{s['score']:6.2f}] {s['viseme_seq']:<22}  "
                  f"{', '.join(s['words'])}")

    out = {
        "metadata": {
            "viseme_count": len(set(vmap.values())),
            "total_sets": len(sets),
            "common_word_threshold_rank": COMMON_WORD_RANK_CUTOFF,
            "max_words_per_set": MAX_WORDS_PER_SET,
            "max_word_len_chars": MAX_WORD_LEN_CHARS,
            "version": "v1",
            "source": "build_viseme_confusion_sets.py "
                      "(viseme_map.txt + cmudict.dict + Norvig count_1w.txt)",
        },
        "sets": sets,
    }
    print(f"Writing {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, separators=(",", ":"), ensure_ascii=False)
    sz_kb = os.path.getsize(OUTPUT_PATH) / 1024
    print(f"  {sz_kb:.1f} KB")


if __name__ == "__main__":
    main()
