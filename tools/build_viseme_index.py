"""Build the viseme inverse index used by Liperty's VisemeRescorer.

Reads:
  - app/src/main/assets/viseme_map.txt  (ARPABET phoneme -> viseme class)
  - cmudict.dict  (downloaded from CMUSphinx — word -> ARPABET phonemes)

Writes:
  - app/src/main/assets/viseme_index.json  ({viseme_sequence: [words...]})

The inverse index lets the Android-side VisemeRescorer look up
viseme-equivalent words for any input word in O(1) — the rescorer then
generates candidate sentences by substituting equivalents and scores
each with the existing KenLM language model.

Output size: ~2-3 MB JSON (135k cmudict words collapsed to ~50k unique
viseme sequences). Ships compressed inside the APK.

Run from the repo root:
    python tools/build_viseme_index.py
"""

import json
import os
import re
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VISEME_MAP_PATH = os.path.join(ROOT, "app", "src", "main", "assets", "viseme_map.txt")
OUTPUT_PATH = os.path.join(ROOT, "app", "src", "main", "assets", "viseme_index.json")
CMUDICT_URL = "https://raw.githubusercontent.com/cmusphinx/cmudict/master/cmudict.dict"


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


def fetch_cmudict():
    print(f"Fetching cmudict from {CMUDICT_URL}...")
    with urllib.request.urlopen(CMUDICT_URL) as r:
        text = r.read().decode("utf-8")
    print(f"  got {len(text):,} bytes")
    return text


def parse_cmudict(text, viseme_map):
    """Yield (word, viseme_sequence) for every cmudict entry whose phonemes
    are entirely covered by the viseme map. Lossy on rare/foreign phones.

    cmudict format per line:
        WORD  P1 P2 P3 ... # optional comment
    or with pronunciation variants:
        WORD(2)  P1 P2 P3 ...
    Stress digits on vowels: AH1, AY2, etc. — stripped before lookup.
    """
    skipped_partial = 0
    skipped_unknown = 0
    yielded = 0
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        wordkey, phones_str = parts
        # Drop variant suffix "(2)", "(3)" — keep only the canonical word
        word = re.sub(r"\(\d+\)$", "", wordkey)
        # Strip stress digits (0/1/2 trailing on each vowel)
        phones = [re.sub(r"\d+$", "", p) for p in phones_str.split()]
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
            skipped_unknown += 1
            continue
        if not visemes:
            skipped_partial += 1
            continue
        yielded += 1
        # Use a delimiter that won't collide with viseme codes like V12.
        # All viseme codes in our map start with V and are 2-3 chars; use
        # space as a separator so codes can be variable-width without
        # ambiguity.
        yield word, " ".join(visemes)
    print(f"  yielded={yielded:,}  skipped_unknown_phoneme={skipped_unknown:,}  "
          f"skipped_partial={skipped_partial:,}")


def main():
    if not os.path.exists(VISEME_MAP_PATH):
        sys.exit(f"missing {VISEME_MAP_PATH}")
    print(f"Loading viseme map: {VISEME_MAP_PATH}")
    vmap = load_viseme_map(VISEME_MAP_PATH)
    print(f"  {len(vmap)} phoneme -> viseme entries")
    text = fetch_cmudict()
    print("Parsing cmudict...")
    inv = {}
    word_to_seq = {}
    for word, vseq in parse_cmudict(text, vmap):
        # Keep only the FIRST pronunciation for each word (most common /
        # canonical in cmudict ordering).
        if word in word_to_seq:
            continue
        word_to_seq[word] = vseq
        inv.setdefault(vseq, []).append(word)
    for seq in inv:
        # Deduplicate + sort alphabetically for deterministic output.
        # (Frequency-based sort would be better but cmudict has no freqs.)
        inv[seq] = sorted(set(inv[seq]))
    print(f"Index stats: {len(word_to_seq):,} words across "
          f"{len(inv):,} unique viseme sequences")
    sizes = [len(v) for v in inv.values()]
    if sizes:
        sizes.sort()
        print(f"  candidates-per-sequence: min={sizes[0]} "
              f"median={sizes[len(sizes)//2]} p95={sizes[int(len(sizes)*0.95)]} "
              f"max={sizes[-1]}")
    print(f"Writing {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(inv, f, separators=(",", ":"))
    sz = os.path.getsize(OUTPUT_PATH) / 1e6
    print(f"  {sz:.1f} MB")


if __name__ == "__main__":
    main()
