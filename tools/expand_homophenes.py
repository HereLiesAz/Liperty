import json
import collections

# Simplified Viseme Mapping
# Grouping phonemes that look identical on the lips
VISEME_MAP = {
    'p': '1', 'b': '1', 'm': '1',
    'f': '2', 'v': '2',
    't': '3', 'd': '3', 'n': '3', 'l': '3', 's': '3', 'z': '3',
    'sh': '4', 'ch': '4', 'j': '4', 'zh': '4',
    'k': '5', 'g': '5', 'ng': '5',
    'r': '6',
    'w': '7', 'wh': '7',
    'th': '8',
    'a': '0', 'e': '0', 'i': '0', 'o': '0', 'u': '0', 'y': '0'
}

def get_viseme_string(word):
    word = word.lower()
    visemes = ""
    i = 0
    while i < len(word):
        # Check for 2-char matches (sh, ch, th, wh, ng, zh)
        if i + 1 < len(word) and word[i:i+2] in VISEME_MAP:
            visemes += VISEME_MAP[word[i:i+2]]
            i += 2
        elif word[i] in VISEME_MAP:
            visemes += VISEME_MAP[word[i]]
            i += 1
        else:
            i += 1
    # De-duplicate consecutive identical visemes (coarticulation simplification)
    # e.g., "cool" -> k-oo-l -> 503. "cook" -> k-oo-k -> 505.
    # Actually, keep them for now to maintain word length info
    return visemes

def expand():
    # In a real scenario, load from a large word list file
    # For this script, we'll use a set of common words if we can't find a list.

    words = [
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
        "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
        "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
        "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
        "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
        "when", "make", "can", "like", "time", "no", "just", "him", "know", "take",
        "people", "into", "year", "your", "good", "some", "could", "them", "see", "other",
        "than", "then", "now", "look", "only", "come", "its", "over", "think", "also",
        "back", "after", "use", "two", "how", "our", "work", "first", "well", "way",
        "even", "new", "want", "because", "any", "these", "give", "day", "most", "us",
        "mat", "bat", "pat", "meet", "beat", "peat", "fan", "van", "safe", "save"
    ]

    # Attempt to load more words from project if available
    # (Simplified: just use a few more for the demo)

    homophene_groups = collections.defaultdict(list)
    for word in words:
        v_str = get_viseme_string(word)
        if v_str:
            homophene_groups[v_str].append(word)

    # Convert to the required JSON format: { "word": ["alt1", "alt2"] }
    result = {}
    for group in homophene_groups.values():
        if len(group) > 1:
            for word in group:
                alts = [w for w in group if w != word]
                result[word] = alts

    output_path = "app/src/main/assets/homophones.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Expanded homophones saved to {output_path} with {len(result)} entries.")

if __name__ == "__main__":
    expand()
