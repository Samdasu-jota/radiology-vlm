import json
import re


# Canonical findings and their aliases (plural forms, synonyms from the IU dataset)
FINDINGS = {
    "cardiomegaly": ["cardiomegaly", "cardiac enlargement", "enlarged heart",
                     "heart is enlarged", "heart enlarged"],
    "pleural effusion": ["pleural effusion", "pleural effusions", "effusion", "effusions"],
    "pneumothorax": ["pneumothorax"],
    "atelectasis": ["atelectasis", "atelectatic"],
    "edema": ["edema", "pulmonary edema", "vascular congestion", "cephalization"],
    "consolidation": ["consolidation", "airspace consolidation", "focal consolidation",
                      "airspace disease", "air space disease"],
    "pneumonia": ["pneumonia", "infiltrate", "infiltrates"],
    "lung opacity": ["opacity", "opacities", "opacification", "airspace opacity"],
    "fracture": ["fracture", "fractures"],
    "lung lesion": ["lesion", "mass", "masses", "nodule", "nodules"],
    "enlarged cardiomediastinum": ["enlarged cardiomediastinum", "mediastinal widening",
                                   "wide mediastinum"],
    "support devices": ["pacemaker", "picc line", "picc", "central line", "chest tube",
                        "endotracheal tube", "tracheostomy"],
}

# Negation cues — longest first so we match greedily
NEGATION_CUES = [
    "no evidence of", "without evidence of", "no radiographic evidence of",
    "no definite", "no acute", "no significant", "no visible", "no suspicious",
    "not identified", "not seen", "not observed", "not demonstrated",
    "negative for", "free of", "clear of",
    "without", "absent", "no ", "deny", "denies",
]

# Phrases that imply absence of specific findings
IMPLICIT_ABSENT_RULES = {
    r"lungs?\s+(are|is)\s+clear": [
        "consolidation", "pneumonia", "lung opacity", "pleural effusion", "pneumothorax"
    ],
    r"heart\s+size\s+(is\s+)?normal": ["cardiomegaly"],
    r"heart\s+(is\s+)?normal\s+in\s+size": ["cardiomegaly"],
    r"cardiomediastinal\s+silhouette\s+(is\s+)?(within\s+)?normal": [
        "cardiomegaly", "enlarged cardiomediastinum"
    ],
    r"no\s+acute\s+(bony|osseous)\s+(abnormality|findings?)": ["fracture"],
    r"(osseous|bony)\s+structures\s+(are\s+)?(within\s+normal|unremarkable|intact)": ["fracture"],
}


def split_sentences(text):
    """Split a radiology report into sentences."""
    text = text.replace("..", ".")
    parts = re.split(r'\.\s+', text)
    return [p.strip().rstrip(".") for p in parts if p.strip()]


def has_negation(text_before):
    """Check if text preceding a mention contains a negation cue."""
    lower = text_before.lower()
    for cue in NEGATION_CUES:
        if cue in lower:
            return True
    return False


def extract_labels(report_text):
    """Extract present/absent findings from a radiology report.

    Returns dict with 'present' and 'absent' lists of canonical finding names.
    """
    present = set()
    absent = set()
    text_lower = report_text.lower()

    # Apply implicit rules on the full report
    for pattern, implied_absent in IMPLICIT_ABSENT_RULES.items():
        if re.search(pattern, text_lower):
            absent.update(implied_absent)

    # Process each sentence for explicit mentions
    for sentence in split_sentences(report_text):
        sent_lower = sentence.lower()
        # Sentence-level negation: covers "No X, Y, or Z" and
        # "Specifically, no evidence of X, Y, or Z" comma-list patterns
        sentence_negated = bool(
            re.match(r'^no\s+', sent_lower)
            or re.search(r'\bno evidence of\b', sent_lower)
            or re.search(r'\bwithout evidence of\b', sent_lower)
            or re.search(r'\bfree of\b', sent_lower)
            or re.search(r'\bclear of\b', sent_lower)
        )

        for canonical, aliases in FINDINGS.items():
            for alias in aliases:
                # Word-boundary match to avoid substring hits (e.g. "ett" in "silhouette")
                pattern = r'\b' + re.escape(alias.lower()) + r'\b'
                match = re.search(pattern, sent_lower)
                if not match:
                    continue
                idx = match.start()

                # Check 50-char window before mention for negation cues
                window_start = max(0, idx - 50)
                text_before = sent_lower[window_start:idx]

                if sentence_negated or has_negation(text_before):
                    absent.add(canonical)
                else:
                    present.add(canonical)
                break  # one alias match per finding per sentence

    # Present wins if a finding appears in both sets
    absent -= present

    return {"present": sorted(present), "absent": sorted(absent)}


def format_structured_text(labels):
    """Convert labels dict to a structured text string for CLIP training."""
    parts = []
    if labels["present"]:
        parts.append("present: " + ", ".join(labels["present"]))
    if labels["absent"]:
        parts.append("absent: " + ", ".join(labels["absent"]))
    if not parts:
        return "no findings mentioned"
    return ". ".join(parts) + "."


def process_dataset(input_path="data/dataset.json",
                    output_path="data/dataset_structured.json"):
    """Run label extraction on the full dataset and save results."""
    with open(input_path, "r") as f:
        pairs = json.load(f)

    print(f"Processing {len(pairs)} reports...")

    results = []
    counts = {"present": {}, "absent": {}}
    no_findings = 0

    for pair in pairs:
        labels = extract_labels(pair["text"])
        structured = format_structured_text(labels)

        results.append({
            "image_path": pair["image_path"],
            "original_text": pair["text"],
            "structured_text": structured,
            "labels": labels,
        })

        if not labels["present"] and not labels["absent"]:
            no_findings += 1

        for status in ("present", "absent"):
            for finding in labels[status]:
                counts[status][finding] = counts[status].get(finding, 0) + 1

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print statistics
    print(f"\nSaved to {output_path}")
    print(f"Total: {len(results)}  |  No findings detected: {no_findings}")
    print(f"\n{'Finding':<30} {'Present':>8} {'Absent':>8}")
    print("-" * 50)
    all_findings = sorted(
        set(list(counts["present"].keys()) + list(counts["absent"].keys()))
    )
    for name in all_findings:
        p = counts["present"].get(name, 0)
        a = counts["absent"].get(name, 0)
        print(f"{name:<30} {p:>8} {a:>8}")


if __name__ == "__main__":
    process_dataset()
