# config/keywords.py

# ─────────────────────────────────────────────────────
# KEYWORD LISTS FOR ANIMAL LEGISLATION CLASSIFICATION
# ─────────────────────────────────────────────────────
#
# Three tiers per category:
#   STRONG  (weight 1.0) — Almost certainly animal-related
#   MODERATE (weight 0.6) — Likely animal-related, needs context
#   WEAK    (weight 0.3) — Could be animal-related, high false-positive risk
#
# The keyword_filter module searches bill title + description
# + subjects against these lists and produces a 0–1 score.
# ─────────────────────────────────────────────────────


# ═══════════════════════════════════════════════════════
# PRO-ANIMAL KEYWORDS
# Bills that protect, defend, or improve conditions for animals
# ═══════════════════════════════════════════════════════

PRO_ANIMAL_STRONG = [
    # ── Cruelty & abuse ──
    "animal cruelty",
    "animal abuse",
    "animal neglect",
    "cruelty to animals",
    "cruelty-free",
    "animal cruelty prevention",
    "animal cruelty penalty",
    "animal cruelty felony",

    # ── Welfare & protection ──
    "animal welfare",
    "animal protection",
    "animal rights",
    "humane treatment",
    "humane slaughter",
    "humane society",
    "animal rescue",
    "animal sanctuary",
    "animal shelter",

    # ── Specific protections ──
    "puppy mill",
    "puppy mills",
    "pet store ban",
    "fur ban",
    "fur trade",
    "foie gras",
    "animal testing ban",
    "cosmetic testing",
    "cosmetics animal testing",
    "laboratory animals",
    "lab animal",
    "circus animal",
    "circus animals",
    "wild animal ban",
    "exotic animal ban",
    "trophy hunting ban",
    "shark fin",
    "ivory ban",
    "cetacean captivity",
    "orca captivity",
    "dolphin captivity",

    # ── Factory farming ──
    "factory farming",
    "factory farm",
    "gestation crate",
    "gestation crates",
    "battery cage",
    "battery cages",
    "cage-free",
    "crate-free",
    "veal crate",
    "farrowing crate",
    "concentrated animal feeding",

    # ── Endangered species ──
    "endangered species protection",
    "wildlife protection",
    "wildlife conservation",
    "species preservation",
    "habitat protection",
    "habitat conservation",

    # ── Veterinary access ──
    "veterinary care",
    "veterinary access",
    "pet insurance",
    "animal emergency",
]

PRO_ANIMAL_MODERATE = [
    # ── General terms ──
    "animal welfare standards",
    "companion animal",
    "companion animals",
    "domestic animal",
    "domestic animals",
    "pet adoption",
    "spay and neuter",
    "spay or neuter",
    "sterilization of animals",
    "microchip",
    "animal identification",
    "stray animal",
    "feral cat",
    "feral cats",
    "trap neuter return",
    "tnr program",

    # ── Food labeling ──
    "plant-based",
    "meat alternative",
    "cell-cultured meat",
    "cultivated meat",
    "animal-free",
    "vegan",

    # ── Wildlife ──
    "wildlife corridor",
    "wildlife crossing",
    "wildlife rehabilitation",
    "bird protection",
    "pollinator protection",
    "bee protection",
    "marine mammal",

    # ── Sentience / legal standing ──
    "animal sentience",
    "nonhuman rights",
    "non-human rights",
    "legal personhood",
    "animal personhood",
]

PRO_ANIMAL_WEAK = [
    "animal",
    "animals",
    "pets",
    "livestock welfare",
    "humane",
    "wildlife",
    "endangered",
    "rescue",
    "shelter",
    "habitat",
    "conservation",
    "species",
    "veterinarian",
    "veterinary",
    "cruelty",
]


# ═══════════════════════════════════════════════════════
# ANTI-ANIMAL KEYWORDS
# Bills that weaken protections or expand exploitation
# ═══════════════════════════════════════════════════════

ANTI_ANIMAL_STRONG = [
    # ── Ag-gag / industry shielding ──
    "ag-gag",
    "ag gag",
    "agricultural interference",
    "farm interference",
    "undercover investigation ban",
    "agricultural fraud",
    "farm trespass",
    "agricultural facility trespass",
    "agricultural operation interference",

    # ── Hunting expansion ──
    "hunting season expansion",
    "hunting age reduction",
    "right to hunt",
    "right to fish",
    "constitutional right to hunt",
    "hunting preserve",
    "canned hunt",
    "trophy hunting",
    "bear baiting",
    "hound hunting",

    # ── Trapping ──
    "trapping expansion",
    "fur trapping",
    "leghold trap",
    "snare trap",
    "body-gripping trap",

    # ── Wildlife management (lethal) ──
    "predator control",
    "wildlife killing contest",
    "bounty program",
    "wolf delisting",
    "wolf hunt",

    # ── Animal agriculture deregulation ──
    "preempt local animal",
    "preempt animal welfare",
    "repeal animal welfare",
    "weaken animal welfare",
    "exempt from animal cruelty",
    "agricultural exemption",
    "animal enterprise terrorism",
    "animal enterprise protection",

    # ── Rodeo / entertainment ──
    "rodeo exemption",
    "bullfighting",
    "cockfighting exemption",
    "dogfighting exemption",

    # ── Breed-specific legislation ──
    "breed-specific",
    "breed specific legislation",
    "pit bull ban",
    "dangerous dog breed",
]

ANTI_ANIMAL_MODERATE = [
    "hunting license",
    "hunting permit",
    "trapping license",
    "trapping permit",
    "wildlife management",
    "nuisance animal",
    "nuisance wildlife",
    "depredation permit",
    "lethal control",
    "animal damage control",
    "predator management",
    "livestock protection",
    "livestock loss",
    "right to farm",
    "agricultural protection",
    "game management",
    "game commission",
]

ANTI_ANIMAL_WEAK = [
    "hunting",
    "trapping",
    "slaughter",
    "rodeo",
    "taxidermy",
    "fur",
    "game",
    "predator",
    "depredation",
]


# ═══════════════════════════════════════════════════════
# TIER WEIGHTS
# Used by keyword_filter.py to compute the 0–1 score
# ═══════════════════════════════════════════════════════

TIER_WEIGHTS = {
    "strong":   1.0,
    "moderate": 0.6,
    "weak":     0.3,
}

# Maximum raw score used to normalize to 0–1 range.
# Set empirically: a bill hitting 3 strong keywords is
# very clearly relevant — anything above is capped at 1.0.
MAX_RAW_SCORE = 3.0
