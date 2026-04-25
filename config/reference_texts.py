# config/reference_texts.py

# ─────────────────────────────────────────────────────
# REFERENCE TEXTS FOR EMBEDDING-BASED CLASSIFICATION
# ─────────────────────────────────────────────────────
#
# These sentences act as "anchor points" in embedding space.
# We embed each one with sentence-transformers, average the
# vectors per category, then compare every bill's embedding
# to each category centroid via cosine similarity.
#
# Why multiple sentences per category?
#   A single sentence captures only one angle. Averaging
#   10–15 diverse phrasings creates a more robust centroid
#   that handles the wide variety of bill language.
#
# Guidelines for editing:
#   - Keep sentences concrete and specific (not vague)
#   - Mirror real legislative language where possible
#   - Each sentence should cover a DIFFERENT sub-topic
#   - 10–15 sentences per category is the sweet spot
# ─────────────────────────────────────────────────────


PRO_ANIMAL_REFERENCES = [
    # ── Cruelty & abuse ──
    "This bill increases criminal penalties for animal cruelty and animal abuse.",
    "Establishes felony charges for aggravated cruelty to animals including torture and mutilation.",
    "Prohibits the sale of dogs and cats from puppy mills and commercial breeding facilities.",

    # ── Welfare standards ──
    "Requires minimum space and enrichment standards for farm animals in confinement.",
    "Bans the use of gestation crates, battery cages, and veal crates in agricultural operations.",
    "Mandates humane slaughter practices and pre-slaughter stunning for all livestock.",

    # ── Testing & research ──
    "Prohibits cosmetic testing on animals and requires cruelty-free alternatives.",
    "Phases out the use of animals in laboratory research where validated alternatives exist.",

    # ── Wildlife & habitat ──
    "Protects endangered species habitat and creates wildlife corridors for migration.",
    "Bans trophy hunting, canned hunting, and wildlife killing contests on public land.",
    "Establishes marine protected areas to safeguard marine mammals and ocean ecosystems.",

    # ── Companion animals ──
    "Requires animal shelters to implement trap-neuter-return programs for feral cats.",
    "Creates a statewide animal cruelty registry to prevent convicted abusers from owning pets.",
    "Expands veterinary care access for low-income pet owners through subsidized clinics.",

    # ── Entertainment ──
    "Bans the use of wild and exotic animals in circuses, traveling shows, and roadside zoos.",
    "Prohibits captive orca and dolphin displays for commercial entertainment purposes.",
]


ANTI_ANIMAL_REFERENCES = [
    # ── Ag-gag & industry protection ──
    "This bill criminalizes undercover investigations at agricultural facilities.",
    "Makes it a felony to photograph or record operations at farms and slaughterhouses without consent.",
    "Shields livestock producers from civil liability for standard agricultural practices.",

    # ── Hunting & trapping expansion ──
    "Expands hunting season dates and lowers the minimum age for hunting licenses.",
    "Establishes a constitutional right to hunt, fish, and trap wildlife.",
    "Authorizes the use of leghold traps and snares for commercial fur trapping.",

    # ── Predator control ──
    "Creates a bounty program for killing wolves, coyotes, and other predator species.",
    "Permits aerial gunning and poisoning for livestock predator control operations.",
    "Delists gray wolves from endangered species protection to allow hunting.",

    # ── Deregulation ──
    "Preempts local governments from enacting animal welfare ordinances stricter than state law.",
    "Exempts standard agricultural practices from state animal cruelty statutes.",
    "Repeals cage-free egg requirements and eliminates farm animal confinement standards.",

    # ── Breed bans & entertainment ──
    "Enacts breed-specific legislation banning ownership of pit bulls and similar breeds.",
    "Grants exemptions for rodeos, bullfighting, and traditional animal sporting events.",

    # ── Industry expansion ──
    "Provides subsidies and tax incentives for concentrated animal feeding operations.",
    "Prohibits states from banning the sale of foie gras, fur, and other animal products.",
]


NEUTRAL_REFERENCES = [
    # ── Generic governance ──
    "This bill amends the state tax code to adjust income tax brackets for fiscal year 2025.",
    "Allocates highway infrastructure funding for bridge repair and road maintenance.",
    "Establishes requirements for cybersecurity audits of state government computer systems.",
    "Modifies zoning regulations for residential and commercial real estate development.",
    "Creates a task force to study broadband internet access in rural communities.",

    # ── Education & health ──
    "Increases funding for public school teacher salaries and classroom technology.",
    "Expands Medicaid eligibility for low-income families and children.",
    "Regulates prescription drug pricing and pharmacy benefit manager transparency.",

    # ── Criminal justice ──
    "Reforms bail and pretrial detention policies for nonviolent offenders.",
    "Establishes body camera requirements for law enforcement officers.",

    # ── Environment (non-animal) ──
    "Sets renewable energy portfolio standards and solar panel installation incentives.",
    "Regulates per- and polyfluoroalkyl substances in municipal drinking water systems.",

    # ── Business & labor ──
    "Raises the state minimum wage and mandates paid family leave for employees.",
    "Creates small business grant programs for minority-owned enterprises.",
]
