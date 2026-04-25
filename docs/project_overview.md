# Paw-dvocate: Legislative Intelligence for Animal Advocacy

Paw-dvocate is an applied AI pipeline designed to automate the discovery and analysis of animal-related legislation across United States state legislatures. By bridging the gap between raw policy data and actionable insights, the system ensures that advocacy organizations can influence the legislative process before critical deadlines pass.

## The Problem: The "Awareness Gap"

Currently, many advocacy groups rely on manual web searches, word-of-mouth, or high-traffic listservs to find relevant bills. This fragmented approach often leads to "late discovery"—finding out about a bill only after the public comment window has closed or a crucial vote has already taken place. In the fast-moving environment of state politics, missing even a 48-hour window can mean the difference between a policy's success or failure.

## Who Benefits?

- **Advocacy Organizations**: Gain a "first-look" advantage to mobilize supporters early.
- **Policy Researchers**: Save hundreds of hours of manual filtering by accessing pre-categorized bill sets.
- **Engineering Interns**: Access a modular, real-world example of how to apply Natural Language Processing (NLP) to civic tech.

## How It Works

Paw-dvocate replaces manual browsing with a multi-stage automated intelligence pipeline:

### Data Acquisition
The system monitors legislative datasets from major hubs including California, Texas, and New York.

### Hybrid Classification
To determine if a bill is Pro-Animal, Anti-Animal, or Neutral, the system uses a "triage" approach. It combines:
- Fast keyword scoring
- Deeper semantic embeddings
- Large Language Model (LLM) reasoning to understand the nuances of legislative language

### Data Management
All identified bills and their metadata are structured in a local SQLite database, which is optimized for quick querying and dashboard integration.

### Intelligence Output
The system generates weekly digests and ranks bills by their potential impact, allowing users to prioritize their limited resources.

## Why Paw-dvocate?

Manual monitoring is often reactive and inconsistent, relying on human effort to scan thousands of pages of text. Paw-dvocate is proactive and persistent; it doesn't just look for keywords—it understands intent. By leveraging semantic analysis, it can distinguish between a bill that mentions "animals" in a passing agricultural context and one that significantly impacts welfare standards. This precision moves advocacy from a reactive stance to a proactive strategy.
