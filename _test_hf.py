from src.classifier.openpaws_scorer import score_bill_alignment

bill = {
    "title": "Animal Cruelty Prevention Act",
    "description": "Bans puppy mills and increases cruelty penalties",
    "subjects": "Animals",
}

print("Testing HF Phi-3.5 via Open Paws scorer...")
r = score_bill_alignment(bill)

print(f"  Success: {r['success']}")
print(f"  Backend: {r['backend']}")
print(f"  Score:   {r['openpaws_alignment_score']:+.2f}")
print(f"  Summary: {r['openpaws_framing_summary'][:120]}")
if r.get("error"):
    print(f"  Error:   {r['error']}")
