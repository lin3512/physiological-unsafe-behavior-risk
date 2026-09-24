"""Apriori mining and conversion of association rules to fixed channel weights."""
from __future__ import annotations

from features import CHANNELS


def apriori_rules(transactions: list[list[str]], cfg: dict):
    """Apriori rules with Unsafe consequent and same-channel level pruning."""
    n = len(transactions)
    if n == 0:
        raise ValueError("Cannot mine association rules from an empty training set")
    tx = [set(t) for t in transactions]
    level = {frozenset([item]) for item in set.union(*tx)}
    frequent, k = {}, 1
    while level:
        counts = {itemset: sum(itemset <= row for row in tx) for itemset in level}
        level = {s for s, count in counts.items() if count / n >= cfg["min_support"]}
        frequent.update({s: counts[s] for s in level})
        if not level:
            break
        candidates = set()
        current = sorted(level, key=lambda s: sorted(s))
        for i, left in enumerate(current):
            for right in current[i+1:]:
                candidate = left | right
                if len(candidate) != k + 1:
                    continue
                if any(sum(item.startswith(f"{ch.upper()}_") for item in candidate) > 1
                       for ch in CHANNELS):
                    continue
                if all((candidate - {item}) in level for item in candidate):
                    candidates.add(candidate)
        level, k = candidates, k + 1

    rules = []
    unsafe_rate = sum("Unsafe" in row for row in tx) / n
    for itemset, count in frequent.items():
        if "Unsafe" not in itemset or len(itemset) < 2:
            continue
        antecedent = itemset - {"Unsafe"}
        if any(item in ("Normal", "Unsafe") for item in antecedent):
            continue
        antecedent_count = sum(antecedent <= row for row in tx)
        confidence = count / antecedent_count if antecedent_count else 0
        min_confidence = (cfg["min_confidence_single"] if len(antecedent) == 1
                          else cfg["min_confidence_multi"])
        lift = confidence / unsafe_rate if unsafe_rate else 0
        if confidence >= min_confidence and lift > cfg["min_lift"]:
            support = count / n
            rules.append({"antecedent": sorted(antecedent), "consequent": "Unsafe",
                          "support": support, "confidence": confidence, "lift": lift,
                          "strength": support * confidence * lift})
    return rules


def channel_weights(rules: list[dict]) -> dict:
    strongest = {ch: 0.0 for ch in CHANNELS}
    for rule in rules:
        if len(rule["antecedent"]) != 1:
            continue
        item = rule["antecedent"][0]
        for ch in CHANNELS:
            if item.startswith(ch.upper() + "_"):
                strongest[ch] = max(strongest[ch], rule["strength"])
    total = sum(strongest.values())
    if total <= 0:
        raise ValueError("No qualifying single-indicator Unsafe rules; cannot derive channel weights")
    return {ch: strength / total for ch, strength in strongest.items()}
