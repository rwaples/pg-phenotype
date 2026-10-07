"""One line per estimate of every pedsum assortative golden: drops, reasons, boundary, stopping, failures.

python3 tools/pedsum_am_golden_summary.py tests/golden/pedsum_am_<sha12>
"""

import collections
import json
import sys
from pathlib import Path

seen = collections.Counter()
for path in sorted(Path(sys.argv[1]).glob("*.json")):
    if path.name == "provenance.json":
        continue
    record = json.loads(path.read_text())
    sample = record["payload"]["mating_pairs"]
    print(
        record["name"],
        "pairs",
        sample["n_total"],
        "unknown",
        sample["n_dropped"]["unknown_stratum"],
        "networks",
        sample["n_mate_networks"],
    )
    for cell in record["payload"]["mate_correlation"]:
        parts = []
        for form in ("crude", "stratified"):
            for name, e in cell.get(form, {}).items():
                if name == "table":
                    continue
                key = next(k for k in ("r", "rho", "value") if k in e)
                s = f"{form[0]}:{name}="
                if e[key] is None:
                    s += f"U({e['reason']})"
                    seen["estimate:" + e["reason"]] += 1
                else:
                    s += f"{e[key]:.3f}" + ("B" if e.get("boundary") else "")
                    seen["boundary"] += bool(e.get("boundary"))
                    if e["ci_unavailable_reason"]:
                        s += f"[ci:{e['ci_unavailable_reason']}]"
                        seen["ci:" + e["ci_unavailable_reason"]] += 1
                    if "bootstrap" in e and e["bootstrap"]["failure_reasons"]:
                        s += f"[bf:{e['bootstrap']['failure_reasons']}]"
                        seen.update("bootstrap_fail:" + r for r in e["bootstrap"]["failure_reasons"])
                    if "permutations" in e:
                        pm = e["permutations"]
                        s += f"[p={e['p_perm']} used={pm['draws_used']} fixed={pm['n_fixed_fathers']}"
                        s += " STOP" * pm["stopped_early"]
                        if pm["failure_reasons"]:
                            s += f" pf={pm['failure_reasons']}"
                            seen.update("perm_fail:" + r for r in pm["failure_reasons"])
                        if e["p_perm_unavailable_reason"]:
                            s += f" pr={e['p_perm_unavailable_reason']}"
                            seen["p:" + e["p_perm_unavailable_reason"]] += 1
                        s += "]"
                        seen["stopped_early"] += pm["stopped_early"]
                parts.append(s)
        dropped = {k: v for k, v in cell["n_dropped"].items() if v}
        print("  ", cell["mother"], cell["father"], "n", cell["n"], dropped, " ".join(parts))
print(dict(sorted(seen.items())))
