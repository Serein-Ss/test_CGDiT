"""Prepare a reproducible audit queue; never infer Curie labels from `tc`."""

import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/magndata"
OUT = ROOT / "data/derived"
SEED = 20260914


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def prepare():
    rows = []
    for split in ("train", "val", "test"):
        for row in read_csv(RAW / f"{split}.csv"):
            row["split"] = split
            row["structure_hash"] = hashlib.sha256(row["cif"].encode()).hexdigest()
            rows.append(row)
    assert len({r["material_id"] for r in rows}) == len(rows)
    assert {r["material_id"] for r in rows} == {
        r["material_id"] for r in read_csv(RAW / "all.csv")
    }
    groups = defaultdict(list)
    for row in rows:
        groups[row["structure_hash"]].append(row)
    conflicts = {
        h for h, group in groups.items()
        if len({float(r["tc"]) for r in group}) > 1
    }

    records = []
    for row in rows:
        tc = float(row["tc"])
        reasons = []
        if tc >= 300:
            reasons.append("recorded_temperature_ge300")
        if row["structure_hash"] in conflicts:
            reasons.append("identical_cif_multiple_temperatures")
        if tc == 0:
            reasons.append("recorded_zero_requires_source_check")
        if row["material_id"] == "0.782_NdScO3":
            reasons.insert(0, "literature_unit_conflict_953_vs_0.953")
        if row["material_id"] in ("0.639_Mn2Au", "0.108_Mn3Ir"):
            reasons.insert(0, "known_afm_temperature_semantics")
        records.append({
            "record_id": row["material_id"], "split": row["split"],
            "seq_id": row["seq_id"], "formula": row["pretty_formula"],
            "recorded_temperature_k": tc,
            "structure_id": row["structure_hash"],
            "natoms": int(row["natoms"]),
            "spacegroup_number": int(row["spacegroup.number"]),
            "priority_reasons": ";".join(reasons),
            "magndata_url": "https://www.cryst.ehu.es/magndata/index.php?this_label="
                + row["material_id"].split("_", 1)[0],
            "observation_status": "unverified_legacy_label",
            "event_type": "unknown", "order_below": "unknown",
            "order_above": "unknown", "curie_temperature_k": None,
            "curie_lower_k": None, "curie_upper_k": None,
            "bound_kind": "unknown", "pressure": "unknown",
            "specimen": "unknown", "source_doi": None,
            "structure_source_match": "unverified",
            "training_eligible": False,
        })
    records.sort(key=lambda r: r["record_id"])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "legacy_observations.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
    )
    fields = ["record_id", "split", "seq_id", "formula", "recorded_temperature_k",
              "natoms", "spacegroup_number", "priority_reasons", "magndata_url"]
    queue = [{k: r[k] for k in fields} for r in records if r["priority_reasons"]]
    write_csv(OUT / "priority_queue.csv", queue, fields)

    # Deliberate diagnostic cases plus seeded controls. Not a representative test set.
    diagnostic_ids = [
        "0.782_NdScO3", "0.639_Mn2Au", "0.108_Mn3Ir",
        "0.186_CeMnAsO", "0.187_CeMnAsO", "0.188_CeMnAsO",
        "0.426_EuMnBi2", "2.51_EuMnBi2", "0.65_Fe2O3-alpha",
        "0.274_Mn4N",
    ]
    by_id = {r["record_id"]: r for r in records}
    assert all(key in by_id for key in diagnostic_ids)
    rng = random.Random(SEED)
    high = [r for r in records if r["recorded_temperature_k"] >= 300
            and r["record_id"] not in diagnostic_ids]
    controls = [r for r in records if not r["priority_reasons"]]
    pilot = [dict(by_id[key], selection_group="targeted_diagnostic")
             for key in diagnostic_ids]
    pilot += [dict(r, selection_group="seeded_high_temperature")
              for r in rng.sample(high, 10)]
    pilot += [dict(r, selection_group="seeded_other_temperature_control")
              for r in rng.sample(controls, 10)]
    pilot_fields = fields + ["selection_group"]
    write_csv(OUT / "pilot30.csv", [{k: r[k] for k in pilot_fields} for r in pilot],
              pilot_fields)

    # SG and atom count are only triage bins, NOT validated structural families.
    bins = defaultdict(list)
    for r in records:
        bins[(r["spacegroup_number"], r["natoms"])].append(r)
    inventory = []
    for (sg, n), members in bins.items():
        inventory.append({
            "spacegroup_number": sg, "natoms": n,
            "records": len(members), "formulas": len({r["formula"] for r in members}),
            "recorded_ge500": sum(r["recorded_temperature_k"] >= 500 for r in members),
            "within_24atom_budget": n <= 24,
            "family_verified": False, "verified_curie_pairs": 0,
        })
    inventory.sort(key=lambda r: (-r["recorded_ge500"], -r["records"],
                                  r["spacegroup_number"], r["natoms"]))
    write_csv(OUT / "structural_triage_bins.csv", inventory, list(inventory[0]))
    counts = {
        "input_records": len(records), "priority_union_records": len(queue),
        "recorded_ge300": sum(r["recorded_temperature_k"] >= 300 for r in records),
        "recorded_ge500": sum(r["recorded_temperature_k"] >= 500 for r in records),
        "recorded_zero": sum(r["recorded_temperature_k"] == 0 for r in records),
        "identical_cif_conflict_groups": len(conflicts),
        "identical_cif_conflict_records": sum(len(groups[h]) for h in conflicts),
        "pilot_records": len(pilot), "seed": SEED,
        "split_counts": dict(Counter(r["split"] for r in records)),
        "note": "Counts refer to original labels, not verified Curie temperatures.",
    }
    (ROOT / "results/preparation_summary.json").write_text(
        json.dumps(counts, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(counts, ensure_ascii=False))


if __name__ == "__main__":
    prepare()
