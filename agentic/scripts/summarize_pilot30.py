"""Record-level review and conservative family availability, without training labels."""
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    pilot=list(csv.DictReader((ROOT/'data/derived/pilot30.csv').open()))
    cases=json.loads((ROOT/'evidence/pilot30_sources_v2.json').read_text())
    responses={r['case_id']:r for r in map(json.loads,(ROOT/'data/curation/pilot30_intern_v2.jsonl').read_text().splitlines())}
    byrecord={rid:c for c in cases for rid in c['record_ids']}
    if set(byrecord)!={r['record_id'] for r in pilot}:
        raise ValueError('pilot reconciliation failed')
    issue_map={
        '0.782':['unit_conflict_1000_fold','event_type_unknown'],
        '0.639':['lower_bound_lost','neel_not_curie'],
        '0.108':['neel_not_curie','original_measurement_pending'],
        '0.186':['neel_not_curie','multiple_events_same_cif'],
        '0.187':['spin_reorientation_and_sublattice_order','multiple_events_same_cif'],
        '0.188':['below_not_exact','multiple_events_same_cif'],
        '0.426':['neel_not_curie','multiple_events_same_cif'],
        '2.51':['cross_source_value_discrepancy','multiple_events_same_cif'],
        '0.65':['canted_afm_not_conventional_fm'],
        '0.274':['original_measurement_pending'],
        '1.495':['neel_not_curie'], '1.253':['neel_not_curie','reported_uncertainty'],
        '0.1024':['canted_afm_not_conventional_fm','approximate_value'],
        '1.281':['neel_not_curie','oxygen_occupancy_match_pending'],
        '0.836':['canted_afm_not_conventional_fm','fit_parameter_vs_measurement'],
        '0.619':['impurity_attribution_conflict'], '0.802':['neel_not_curie','phase_match_pending'],
        '1.420':['insufficient_context','original_measurement_pending'],
        '0.617':['upper_bound_lost','event_type_unknown'],
        '1.292':['transition_vs_measurement_unresolved','superconducting_tc_ambiguity'],
        '0.909':['cross_source_value_discrepancy','neel_not_curie'],
        '0.36':['canted_afm_not_conventional_fm'],
        '0.327':['fm_order_supported_temperature_unverified'],
        '1.314':['multiple_events','natural_vs_synthetic'],
        '1.0.59':['polytype_match_pending','cross_source_value_discrepancy'],
        '1.775':['cross_sample_value_discrepancy','neel_not_curie'],
        '1.68':['possible_cross_compound_attribution','neel_not_curie']}
    output=[]
    for r in pilot:
        c=byrecord[r['record_id']]; a=responses.get(c['case_id'],{})
        label=r['record_id'].split('_')[0]
        types={'0.186':'neel','0.187':'spin_reorientation_and_neel','0.188':'other'}
        assessment=types.get(label,c['assistant_event_assessment'])
        issues=issue_map.get(label,['primary_evidence_unresolved'])
        output.append(dict(record_id=r['record_id'],split=r['split'],legacy_temperature_k=r['recorded_temperature_k'],
            assistant_event_assessment=assessment,evidence_available=bool(c['excerpts']),
            source_role=c['source_role'],source_url=c['url'],doi=c.get('doi'),
            candidate_family=c.get('candidate_family'),family_validated=False,
            structure_match_status='database_id_matched_coordinates_unverified' if label=='0.617' else 'unverified',
            sample_conditions_verified=False,independent_review_complete=False,training_eligible=False,
            issues=';'.join(issues),assistant_note=c['assistant_note'],
            api_status=a.get('status','not_run'),api_semantic_flags=';'.join(a.get('semantic_flags',[])),
            api_events_json=json.dumps(a.get('extraction',{}).get('events',[]),ensure_ascii=False)))
    with (ROOT/'data/curation/pilot30_review.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(output[0]));writer.writeheader();writer.writerows(output)
    (ROOT/'data/curation/pilot30_review.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in output))
    families=defaultdict(list)
    for r in output:
        if r['candidate_family']:
            families[r['candidate_family']].append(r)
    family_rows=[dict(candidate_family=k,records=len(v),train_records=sum(r['split']=='train' for r in v),
        curie_candidate_train_records=sum(r['split']=='train' and r['assistant_event_assessment']=='curie_candidate' for r in v),
        validated_family=False,verified_preference_pairs=0,record_ids=';'.join(r['record_id'] for r in v)) for k,v in sorted(families.items())]
    with (ROOT/'results/pilot30_family_availability.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(family_rows[0]));writer.writeheader();writer.writerows(family_rows)
    summary=dict(pilot_records=len(output),evidence_available_records=sum(r['evidence_available'] for r in output),
        unresolved_record_ids=[r['record_id'] for r in output if not r['evidence_available']],
        assessment_counts=dict(Counter(r['assistant_event_assessment'] for r in output)),
        issue_counts=dict(Counter(i for r in output for i in r['issues'].split(';'))),
        candidate_families=len(families),validated_families=0,training_eligible_records=0,verified_preference_pairs=0,
        decision='Do not start diffusion preference training: event/structure/sample evidence insufficient.',
        bias_note='Diagnostic selection, not a random population survey. No global contamination estimate.',
        source_link_note='Same formula and space group are not coordinate/sample matching.')
    (ROOT/'results/pilot30_readiness.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    # Verify original snapshot files stayed byte-identical to project inputs.
    raw=ROOT/'data/raw/magndata'
    checks=[]
    for p in raw.iterdir():
        original=ROOT.parent/'data/magndata'/p.name
        if p.is_file() and original.is_file():
            equal=hashlib.sha256(p.read_bytes()).digest()==hashlib.sha256(original.read_bytes()).digest()
            checks.append(dict(file=p.name,identical=equal))
    (ROOT/'results/stage2_snapshot_checks.json').write_text(json.dumps(checks,indent=2))
    assert all(c['identical'] for c in checks)
    assert not any(r['training_eligible'] for r in output)
    print(json.dumps(summary,ensure_ascii=False))

if __name__=='__main__':
    main()
