"""Descriptive, paired audit of the stopped 500-structure campaign."""
import collections
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import warnings

import numpy as np
from pymatgen.io.vasp.inputs import Incar
from pymatgen.io.vasp.outputs import Eigenval

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "output/basinguide/returned/stopped_20260914_audit"
OUT = ROOT / "output/basinguide/analysis/stopped_20260914"


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def hit(value):
    return 2.5 <= value <= 3.5


def distribution(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return dict(n=0)
    return dict(n=len(values), mean=float(values.mean()), median=float(np.median(values)),
                q25=float(np.quantile(values, .25)), q75=float(np.quantile(values, .75)),
                hits=int(sum(hit(x) for x in values)), below_0p1=int(sum(values < .1)),
                below_target=int(sum(values < 2.5)), above_target=int(sum(values > 3.5)))


def paired(rows, first, second):
    selected = [r for r in rows if r.get(first) is not None and r.get(second) is not None]
    a = np.array([r[first] for r in selected]); b = np.array([r[second] for r in selected])
    counts = {f'{x}_{y}': 0 for x in ['hit', 'miss'] for y in ['hit', 'miss']}
    for x, y in zip(a, b):
        counts[f'{"hit" if hit(x) else "miss"}_{"hit" if hit(y) else "miss"}'] += 1
    return dict(n=len(selected), categories=counts, first=distribution(a), second=distribution(b),
                mean_second_minus_first=float(np.mean(b-a)) if len(a) else None,
                median_second_minus_first=float(np.median(b-a)) if len(a) else None,
                mae=float(np.mean(abs(b-a))) if len(a) else None,
                abs_shift_above_0p5=int(sum(abs(b-a) > .5)),
                abs_shift_above_1=int(sum(abs(b-a) > 1)),
                closer_to_target=int(sum(abs(b-3) < abs(a-3))))


def main():
    manifest = read(DATA / "audit_manifest.json")
    if not manifest:
        raise RuntimeError("Archive audit is not complete")
    cohort = read(DATA / "campaign/cohort.json")
    assert [r['sample_id'] for r in cohort] == list(range(500))
    original = ROOT / "transfer/basinguide_bg3_500_dft"
    assert (DATA / 'campaign/cohort.json').read_bytes() == (original / 'campaign/cohort.json').read_bytes()
    outcars = manifest['outcar_checks']
    rows, problems, parser_warnings, protocols, pot_changes = [], [], [], collections.Counter(), []
    for item in cohort:
        i = item['sample_id']; sample = DATA / 'campaign/samples' / f'{i:04d}'
        assert (sample / 'generated.json').read_bytes() == (original / 'campaign/samples' / f'{i:04d}' / 'generated.json').read_bytes()
        dft = sample / 'dft'; result = read(dft / 'result.json')
        if result:
            assert result['sample_id'] == i
        row = dict(sample_id=i, formula=item['formula'], num_atoms=item['num_atoms'],
                   predictor_raw_eV=item['predicted_bg_raw_eV'], status=result.get('status','missing'),
                   error=result.get('error'), dft_started=dft.exists(), raw_eV=None, final_eV=None, bands_eV=None)
        relax_valid = False
        for stage in ['raw_scf', 'relax_0', 'relax_1', 'scf', 'bands']:
            folder = dft / stage; record = read(folder / 'result.json')
            oc = outcars.get(f'basinguide_bg3_500_dft/campaign/samples/{i:04d}/dft/{stage}/OUTCAR', {})
            good = record.get('electronic_converged') is True and record.get('returncode') == 0 and oc.get('normal_footer') is True
            row[f'{stage}_electronic_ok'] = good
            if stage.startswith('relax'):
                row[f'{stage}_ionic_ok'] = good and record.get('ionic_converged') is True and oc.get('ionic_accuracy_message') is True
                relax_valid |= row[f'{stage}_ionic_ok']
                continue
            if (folder / 'INCAR').exists():
                inc = Incar.from_file(folder / 'INCAR')
                params = {k: inc.get(k) for k in ['ENCUT','EDIFF','ISPIN','ISYM','ISMEAR','SIGMA','LREAL','LASPH','LDAU','ICHARG']}
                protocols[(stage, json.dumps(params, sort_keys=True))] += 1
                inp = read(folder / 'input_manifest.json')
                for pot in inp.get('pseudopotentials', []):
                    path = original / 'potentials' / pot['symbol'] / 'POTCAR'
                    if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != pot['sha256']:
                        pot_changes.append(dict(sample_id=i,stage=stage,symbol=pot['symbol']))
            if not good:
                continue
            if stage in ['raw_scf', 'scf']:
                electronic = [line.split() for line in (folder / 'OSZICAR').read_text().splitlines()
                              if re.match(r'\s*(DAV|RMM|CG):', line)][-1]
                oszicar_ok = (int(electronic[1]) < 200 and abs(float(electronic[3])) < 1e-6
                              and abs(float(electronic[4])) < 1e-6)
                row[f'{stage}_oszicar_ok'] = oszicar_ok
                if not oszicar_ok:
                    problems.append(dict(sample_id=i,stage=stage,type='oszicar_not_converged'))
                    continue
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    eigen = Eigenval(folder / 'EIGENVAL')
                    assert len(eigen.kpoints) == eigen.nkpt
                    gap = float(eigen.eigenvalue_band_properties[0])
                parser_warnings.extend(dict(sample_id=i,stage=stage,message=str(w.message)) for w in caught)
                assert np.isfinite(gap)
                row[f'{stage}_eigenval_gap_eV'] = gap
                row[f'{stage}_gap_delta_eV'] = gap - record['gap_eV']
                if abs(gap - record['gap_eV']) > .02:
                    problems.append(dict(sample_id=i,stage=stage,type='gap_disagreement',difference=gap-record['gap_eV']))
                key = {'raw_scf':'raw_eV','scf':'final_eV','bands':'bands_eV'}[stage]
                if stage == 'raw_scf' or relax_valid:
                    row[key] = record['gap_eV']
            except Exception as exc:
                problems.append(dict(sample_id=i,stage=stage,type='parse_error',message=str(exc)))
        row['relaxation_verified'] = relax_valid
        row['final_verified'] = row['final_eV'] is not None
        row['final_hit'] = hit(row['final_eV']) if row['final_verified'] else None
        rows.append(row)
    triples = [r for r in rows if r['raw_eV'] is not None and r['final_eV'] is not None]
    patterns = collections.Counter(''.join('H' if hit(r[k]) else 'M' for k in ['predictor_raw_eV','raw_eV','final_eV']) for r in triples)
    summary = dict(n_total=500,statuses=dict(collections.Counter(r['status'] for r in rows)),
        n_started=sum(r['dft_started'] for r in rows),
        initial_predictor=distribution([r['predictor_raw_eV'] for r in rows]),
        raw_dft=distribution([r['raw_eV'] for r in rows if r['raw_eV'] is not None]),
        final_dft=distribution([r['final_eV'] for r in rows if r['final_eV'] is not None]),
        predictor_to_raw=paired(rows,'predictor_raw_eV','raw_eV'),
        predictor_to_final=paired(rows,'predictor_raw_eV','final_eV'),
        raw_to_final=paired(rows,'raw_eV','final_eV'),
        final_to_bands=paired(rows,'final_eV','bands_eV'),
        triple_patterns=dict(patterns), n_triples=len(triples),
        completion_groups={label:dict(n=len(group), predictor=distribution([r['predictor_raw_eV'] for r in group]),
            mean_atoms=float(np.mean([r['num_atoms'] for r in group])))
            for label,group in [(name,[r for r in rows if r['final_verified']==flag]) for name,flag in [('final_available',True),('final_unavailable',False)]]},
        examples={name:[{k:r[k] for k in ['sample_id','formula','predictor_raw_eV','raw_eV','final_eV']} for r in triples
            if ''.join('H' if hit(r[k]) else 'M' for k in ['predictor_raw_eV','raw_eV','final_eV'])==name][:12]
            for name in ['HHM','HMH','HMM','MHM','MMH','HHH']},
        audit_problems=problems, parser_warnings=parser_warnings, changed_potential_records=pot_changes,
        protocols=[dict(stage=k[0],settings=json.loads(k[1]),n=v) for k,v in protocols.items()],
        pymatgen_version=importlib.metadata.version('pymatgen'),
        units='eV; original periodic structures unchanged; no new structural transformations',
        gap_verification='EIGENVAL occupation threshold 1e-8; JSON gaps retained for original XML precision; OUTCAR termination and ionic-accuracy markers checked')
    summary['eigenval_label_disagreements'] = {
        stage: [r['sample_id'] for r in rows if r.get(field) is not None
                and hit(r[field]) != hit(r[f'{stage}_eigenval_gap_eV'])]
        for stage, field in [('raw_scf','raw_eV'),('scf','final_eV'),('bands','bands_eV')]}
    summary['oszicar_verified_counts'] = {stage: sum(r.get(f'{stage}_oszicar_ok',False) for r in rows)
                                          for stage in ['raw_scf','scf']}
    summary['known_yield'] = sum(r['final_hit'] is True for r in rows) / 500
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    with (OUT/'samples.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=sorted(set().union(*(r.keys() for r in rows))))
        writer.writeheader();writer.writerows(rows)
    with (OUT/'target_hits.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=['sample_id','formula','predictor_raw_eV','raw_eV','final_eV','bands_eV'],extrasaction='ignore')
        writer.writeheader();writer.writerows(r for r in rows if r['final_hit'] is True)
    print(json.dumps({k:v for k,v in summary.items() if k not in ['protocols','examples','audit_problems','parser_warnings']},indent=2))
    print('AUDIT_PROBLEMS',json.dumps(problems[:15]),'total',len(problems))


if __name__ == '__main__':
    main()
