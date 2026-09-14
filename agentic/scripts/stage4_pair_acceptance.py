"""Offline source-reference pair acceptance; no training or source-data mutation."""
import csv
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import sys
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
os.environ.setdefault('PROJECT_ROOT', str(ROOT.parent))
from pymatgen.io.cif import CifParser
from pymatgen.analysis.structure_matcher import StructureMatcher, ElementComparator
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse(text):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        parser = CifParser(io.StringIO(text))
        structures = parser.parse_structures(primitive=False, check_occu=True, on_error='raise')
    if len(structures) != 1:
        raise ValueError('Expected exactly one structure')
    return structures[0], sorted(set(list(parser.warnings) + [str(w.message) for w in caught]))


def assign_pair(matches, parse_failures):
    """Fail closed on incomplete audit or any held-out endpoint duplicate."""
    if parse_failures:
        return 'blocked_incomplete_audit'
    if any(m['split'] in ('val', 'test') for entries in matches.values() for m in entries):
        return 'blocked_heldout_overlap'
    return 'train_extension_pipeline_only'


def main():
    out = ROOT/'results/stage4'; out.mkdir(parents=True, exist_ok=True)
    data = ROOT/'data/curation/stage4_pair'; data.mkdir(parents=True, exist_ok=True)
    raw = ROOT/'data/raw/magndata'
    originals = [ROOT.parent/'data/magndata'/p.name for p in raw.iterdir() if p.is_file()]
    before = {str(p): digest(p) for p in originals}
    cells = json.loads((ROOT/'data/curation/stage3_reference_structures/manifest.json').read_text())
    obs = {x['formula']: x for x in json.loads((ROOT/'data/curation/stage3_1962_observations.json').read_text())}
    refs, validation = {}, []
    for cell in cells:
        path = ROOT/cell['path']; assert digest(path) == cell['sha256']
        s, messages = parse(path.read_text()); refs[cell['formula']] = s
        assert s.is_ordered and len(s) == 5 and s.is_valid()
        assert all(abs(sum(site.species.values())-1) < 1e-8 for site in s)
        symmetry = []
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            for tol in (0.001, 0.01, 0.1):
                analyzer = SpacegroupAnalyzer(s, symprec=tol, angle_tolerance=5)
                symmetry.append(dict(symprec_angstrom=tol, angle_tolerance_deg=5,
                                     number=analyzer.get_space_group_number(), symbol=analyzer.get_space_group_symbol()))
        validation.append(dict(formula=cell['formula'], composition=s.composition.as_dict(), natoms=len(s),
            periodic=True, coordinates='fractional', lattice_parameters_angstrom_degrees=list(s.lattice.parameters),
            minimum_distance_angstrom=min(n.nn_distance for nn in s.get_all_neighbors(4) for n in nn),
            ordered=s.is_ordered, oxidation_states_assigned=False, symmetry=symmetry,
            parser_warnings=messages, symmetry_warnings=sorted(set(str(w.message) for w in caught)), **{k:cell[k] for k in ('path','sha256','provenance')}))
    matcher = StructureMatcher(ltol=0.01, stol=0.05, angle_tol=1, primitive_cell=True,
                               scale=False, attempt_supercell=False, comparator=ElementComparator())
    matches = {f: [] for f in refs}; failures=[]; audit=[]; count=0
    for split in ('train','val','test'):
        with (raw/(split+'.csv')).open() as stream:
            for row in csv.DictReader(stream):
                count += 1
                try:
                    s, messages = parse(row['cif'])
                except Exception as exc:
                    failures.append(dict(split=split, material_id=row['material_id'], error=str(exc)))
                    continue
                audit.append(dict(material_id=row['material_id'], split=split, parser_warnings=messages,
                                  actual_composition=s.composition.as_dict(), metadata_formula=row['pretty_formula']))
                for formula, ref in refs.items():
                    if s.composition.fractional_composition == ref.composition.fractional_composition and matcher.fit(ref, s):
                        matches[formula].append(dict(split=split, material_id=row['material_id'], tc=float(row['tc']),
                                                     cif_sha256=hashlib.sha256(row['cif'].encode()).hexdigest()))
    assignment=assign_pair(matches, failures)
    assert assignment == 'train_extension_pipeline_only', assignment
    rows=[]
    for cell in cells:
        formula=cell['formula']; observation=obs[formula]
        rows.append(dict(material_id=formula+'_1962_reference', pretty_formula=formula,
                         cif=(ROOT/cell['path']).read_text(), tc=observation['temperature_k']))
    csv_path=data/'endpoints.csv'
    with csv_path.open('w') as stream:
        writer=csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    # Use the actual project graph preprocessing and PyG batch interface. Newly created cache only.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        from cgdit.pl_data.dataset import CrystDataset
        from torch_geometric.data import Batch
        import torch
        cache=out/'endpoint_graphs.pt'
        if cache.exists(): cache.unlink()
        ds=CrystDataset(name='Stage4 pipeline acceptance only', path=str(csv_path), save_path=str(cache),
            prop='tc', prop_list=['tc'], niggli=False, primitive=False, graph_method='crystalnn',
            preprocess_workers=1, lattice_scale_method='scale_length', tolerance=0.1,
            use_space_group=False, use_pos_index=False)
        graphs=[ds[i] for i in range(len(ds))]; batch=Batch.from_data_list(graphs)
    graph_report=[]
    for row, g in zip(rows, graphs):
        assert g.num_nodes == 5 and g.num_bonds > 0 and g.edge_index.shape[0] == 2
        assert g.edge_index.min() >= 0 and g.edge_index.max() < 5
        assert torch.isfinite(g.frac_coords).all() and g.y.item() == row['tc']
        assert sorted(g.atom_types.tolist()) == sorted(refs[row['pretty_formula']].atomic_numbers)
        graph_report.append(dict(material_id=row['material_id'], nodes=g.num_nodes, directed_edges=g.num_bonds,
                                 atom_numbers=g.atom_types.tolist(), tc_kelvin=g.y.item()))
    assert batch.num_nodes == 10 and batch.num_graphs == 2
    pair=dict(pair_id='Mn4N_1962_gt_Mn3AlC_1962', winner_id=rows[0]['material_id'], loser_id=rows[1]['material_id'],
        winner_row=0, loser_row=1, winner_temperature_k=745, loser_temperature_k=272, delta_k=473,
        comparison_margin_k=50, evidence_kind='experimental_report', normalized_event='curie',
        original_heading='Neel temperature', order_below={'winner':'FiM','loser':'FM'},
        event_basis='1962 Tables I,V and section III; historical heading retained',
        source_doi='10.1103/PhysRev.125.1893', source_pdf='evidence/stage3/mn4n_1962.pdf',
        source_sha256=digest(ROOT/'evidence/stage3/mn4n_1962.pdf'), table='II', printed_page=1894,
        structure_kind='source_reconstructed_nuclear_reference', magnetic_moments_included=False,
        split=assignment, pipeline_ready=True, production_training_eligible=False,
        limitation='One assistant-source-reviewed comparison; idealized reference cells, no uncertainty interval or independent generalization evaluation.',
        endpoints_csv='data/curation/stage4_pair/endpoints.csv', endpoints_sha256=digest(csv_path))
    (data/'pairs.jsonl').write_text(json.dumps(pair)+'\n')
    checks=[dict(path=p, unchanged=digest(Path(p))==h, sha256=h) for p,h in before.items()]
    assert all(c['unchanged'] for c in checks)
    results={'structure_validation':validation, 'split_audit':dict(records_scanned=count, matches=matches,
        parse_failures=failures, assignment=assignment, matcher=dict(ltol=0.01, stol=0.05, angle_tol=1,
        scale=False, primitive_cell=True, attempt_supercell=False, comparator='ElementComparator'),
        excluded_record='0.275_Mn3AlN', exclusion='Unresolved source C/N identity conflict; original data untouched'),
        'parser_audit':audit, 'graph_validation':dict(graphs=graph_report, batch_nodes=batch.num_nodes,
        batch_graphs=batch.num_graphs, warnings=sorted(set(str(w.message) for w in caught))),
        'source_integrity':checks,
        'summary':dict(pipeline_pairs=1, production_training_pairs=0, model_training_run=False, api_calls=0,
        versions={m:importlib.metadata.version(m) for m in ['pymatgen','spglib','torch','torch-geometric']})}
    for name, result in results.items(): (out/(name+'.json')).write_text(json.dumps(result, indent=2))
    print(json.dumps(dict(summary=results['summary'], matches=matches, graphs=graph_report), indent=2))

if __name__ == '__main__': main()
