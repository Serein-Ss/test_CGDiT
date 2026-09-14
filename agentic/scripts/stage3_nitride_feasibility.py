"""Audit explicit small-cell antiperovskite geometry and literature pair feasibility."""
import csv
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
import shlex

ROOT=Path(__file__).resolve().parents[1]


def ordered_atom_lower_bound(x):
    """Five sites/formula; nominal x only, not a measured occupancy determination."""
    return 5*Fraction(str(x)).denominator


def inspect_explicit_cell(cif):
    lines=cif.splitlines()
    if not any(re.match(r'_symmetry_Int_Tables_number\s+1$',line) for line in lines):
        return {'supported_explicit_p1':False,'template_match':False}
    lengths=[float(next(line.split()[1] for line in lines if line.startswith('_cell_length_'+axis))) for axis in 'abc']
    angles=[float(next(line.split()[1] for line in lines if line.startswith('_cell_angle_'+axis))) for axis in ['alpha','beta','gamma']]
    start=lines.index(' _atom_site_type_symbol')
    headers=[];sites=[]
    for line in lines[start:]:
        if line.strip().startswith('_atom_site_'):
            headers.append(line.strip())
        elif line.strip():
            vals=shlex.split(line)
            if len(vals)!=len(headers):break
            sites.append(dict(zip(headers,vals)))
    actual={tuple(float(s['_atom_site_fract_'+a])%1 for a in 'xyz'):s['_atom_site_type_symbol'] for s in sites}
    expected={(0.,0.,0.),(0.,.5,.5),(.5,0.,.5),(.5,.5,0.),(.5,.5,.5)}
    full=all(float(s['_atom_site_occupancy'])==1 for s in sites)
    match=(len(sites)==5 and len(actual)==5 and set(actual)==expected and actual[(.5,.5,.5)]=='N'
        and all(actual[p]!='N' for p in expected-{(.5,.5,.5)}) and full
        and max(lengths)-min(lengths)<1e-6 and all(abs(a-90)<1e-6 for a in angles))
    return dict(supported_explicit_p1=True,template_match=match,full_occupancy=full,
                cell_lengths_angstrom=lengths,sites=sites,note='Exact listed-cell motif check, not general structure matching or sample identity.')


def pair_gate(pair):
    reasons=[]
    if pair['winner_event']!='curie' or pair['loser_event']!='curie':reasons.append('non_curie_event')
    if pair['evidence_kind']!='experimental_report':reasons.append('not_experimental_report')
    if not pair['same_paper']:reasons.append('cross_source_conditions_unresolved')
    if pair['winner_temperature_k']-pair['loser_temperature_k']<=pair['comparison_margin_k']:
        reasons.append('insufficient_temperature_separation')
    literature_supported=not reasons
    if not pair['local_structure_mapping_verified']:reasons.append('structure_mapping_unverified')
    if pair['winner_split']!='train' or pair['loser_split']!='train':reasons.append('both_not_in_training_split')
    if pair['minimum_ordered_atoms']>pair['configured_max_atoms']:reasons.append('nominal_ordered_representation_exceeds_atom_limit')
    return dict(literature_comparison_supported=literature_supported,training_eligible=not reasons,reasons=reasons)


def main():
    folder=ROOT/'results/stage3';folder.mkdir(exist_ok=True)
    source=ROOT/'evidence/stage3/mn4n_2022.txt'
    text=source.read_text();pages=text.split('\f')
    # Capture bounded context from actual locally retrieved full text, not search snippets.
    anchors=[('Mn4N','Mn4N has a high Curie temperature', 'the inability'),
             ('Mn3.76Ga0.24N','3b. The larger intensity','The M-H curves')]
    cases=[]
    for formula,start,end in anchors:
        i=text.index(start)
        j=text.lower().find(end.lower(),i)
        if j<0:j=i+1100
        excerpt=' '.join(text[i:j].split())
        cases.append(dict(case_id=formula,formula=formula,excerpts=[excerpt],source_url='https://arxiv.org/abs/2203.09641',
            doi='10.1016/j.actamat.2022.118021',pdf_page_1based=text[:i].count('\f')+1,
            evidence_kind='experimental_report',input_sha256=hashlib.sha256(excerpt.encode()).hexdigest(),
            source_file_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),training_eligible=False))
    (ROOT/'evidence/stage3/api_cases.json').write_text(json.dumps(cases,ensure_ascii=False,indent=2))
    inventory=[]
    for split in ['train','val','test']:
        for r in csv.DictReader((ROOT/f'data/raw/magndata/{split}.csv').open()):
            elements=set(re.findall('[A-Z][a-z]?',r['pretty_formula']))
            if 'N' not in elements or not elements.intersection({'Mn','Fe','Co','Ni'}):continue
            record=dict(record_id=r['material_id'],formula=r['pretty_formula'],split=split,
                legacy_temperature_k=float(r['tc']),natoms=int(r['natoms']),cif_sha256=hashlib.sha256(r['cif'].encode()).hexdigest())
            # The large chemical filter includes nitrates/organics; motif check is deliberately narrower.
            if int(r['natoms'])==5:
                record['geometry']=inspect_explicit_cell(r['cif'])
            else:record['geometry']={'template_match':False}
            inventory.append(record)
    (folder/'inventory.json').write_text(json.dumps(inventory,indent=2))
    motif=[r for r in inventory if r['geometry']['template_match']]
    with (folder/'local_antiperovskite_candidates.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=['record_id','formula','split','legacy_temperature_k','natoms','cif_sha256']);w.writeheader()
        w.writerows({k:r[k] for k in w.fieldnames} for r in motif)
    # Local parent geometry matches the literature prototype, not necessarily the same experimental sample.
    pair=dict(pair_id='Mn4N_vs_Mn3.76Ga0.24N_2022',winner='Mn4N',loser='Mn3.76Ga0.24N',
        winner_event='curie',loser_event='curie',winner_temperature_k=780,loser_temperature_k=610,
        loser_compensation_temperature_k=408,comparison_margin_k=50,margin_note='Conservative analyst sensitivity parameter, not an experimental uncertainty.',
        evidence_kind='experimental_report',same_paper=True,doi='10.1016/j.actamat.2022.118021',
        source_url='https://arxiv.org/abs/2203.09641',source_pages=[4,7],
        local_structure_mapping_verified=False,winner_split='train',loser_split='external_unassigned',
        local_winner_record='0.274_Mn4N',local_winner_legacy_temperature_k=745,
        minimum_ordered_atoms=ordered_atom_lower_bound('0.24'),configured_max_atoms=100,
        nominal_composition_note='125 atoms is a lower bound for exact periodic full-occupancy representation of nominal x=0.24; it does not certify any atomic arrangement.',
        reviewer='assistant_fulltext_review_not_independent_gold')
    pair.update(pair_gate(pair));(folder/'literature_pair.json').write_text(json.dumps(pair,indent=2))
    summary=dict(broad_nitrogen_containing_records=len(inventory),local_motif_records=len(motif),
        train_motif_records=sum(r['split']=='train' for r in motif),
        train_motif_unique_formulas=len({r['formula'] for r in motif if r['split']=='train'}),
        literature_comparisons=1,training_eligible_pairs=int(pair['training_eligible']),
        configured_max_atoms=100,ga024_minimum_ordered_atoms=125,
        evidence_limit='Prototype correspondence demonstrated for local cells; experimental sample and disorder-to-ordered mapping unverified.')
    (folder/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary))

if __name__=='__main__':main()
