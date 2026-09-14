"""Local chemistry-screened overlap audit and FeNi vacancy feasibility."""
import ast
import csv
from fractions import Fraction
import hashlib
import importlib.metadata
import json
from math import lcm
from pathlib import Path
import re
from pymatgen.core import Composition,Structure,Lattice
from pymatgen.analysis.structure_matcher import StructureMatcher,ElementComparator
from stage4_pair_acceptance import parse
ROOT=Path(__file__).resolve().parents[1]


def ordered_lower_bound(counts):
    numbers=[Fraction(str(v)) for v in counts]
    units=lcm(*(v.denominator for v in numbers))
    return dict(formula_units=units,atoms=int(sum(numbers)*units))


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    out=ROOT/'results/stage8';out.mkdir(parents=True,exist_ok=True)
    refs={}
    for f in ['Mn4N','Mn3AlC']:
        refs[f],_=parse((ROOT/f'data/curation/stage3_reference_structures/{f}_1962_reference.cif').read_text())
    # Ideal nominal templates are used ONLY for overlap comparison, never assigned experimental Tc.
    positions=[[0,0,0],[0,.5,.5],[.5,0,.5],[.5,.5,0],[.5,.5,.5]]
    refs['Fe4N']=Structure(Lattice.cubic(3.797),['Fe','Fe','Fe','Fe','N'],positions)
    refs['Fe3NiN']=Structure(Lattice.cubic(3.786),['Ni','Fe','Fe','Fe','N'],positions)
    target_sets=[{'Fe','N'},{'Fe','Ni','N'},{'Mn','N'},{'Mn','Al','C'}]
    matchers={name:StructureMatcher(ltol=tol,stol=stol,angle_tol=angle,scale=scale,
        primitive_cell=True,attempt_supercell=False,comparator=ElementComparator())
        for name,tol,stol,angle,scale in [('strict',.01,.05,1,False),('relaxed',.2,.3,5,True)]}
    records=[];files=[];failures=[];metadata_failures=[];hits=[];cif_rows=[]
    for dataset in ['mp_20','magndata']:
        for split in ['train','val','test']:
            path=ROOT.parent/'data'/dataset/(split+'.csv');before=sha(path);count=0;selected=0
            with path.open() as stream:
                for row in csv.DictReader(stream):
                    count+=1
                    # Screen BOTH supplied element metadata and CIF formula metadata.
                    candidates=[]
                    try:candidates.append(set(ast.literal_eval(row['elements'])))
                    except (ValueError,SyntaxError,KeyError):pass
                    try:candidates.append(set(Composition(row['pretty_formula']).as_dict()))
                    except (ValueError,KeyError):pass
                    for text in re.findall(r'^_chemical_formula_(?:sum|structural)\s+(.+)$',row['cif'],re.M):
                        try:candidates.append(set(Composition(text.strip().strip("'\"")).as_dict()))
                        except ValueError:pass
                    if not candidates:
                        metadata_failures.append(dict(dataset=dataset,split=split,id=row['material_id']))
                    if candidates and not any(c in target_sets for c in candidates):continue
                    selected+=1
                    try:s,messages=parse(row['cif'])
                    except Exception as e:
                        failures.append(dict(dataset=dataset,split=split,id=row['material_id'],error=type(e).__name__));continue
                    if set(s.composition.as_dict()) not in target_sets:continue
                    record=dict(dataset=dataset,split=split,material_id=row['material_id'],
                        metadata_formula=row['pretty_formula'],actual_composition=s.composition.as_dict(),
                        ordered=s.is_ordered,natoms=len(s),parser_warnings=messages,
                        cif_sha256=hashlib.sha256(row['cif'].encode()).hexdigest(),
                        experimental_tc_transferred=False)
                    records.append(record)
                    cif_rows.append(dict(dataset=dataset,split=split,material_id=row['material_id'],cif=row['cif']))
                    for f,ref in refs.items():
                        if s.composition.fractional_composition != ref.composition.fractional_composition:continue
                        fits={name:bool(matcher.fit(ref,s)) for name,matcher in matchers.items()}
                        hits.append(dict(record,reference=f,matches=fits))
            assert before==sha(path)
            files.append(dict(path=str(path),sha256=before,rows=count,cif_parsed_after_screen=selected,unchanged=True))
            print(dataset,split,count,selected,flush=True)
    feasibility=[]
    for formula,ideal,deficient in [('Fe3.6Ni0.4N',[3.6,.4,1],[3.6,.4,.92]),('Fe3NiN',[3,1,1],[3,1,.92])]:
        feasibility.append(dict(formula=formula,ideal_counts=ideal,ideal_minimum=ordered_lower_bound(ideal),
            nitrogen_occupancy_scenario=.92,deficient_counts=deficient,deficient_minimum=ordered_lower_bound(deficient),
            scenario_caveat='8% report treated as exactly 0.92 ONLY for an arithmetic feasibility scenario; not an exact measured stoichiometry',
            ordered_realization_known=False,training_eligible=False))
    result=dict(scope='Chemistry screening via element/pretty_formula/CIF-formula metadata, then full CIF parsing and structure matching of candidates; not exhaustive all-CIF parsing',
        target_chemical_systems=[sorted(x) for x in target_sets],files=files,metadata_failures=metadata_failures,
        parse_failures=failures,records=records,reference_composition_hits=hits,feasibility=feasibility,
        tolerances=dict(strict=dict(ltol=.01,stol=.05,angle_tol=1,scale=False),relaxed=dict(ltol=.2,stol=.3,angle_tol=5,scale=True)),
        normalization='fractional composition; primitive_cell=True, attempt_supercell=False, ElementComparator',
        reference_caveat='Fe4N/Fe3NiN templates are ideal nominal full-occupancy cells for matching, not the deficient experimental samples; no Tc transfer',version=importlib.metadata.version('pymatgen'),api_calls=0,training_run=False)
    with (ROOT/'data/curation/stage8_structure_candidates.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=['dataset','split','material_id','cif']);writer.writeheader();writer.writerows(cif_rows)
    (out/'structure_audit.json').write_text(json.dumps(result,indent=2))
    (out/'feni_candidates.json').write_text(json.dumps([r for r in records if 'Fe' in r['actual_composition']],indent=2))
    (out/'vacancy_feasibility.json').write_text(json.dumps(feasibility,indent=2))
    print('records',len(records),'hits',[(h['dataset'],h['split'],h['material_id'],h['reference'],h['matches']) for h in hits],flush=True)

if __name__=='__main__':main()
