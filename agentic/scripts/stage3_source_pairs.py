"""Preserve original 1962 observations and source-derived reference cells separately."""
import csv
import hashlib
import json
from pathlib import Path
from stage3_nitride_feasibility import ROOT,pair_gate


def reference_cif(formula,a,corner,center):
    sites=[(corner,0,0,0),('Mn',0,.5,.5),('Mn',.5,0,.5),('Mn',.5,.5,0),(center,.5,.5,.5)]
    return '\n'.join(['# Source-derived reference cell; NOT a measured mcif or a generated discovery.',
        '# DOI 10.1103/PhysRev.125.1893, Tables I-II and section III.',
        'data_'+formula,"_symmetry_space_group_name_H-M 'P 1'",'_symmetry_Int_Tables_number 1',
        *[f'_cell_length_{axis} {a}' for axis in 'abc'],
        *[f'_cell_angle_{axis} 90' for axis in ['alpha','beta','gamma']],
        f"_chemical_formula_structural {formula}", 'loop_',
        '_symmetry_equiv_pos_site_id','_symmetry_equiv_pos_as_xyz',"1 'x, y, z'",'loop_',
        ' _atom_site_type_symbol',' _atom_site_label',' _atom_site_symmetry_multiplicity',
        ' _atom_site_fract_x',' _atom_site_fract_y',' _atom_site_fract_z',' _atom_site_occupancy',
        *[f'{el} {el}{i} 1 {x} {y} {z} 1' for i,(el,x,y,z) in enumerate(sites)]])+'\n'


def main():
    out=ROOT/'results/stage3';structures=ROOT/'data/curation/stage3_reference_structures';structures.mkdir(exist_ok=True)
    obs=[]
    for formula,a,t,order,atoms in [('Mn4N',3.865,745,'FiM',5),('Mn4N0.92',3.855,750,'FiM',123),
        ('Mn4N0.8',3.836,778,'FiM',24),('Mn4N0.75C0.25',3.865,850,'FiM',20),('Mn3AlC',3.869,272,'FM',5)]:
        obs.append(dict(observation_id=formula+'_1962',formula=formula,lattice_a_angstrom=a,temperature_k=t,
            reported_temperature_heading='Neel temperature',normalized_event='curie',order_below=order,
            event_classification_basis='Tables I,V and section III magnetic configurations; historical heading retained',
            uncertainty_k=None,qualifier='reported_value',sample_form='powder',
            source='https://doi.org/10.1103/PhysRev.125.1893',pdf_page=2,printed_page=1894,table='II',
            transcription='assistant_visual_checked_table',minimum_ordered_atoms_for_nominal_composition=atoms,
            sample_limit='Composition/impurity uncertainty and historical magnetic models remain; no certified confidence interval.',training_eligible=False))
    (ROOT/'data/curation/stage3_1962_observations.json').write_text(json.dumps(obs,indent=2))
    cells=[]
    for formula,a,corner,center in [('Mn4N',3.865,'Mn','N'),('Mn3AlC',3.869,'Al','C')]:
        path=structures/(formula+'_1962_reference.cif');path.write_text(reference_cif(formula,a,corner,center))
        cells.append(dict(formula=formula,path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            provenance='Constructed reference cell from paper site assignments and lattice parameter; not downloaded experimental CIF.',
            magnetic_moments_included=False,training_eligible=False))
    (structures/'manifest.json').write_text(json.dumps(cells,indent=2))
    pairs=[]
    for winner,loser in [(obs[3],obs[0]),(obs[0],obs[4]),(obs[2],obs[0]),(obs[1],obs[0])]:
        p=dict(pair_id=winner['observation_id']+'_gt_'+loser['observation_id'],winner=winner['formula'],loser=loser['formula'],
            winner_event='curie',loser_event='curie',winner_temperature_k=winner['temperature_k'],loser_temperature_k=loser['temperature_k'],
            same_paper=True,evidence_kind='experimental_report',comparison_margin_k=50,
            configured_max_atoms=100,minimum_ordered_atoms=max(winner['minimum_ordered_atoms_for_nominal_composition'],loser['minimum_ordered_atoms_for_nominal_composition']),
            local_structure_mapping_verified=False,winner_split='external_unassigned',loser_split='external_unassigned',
            source='https://doi.org/10.1103/PhysRev.125.1893',training_eligible=False)
        p.update(pair_gate(p));pairs.append(p)
    (out/'1962_candidate_pairs.json').write_text(json.dumps(pairs,indent=2))
    conflict=dict(record_id='0.275_Mn3AlN',local_formula='Mn3AlN',local_temperature_k=272,local_lattice_a=3.869,
        cited_paper_formula='Mn3AlC',paper_temperature_k=272,paper_lattice_a=3.869,
        database_url='https://www.cryst.ehu.eus/magndata/index.php?this_label=0.275',
        paper_doi='10.1103/PhysRev.125.1893',finding='C/N identity conflict between cited original article and local/database record',
        action='Quarantine from new preference construction; original file retained; no silent C-for-N replacement.',
        independent_review_complete=False)
    (out/'identity_conflict.json').write_text(json.dumps(conflict,indent=2))
    # Sensitivity records are descriptive separations, not confidence probabilities.
    sensitivity=[dict(margin_k=m,accepted=sum(p['winner_temperature_k']-p['loser_temperature_k']>m for p in pairs)) for m in [25,50,100]]
    (out/'margin_sensitivity.json').write_text(json.dumps(sensitivity,indent=2))
    print('original_observations',len(obs),'candidate_pairs',len(pairs),'pass_50K',sum(p['literature_comparison_supported'] for p in pairs),'reference_cells',len(cells))

if __name__=='__main__':main()
