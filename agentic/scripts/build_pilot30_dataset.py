"""Build an explicitly prototype-level experimental preference pilot; never sample-exact CIFs."""
import csv, hashlib, json, re
from collections import Counter
from pathlib import Path
from pymatgen.core import Composition, Structure, Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.structure_matcher import StructureMatcher
ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'data/pilot30_v1'; E=ROOT/'evidence/pilot30_v1'; R=ROOT/'results/pilot30_v1'

def write(path,obj): path.write_text(json.dumps(obj,ensure_ascii=False,indent=2))
def source(key,doi,group,locator,excerpts):
 return dict(source_id=key,doi=doi,source_group=group,locator=locator,excerpts=excerpts,
             transcription='Codex source-checked transcription; not independent human gold')

def main():
 for p in (D,E,R,D/'structures'):p.mkdir(parents=True,exist_ok=True)
 sources=[
 source('laves_2008','10.1103/PhysRevB.77.125132','Zaragoza_Laguna_Chaboy','PDF page 2, Table I; page 1 experimental',[
 'Table I: TC is the Curie temperature obtained as the inflection point of the experimental M(T) curves. PM means paramagnetic. Transcribed Sample | TC (K):\nYFe2 | 541\nGdFe2 | 793\nGdCo2 | 400\nGdAl2 | 164\nTbFe2 | 653\nTbCo2 | 235\nTbAl2 | 109\nDyFe2 | 628\nDyCo2 | 150\nDyAl2 | 58\nHoFe2 | 606\nHoCo2 | 78\nHoAl2 | 28\nErFe2 | 582\nErCo2 | 32\nErAl2 | 13\nLuFe2 | 582',
 'Binary RAl2, RFe2, RCo2 Laves intermetallics. Macroscopic magnetization measured with a SQUID and high-temperature Faraday balance. Table I does not give Tc uncertainties. YCo2 and LuCo2 are PM; mixed Lu(Al0.1Co0.9)2 is excluded.']),
 source('ral2_2009','10.1107/S0909049509009807','Zaragoza_Laguna_Chaboy','Original PDF p407, Table 1, Results; archived web_original_sources.json',[
 'Both RAl2 and substituted series crystallize in cubic Laves C15 MgCu2 structure. Rietveld refinement finds all samples single phase. RAl2 compounds order ferromagnetically. Table 1 row: SmAl2 | a=7.942 angstrom | V=501.0 angstrom^3 | Curie temperature TC=64 K | M5T=0.24 muB/f.u. Uncertainty on TC is not reported.']),
 source('prnd_2005','10.1063/1.1876575','UNICAMP_Carvalho_Campoy','Original p083905-2, Results, Fig 1/2; public author full text archived web_original_sources.json',[
 'X-ray powder diffraction for both compounds shows only the desired C-15 pure phase, confirmed by metallographic analyses. Figure 2 shows magnetization versus temperature for PrAl2 and NdAl2 under H=100 Oe. The magnetic transition temperatures are 32.5 and 77 K, respectively. The text identifies the magnetic entropy maxima with Curie temperatures in these normal ferromagnetic systems. These are separately prepared compounds, not the later mechanically mixed composite. No Tc uncertainty reported.']),
 source('co2fesi_2005','10.1103/PhysRevB.73.094422','Mainz_Felser','arXiv cond-mat/0506729v4, structure section and high-T magnetometry',[
 'Co2FeSi: L21 structure, experimental lattice parameter 5.64 angstrom. Co/Fe DO3 disorder excluded; small (<10%) Fe/Si B2 disorder cannot be excluded. High-temperature magnetization measured using VSM at 0.1 T. The ferromagnetic Curie temperature is TC=(1100 +/- 20) K. Half-metallic spin-channel gap is not a conventional band gap.']),
 source('co2ti_2009','arXiv:0907.3562','Mainz_Felser','Original page 2, sample preparation and magnetometry',[
 'All Co2TiZ samples exhibit L21 structure. Lattice parameters at 300 K: Co2TiSi 5.733 angstrom; Co2TiGe 5.819 angstrom; Co2TiSn 6.066 angstrom. Curie temperature TC from magnetization measured at 10 mT: Co2TiSi 380 K, Co2TiGe 380 K, Co2TiSn 355 K. Tc uncertainty not specified. Resistivity behavior is metallic below Tc. Transport-peak estimates (370,350,360 K respectively) differ from magnetometry; do not replace the magnetic values.']),
 source('mn4n_1962','10.1103/PhysRev.125.1893','Westinghouse_1962','Previously audited original p1894 Table II and magnetic-order context',[
 'Transcribed source rows: Mn4N | a=3.865 angstrom | magnetic transition=745 K; Mn3AlC | a=3.869 angstrom | magnetic transition=272 K. Mn4N is ferrimagnetic, Mn3AlC ferromagnetic. Historical table heading says Neel temperature; explicit ferri/ferromagnetic context makes these ordering events relevant, not antiferromagnetic TN labels. No Tc uncertainty reported. Mn3AlN is a different compound and must not replace Mn3AlC.']),
 source('euo_eus_1976','10.1103/PhysRevB.14.4897','Brookhaven_Riso_Passell','Original PDF pages 1 and 4, printed pp4897 and 4900, experimental critical-scattering peaks',[
 'EuO and EuS are europium chalcogenide magnetic insulators with fcc Eu sublattices. Enriched 153Eu polycrystalline samples. Critical scattering peaks identify the Curie temperatures. Body text: EuO TC=69.15 +/- 0.05 K; EuS TC=16.57 +/- 0.02 K. Abstract gives EuS +/-0.01 K; retain the larger body uncertainty and flag the disagreement. Neutron and X-ray powder patterns show no evidence of other europium compounds.']),
 source('comn_2021','10.1016/j.physb.2020.412761','Japan_Saito_NishioHamane','Original publisher abstract and Results preview; archived web_original_sources.json',[
 'Mold-cast Co2MnGa and Co2MnSi annealed at 1073 K for 24 h. XRD specimens basically composed of Heusler L21 structure; structures further examined by TEM, but the detailed TEM result is not available in the public preview. Co2MnGa: Curie temperature 700 K; Co2MnSi: Curie temperature 1015 K. Both confirmed ferromagnetic. Room-temperature resistivity respectively 1.30 and 0.198 microohm m. No Tc uncertainty reported. L21 Co2MnGa transforms into B2 at 1200 K, above its Tc; Co2MnSi L21 stable to melting at 1450 K.'])]
 write(E/'sources.json',sources)
 rows=[]
 def add(formula,tc,src,proto,unc=None,a=None,notes=''):
  rows.append(dict(material_id=formula,formula=formula,tc_k=tc,tc_uncertainty_k=unc,source_id=src,prototype=proto,experimental_a_angstrom=a,notes=notes))
 for f,t in [('YFe2',541),('GdFe2',793),('GdCo2',400),('GdAl2',164),('TbFe2',653),('TbCo2',235),('TbAl2',109),('DyFe2',628),('DyCo2',150),('DyAl2',58),('HoFe2',606),('HoCo2',78),('HoAl2',28),('ErFe2',582),('ErCo2',32),('ErAl2',13),('LuFe2',582)]: add(f,t,'laves_2008','C15')
 add('SmAl2',64,'ral2_2009','C15',a=7.942)
 add('PrAl2',32.5,'prnd_2005','C15');add('NdAl2',77,'prnd_2005','C15')
 add('Co2FeSi',1100,'co2fesi_2005','L21',20,5.64,'Below 10% B2 disorder not excluded; sensitivity exclusion required.')
 for f,t,a in [('Co2TiSi',380,5.733),('Co2TiGe',380,5.819),('Co2TiSn',355,6.066)]:add(f,t,'co2ti_2009','L21',a=a,notes='Alternative transport-peak Tc differs; retain magnetometry event.')
 add('Mn4N',745,'mn4n_1962','antiperovskite',a=3.865);add('Mn3AlC',272,'mn4n_1962','antiperovskite',a=3.869)
 add('EuO',69.15,'euo_eus_1976','rocksalt',.05,notes='Experimental insulator; MP20 computed gap=0 contradicts qualitative electrical evidence. No gap supervision.')
 add('EuS',16.57,'euo_eus_1976','rocksalt',.02,notes='Body +/-0.02 versus abstract +/-0.01 K; larger retained. Experimental insulator; MP20 gap=0 unsuitable as electrical truth.')
 add('Co2MnGa',700,'comn_2021','L21',notes='Publisher preview supports dominant L21, not exact refined occupancy; source-held-out only.')
 add('Co2MnSi',1015,'comn_2021','L21',notes='Publisher preview supports dominant L21, not exact refined occupancy; source-held-out only.')
 local=json.loads((R/'local_candidates.json').read_text());sm={s['source_id']:s for s in sources}
 # Recover previously accepted Mn reference mappings without modifying source data.
 for split in ('train','val','test'):
  with (ROOT.parent/f'data/mp_20/{split}.csv').open() as h:
   for x in csv.DictReader(h):
    if x['material_id'] in ('mp-505622','mp-4593'):
     s=Structure.from_str(x['cif'],fmt='cif');local.append(dict(id=x['material_id'],formula=x['pretty_formula'],split=split,sg=SpacegroupAnalyzer(s,.1).get_space_group_number(),a=s.lattice.a,band_gap=float(x['band_gap']),cif=x['cif']))
 for r in rows:
  f=r['formula'];expected={'C15':227,'L21':225,'rocksalt':225,'antiperovskite':221}[r['prototype']]
  matches=[x for x in local if Composition(x['formula']).reduced_composition==Composition(f).reduced_composition and x['sg']==expected]
  if f=='GdCo2' and not matches:
   m=dict(a=13.6988*0.529177210903,id=None,split=None,band_gap=None,cif=None)
   r['notes']='GdCo2 theoretical equilibrium a=13.6988 Bohr reported on PDF p6 of DOI 10.1016/j.jallcom.2022.166116; separate source, not measured lattice of 2008 sample. No MP20 gap label.'
   r['structure_reference_doi']='10.1016/j.jallcom.2022.166116'
  else:
   assert len(matches)==1,(f,[x['id'] for x in matches]);m=matches[0]
  a=r['experimental_a_angstrom'] or m['a']
  if r['prototype']=='C15':
   species=list(Composition(f).as_dict());s=Structure.from_spacegroup(227,Lattice.cubic(a),species,[[0,0,0],[.625,.625,.625]]).get_primitive_structure()
  elif r['prototype']=='L21':
   elems=list(Composition(f).as_dict());other=[x for x in elems if x!='Co'];s=Structure.from_spacegroup(225,Lattice.cubic(a),['Co']+other,[[.25]*3,[0]*3,[.5]*3]).get_primitive_structure()
  elif r['prototype']=='rocksalt':s=Structure.from_spacegroup(225,Lattice.cubic(a),list(Composition(f).as_dict()),[[0]*3,[.5]*3]).get_primitive_structure()
  else:s=Structure.from_file(ROOT/f'data/curation/stage3_reference_structures/{f}_1962_reference.cif')
  assert s.is_ordered and s.is_valid() and len(s)<=20
  assert s.composition.reduced_composition==Composition(f).reduced_composition
  assert SpacegroupAnalyzer(s,.01).get_space_group_number()==expected
  matcher=StructureMatcher(ltol=.2,stol=.3,angle_tol=5,scale=True)
  if m['cif'] is not None:assert matcher.fit(s,Structure.from_str(m['cif'],fmt='cif')),f
  path=D/'structures'/f'{f}.cif';s.to(filename=str(path),fmt='cif')
  split=('validation' if r['source_id']=='prnd_2005' else 'test' if r['source_id']=='comn_2021' else 'electrical_control' if r['source_id']=='euo_eus_1976' else 'train')
  r.update(source_group=sm[r['source_id']]['source_group'],doi=sm[r['source_id']]['doi'],split=split,spacegroup=expected,natoms=len(s),structure_path=str(path.relative_to(ROOT)),structure_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    structure_scope='Ordered parent-prototype reference, not sample-exact refinement or magnetic CIF',cell_scale_source=('same_paper_experimental' if r['experimental_a_angstrom'] else 'external_DFT_reference' if f=='GdCo2' else 'MP20_DFT_reference'),local_mp_id=m['id'],local_mp_split=m['split'],local_mp_band_gap_ev=m['band_gap'],
    experimental_band_gap_ev=None,electrical_class=('insulator' if f in ('EuO','EuS') else 'metallic_transport' if f.startswith('Co2Ti') or f.startswith('Co2Mn') else 'not_directly_measured_in_Tc_excerpt'),
    comparison_scope='Composition and ordered prototype preference; not isolated geometry effect',full_occupancy_reference=True,pretraining_unseen_claim=False,minimum_distance_angstrom=min(n.nn_distance for nns in s.get_all_neighbors(4) for n in nns))
 write(D/'materials.json',rows)
 with (D/'endpoints.csv').open('w') as h:
  w=csv.DictWriter(h,fieldnames=['material_id','pretty_formula','cif','tc']);w.writeheader()
  for r in rows:w.writerow(dict(material_id=r['material_id'],pretty_formula=r['formula'],cif=(ROOT/r['structure_path']).read_text(),tc=r['tc_k']))
 by={r['material_id']:r for r in rows};pairs=[];degree=Counter()
 def pair(w,l,split,margin=50,buffer=25):
  a,b=by[w],by[l];assert a['prototype']==b['prototype']
  u=a['tc_uncertainty_k'] if a['tc_uncertainty_k'] is not None else buffer
  v=b['tc_uncertainty_k'] if b['tc_uncertainty_k'] is not None else buffer
  gap=a['tc_k']-b['tc_k']-u-v;assert gap>margin,(w,l,gap)
  pairs.append(dict(pair_id=f'{w}__over__{l}',winner=w,loser=l,split=split,source_groups=sorted({a['source_group'],b['source_group']}),source_ids=sorted({a['source_id'],b['source_id']}),prototype=a['prototype'],delta_tc_k=a['tc_k']-b['tc_k'],sensitivity_gap_k=gap,margin_k=margin,unreported_uncertainty_buffer_k=buffer,
    buffer_is_not_measurement_uncertainty=True,high_tc_anchor=a['tc_k']-u>=500,independent_experiment=False,causal_structure_claim=False))
  degree[w]+=1;degree[l]+=1
 # Degree-balanced, large-gap C15 edges; selection uses source labels, never model scores.
 c15=[r for r in rows if r['prototype']=='C15' and r['split']=='train']
 possible=[(a['material_id'],b['material_id']) for a in c15 for b in c15 if a['tc_k']>=500 and b['tc_k']<500 and a['tc_k']-b['tc_k']>100]
 for _ in range(43):
  w,l=min(possible,key=lambda p:(degree[p[0]]+degree[p[1]],max(degree[p[0]],degree[p[1]]),p));possible.remove((w,l));pair(w,l,'train')
 pair('Mn4N','Mn3AlC','train')
 for f in ('Co2TiSi','Co2TiGe','Co2TiSn'):pair('Co2FeSi',f,'train')
 pair('NdAl2','PrAl2','validation',margin=10,buffer=10)
 pair('Co2MnSi','Co2MnGa','test')
 pair('EuO','EuS','electrical_control',buffer=0)
 assert len(rows)==30 and len(pairs)==50 and all(degree[r['material_id']]>0 for r in rows)
 for split in ('train','validation','test','electrical_control'):
  groups={g for p in pairs if p['split']==split for g in p['source_groups']}
  other={g for p in pairs if p['split']!=split for g in p['source_groups']};assert not groups&other
 write(D/'pairs.json',pairs)
 write(R/'dataset_audit.json',dict(materials=len(rows),comparisons=len(pairs),source_groups=len({r['source_group'] for r in rows}),papers=len(sources),split_pairs=dict(Counter(p['split'] for p in pairs)),split_materials=dict(Counter(r['split'] for r in rows)),maximum_endpoint_degree=max(degree.values()),all_materials_have_comparison=True,all_ordered_reference_cifs_valid=True,mp20_reference_prototype_matches=29,external_literature_reference_structures=1,source_group_overlap_between_splits=False,
    sample_exact_experimental_cifs_claimed=0,pretraining_unseen_evaluation=False,unknown_experimental_gap_labels=30,high_tc_train_pairs=sum(p['split']=='train' and p['high_tc_anchor'] for p in pairs),
    excluded_candidates={'RNi2':'Possible vacancy superstructures and sample phase fractions; not treated as ordered C15 labels','FeNiN':'Known N deficiency unresolved'},limitations=['Only 3 training source groups; 43/47 training edges from one group. Use source-balanced objective.','Validation/test each one pair; no generalization or statistical-significance claim.','29 of 30 reference compositions occur in current MP20; new experimental source holdout is not unseen-structure holdout.','Prototype-level labels permit ideal parent references, not sample-exact quantitative Tc regression.','All 30 experimental numerical band gaps are missing; EuO/EuS MP20 zero gaps conflict with insulator evidence.']))
 print((R/'dataset_audit.json').read_text())
if __name__=='__main__':main()
