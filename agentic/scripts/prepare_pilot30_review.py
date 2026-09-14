"""Persist source-grounded assistant annotations; not an independent gold dataset."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
rows = list(csv.DictReader((ROOT/'data/derived/pilot30.csv').open()))
old = json.loads((ROOT/'evidence/pilot_sources.json').read_text())
cases = [c for c in old if c['excerpts']]


def add(label, url, doi, excerpts, event, note, role='primary_paper_excerpt', family=None):
    row = next(r for r in rows if r['record_id'].split('_')[0] == label)
    cases.append(dict(case_id=label, formula=row['record_id'].split('_',1)[1], record_ids=[row['record_id']],
        url=url, doi=doi, excerpts=excerpts, source_role=role, locator='Abstract or explicitly identified introductory passage; short excerpts',
        assistant_event_assessment=event, assistant_note=note, candidate_family=family,
        retrieved_date='2026-09-14', retrieval_method='web_search_or_open', structure_match_verified=False,
        normalization='Subscripts, math and whitespace normalized; snippets are not full-text verification.'))

add('0.639','https://www.nature.com/articles/s41467-023-41569-z','10.1038/s41467-023-41569-z',
 ['Mn2Au is a metallic collinear antiferromagnet with a high Néel temperature above 1000 K.'], 'neel',
 '1000 is a lower bound in this background statement, not an exact Curie value. Original measurement/extrapolation must be traced.', 'cited_background')
add('0.108','https://www.nature.com/articles/ncomms3892','10.1038/ncomms3892',
 ['AFM Mn compounds with Néel temperatures well above room temperature, namely Mn3Ir (TN=960 K)'], 'neel',
 'Cited comparison compound, not the material measured in this paper; trace reference 14.', 'cited_background', 'Cu3Au_type_candidate')
add('2.51','https://arxiv.org/abs/1407.6203','10.1103/PhysRevB.90.075109',
 ['Magnetic susceptibility measurements suggest antiferromagnetic (AFM) ordering of moments on divalent Eu ions near T_N=22K.'], 'neel',
 'Source reports near 22 K, not local 19 K. Cross-sample discrepancy, not an automatic correction.',family='AMnBi2_candidate')
add('0.65','https://www.sciencedirect.com/science/article/abs/pii/S0304885321007666',None,
 ['It is perhaps the best-known antiferromagnet, with a Néel temperature TN of 955 K'], 'neel',
 'Hematite has canted AFM order and a weak net moment; do not equate weak ferromagnetism with the target FM/FiM class.', 'cited_background', 'corundum_candidate')
add('1.495','https://www.sciencedirect.com/science/article/pii/0038109881905329','10.1016/0038-1098(81)90532-9',
 ['PrMn2Si2, NdMn2Si2, YMn2Si2 and YMn2Ge2','All were found to be antiferromagnetic with Néel points at 368, 380, 460 and 395 K respectively.'], 'neel',
 'Temperature must be linked by respective order; 460 belongs to YMn2Si2.',family='ThCr2Si2_candidate')
add('1.253','https://www.sciencedirect.com/science/article/pii/S0925838897004866','10.1016/S0925-8388(97)00486-6',
 ['CaCo2P2 and CeCo2P2 order antiferromagnetically below the Néel temperatures TN=113(2) K and TN=440(5) K, respectively.'], 'neel',
 'Preserve 440(5) K, assign Co sublattice only where source context supports it.',family='ThCr2Si2_candidate')
add('0.1024','https://www.osti.gov/servlets/purl/1772614','10.1103/PhysRevB.102.180403',
 ['unique magnetic transition at ~ 310 K caused by canted antiferromagnetic interactions between the neighboring Fe-Fe atoms'], 'neel',
 'Canted AFM, approximate 310 K; spontaneous moment does not establish a conventional ferromagnet.',family='BaFe2X4_chain_candidate')
add('1.281','https://www.sciencedirect.com/science/article/abs/pii/003810989090535J','10.1016/0038-1098(90)90535-J',
 ['YBaCuFeO5+δ','Below the antiferromagnetic ordering temperature TN = 446 K'], 'neel',
 'Paper concerns oxygen-nonstoichiometric phase; delta and Fe/Cu occupancy must match before using local structure.',family='layered_YBaCuFeO5_candidate')
add('0.836','https://advanced.onlinelibrary.wiley.com/doi/10.1002/admi.202400938','10.1002/admi.202400938',
 ['DyFeO3, a canted bulk antiferromagnet with a high Néel temperature (645 K)'], 'neel',
 'Bulk background 645 K differs from local 650 K; article also fixes 650 K as a fit parameter. Do not treat a fixed parameter as a fresh measurement.', 'cited_background','orthoferrite_candidate')
add('0.619','https://link.springer.com/article/10.1186/s40679-016-0019-9','10.1186/s40679-016-0019-9',
 ['previous reports of TN via magnetization measurements have been masked by the presence of a ferromagnetic MnAs impurity with a Curie temperature of 317 K'], 'unknown',
 '317 K is explicitly linked to MnAs impurity, NOT LaMnAsO. Quarantine target value, do not produce a negative Tc label.',family='1111_oxypnictide_candidate')
add('0.802','https://www.researchgate.net/publication/345127292_Low-temperature_synthesis_of_micro-_and_nano-crystalline_CuFeS2_polymorphs','10.1007/s42452-020-03729-4',
 ['antiferromagnetic ordering below its Néel temperature, TN = 823 K'], 'neel',
 'Original paper background on chalcopyrite phase, surfaced through indexed paper text; bulk/phase correspondence not verified.', 'cited_background_author_uploaded','chalcopyrite_candidate')
add('1.420','https://www.ncnr.nist.gov/staff/jeff/NCCOprb68%20144503%202003.pdf','10.1103/PhysRevB.68.144503',
 ['La2CuO4 and YBa2Cu3O6','420 K, respectively','antiferromagnetic'], 'neel',
 'Short background excerpts lack full value-order context. Use as retrieval lead; original YBa2Cu3O6 measurement needed.', 'cited_background','123_cuprate_candidate')
add('0.617','https://www.cryst.ehu.es/magndata/index.php?this_label=0.617','10.1002/(SICI)1521-3749(199901)625:1<31::AID-ZAAC31>3.0.CO;2-S',
 ['Transition Temperature: <295 K','Experiment Temperature: 11 K'], 'unknown',
 'Exact MAGNDATA record ID matched. Transition upper bound lost in CSV. Experiment temperature is NOT an event. Magnetic type not explicit in excerpt.', 'database_record')
add('1.292','https://www.sciencedirect.com/science/article/abs/pii/S0304885300009756',None,
 ['superconductivity (TC≈8 K) and magnetism (TN≈6 K)','below 5 K it coexists with a commensurate antiferromagnetic AFM ground structure.'], 'unknown',
 'Local 2 K not established as a transition; possible measurement/context mismatch. Do not replace it by 5 or 6 K without original record.',family='borocarbide_candidate')
add('0.909','https://ruj.uj.edu.pl/server/api/core/bitstreams/dda3ca05-1a5e-4955-a8dd-f3859cee9c2b/content','10.1016/j.jmmm.2020.167152',
 ['antiferromagnetic with the Néel temperatures ranging from 4.9 K for Er2PtGe6 up to 48 K for Tb2PdGe6.'], 'neel',
 'Original paper reports 4.9 K for Er2PtGe6; local 9 K is a conflict requiring original record correspondence.',family='Yb2PdGe6_candidate')
add('0.36','https://www.sciencedirect.com/science/article/pii/S0304885306018324','10.1016/j.jmmm.2006.10.607',
 ['rutile structure antiferromagnet NiF2 (S=1, TN=73 K)'], 'neel',
 'Canted rutile antiferromagnet; target event is not Curie.',family='rutile_candidate')
add('0.327','https://www.sciencedirect.com/science/article/abs/pii/0022459680905599','10.1016/0022-4596(80)90559-9',
 ['The magnetic Bragg peaks which appeared yield a ferromagnetic structure with the magnetic moments perpendicular to the tetragonal axis.'], 'unknown',
 'Ferromagnetic structure supported; this excerpt does not establish 9 K or the transition itself. Curie candidate pending temperature evidence.',family='CsFeF4_layer_candidate')
add('1.314','https://kups.ub.uni-koeln.de/5258/','https://kups.ub.uni-koeln.de/5258/',
 ['Unterhalb von 8 K setzt eine transversale Spindichtewelle','unterhalb von 6 K in eine helikale Struktur'], 'other',
 'Primary experimental thesis abstract: 8 K spin-density wave onset, 6 K helix; natural versus synthetic samples differ.', 'primary_thesis_abstract','pyroxene_candidate')
add('1.0.59','https://oiks.pnpi.spb.ru/articles/953','10.1103/PhysRevB.105.064416',
 ['Na2MnTeO6 experiences an antiferromagnetic order at TN = 5.5 K.'], 'neel',
 'P-31c (163) phase in source consistent with local SG, but coordinates not matched. 3R R-3 polytype is distinct; do not transfer its absence of order.',family='2H_Na2MnTeO6_candidate')
add('1.775','https://www.sciencedirect.com/science/article/abs/pii/S0038109802000613','10.1016/S0038-1098(02)00061-3',
 ['antiferromagnetic phase transition at TN=24.1(1) K.'], 'neel',
 'Single crystal source gives 24.1(1) K, while older powder work may give 27 K. Preserve sample-dependent discrepancy.',family='quadruple_perovskite_candidate')
add('1.68','https://www.ornl.gov/publication/polar-and-magnetic-layered-site-and-rock-salt-b-site-ordered-nalnfewo6-ln-la-nd',None,
 ['ordered antiferromagnetically below TN ≈ 25 K for NaLaFeWO6 and at ∼21 K for NaNdFeWO6.'], 'neel',
 '25 K belongs to La compound in this paper, ~21 K to Nd. Potential cross-compound attribution error; retain local value pending record match.',family='AAprimeFeWO6_candidate')

# Short quotations are bounded; use paraphrases only in the separately marked notes.
# An excerpt is a retrieval anchor, not certification that the local sample is identical.
for c in cases:
    c.setdefault('assistant_event_assessment','unknown')
    c.setdefault('assistant_note','Previously sourced diagnostic; requires independent review and structure matching.')
    c.setdefault('candidate_family',None)
    c['reviewer']='assistant_source_review_not_independent_gold'
    c['training_eligible']=False
    c['structure_match_verified']=False
    if c['case_id']=='nd_unit':
        c['assistant_note']='0.953 K conflicts with 953 K. Symbol Tc does not establish a Curie event.'
    if c['case_id']=='ce_events':
        c['assistant_event_assessment']='multiple_non_curie'
        c['candidate_family']='1111_oxypnictide_candidate'
    if c['case_id']=='eu_mn':
        c['assistant_event_assessment']='neel'
        c['candidate_family']='AMnBi2_candidate'
    if c['case_id']=='mn4n_curie':
        c['assistant_event_assessment']='curie_candidate'
        c['candidate_family']='antiperovskite_nitride_candidate'
    if c['case_id']=='1.314':
        c['doi']=None

covered={r for c in cases for r in c['record_ids']}
for r in rows:
    if r['record_id'] not in covered:
        cases.append(dict(case_id=r['record_id'].split('_')[0],formula=r['record_id'].split('_',1)[1],record_ids=[r['record_id']],
            url=None,doi=None,excerpts=[],source_role='unresolved_primary_evidence',
            assistant_event_assessment='unknown',assistant_note='Bounded searches performed; only indirect/related-compound leads found. Original MAGNDATA page unavailable. No label inferred.',
            candidate_family=None,structure_match_verified=False,training_eligible=False,reviewer='assistant_source_review_not_independent_gold'))
(ROOT/'evidence/pilot30_sources_v2.json').write_text(json.dumps(cases,ensure_ascii=False,indent=2))
print('records',len(rows),'cases',len(cases),'source_covered_records',len(covered),'unresolved',len(rows)-len(covered))
