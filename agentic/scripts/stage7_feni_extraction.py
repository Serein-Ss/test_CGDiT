"""Bounded Intern-S1 extraction from a visually checked original experimental table."""
import hashlib
import json
import re
import time
from datetime import datetime, timezone
import requests
from run_evidence_pilot import ROOT, read_key
from run_pilot30 import PROMPT,check

SOURCE='https://harvest.aps.org/v2/journals/articles/10.1103/PhysRev.126.49/fulltext'
TABLE='''Assistant transcription checked against PDF page 2, printed page 50, Table II. Original heading: Magnetic data for (Fe, Ni)4N. Columns: Fe4N | (Fe3.6Ni0.4)N | (Fe3Ni)N.
a0 (±0.002 Å): 3.797 | 3.788 | 3.786.
tc (±10 K): 767 | 695 | 640.
The uncertainty is reported by the table; its confidence interpretation is not specified.'''
CONTEXT='''Assistant transcription checked against the same original paper: The abstract describes ferromagnetic Fe4N and the ordered structures of (Fe3.6Ni0.4)N and (Fe3Ni)N, in which Ni replaces corner Fe preferentially. The introduction describes Fe3NiN as a ferromagnetic arrangement of three equivalent face-center Fe. The experimental section reports powder nitrides prepared using ammonia and hydrogen, single-phase X-ray results with no trace of free Fe, nitrogen deficiency of 8% in the Ni-containing samples and less than 3% in Fe4N. The lattice parameters and magnetic properties are described as reasonably consistent with earlier work. No Tc-specific measurement procedure or uncertainty confidence level is supplied in these excerpts. A later discussion uses the historical wording Neel temperature; preserve that ambiguity alongside the explicit ferromagnetic context. Do not treat 300 K or 0 K saturation-moment measurement temperatures as transitions.'''


def robust_pair(winner,loser,margin=50):
    gap=winner['temperature_k']-winner['uncertainty_k']-(loser['temperature_k']+loser['uncertainty_k'])
    return dict(winner=winner['material_formula'],loser=loser['material_formula'],
        nominal_delta_k=winner['temperature_k']-loser['temperature_k'],interval_gap_k=gap,
        margin_k=margin,passes_margin=gap>margin,training_eligible=False,
        source_group='Westinghouse_FeNiN_1962_TableII',
        split='unassigned_source_group',reason='Nominal formulas; nitrogen deficiency; no structure or pretraining-overlap audit')


def main():
    ev=ROOT/'evidence/stage7';out=ROOT/'results/stage7'
    pdf=ev/'fe4n_1962.pdf';assert pdf.read_bytes().startswith(b'%PDF')
    sha=hashlib.sha256(pdf.read_bytes()).hexdigest()
    provenance=dict(doi='10.1103/PhysRev.126.49',url=SOURCE,pdf_sha256=sha,pdf_page=2,printed_page=50,
        table='II',retrieved_at=datetime.now(timezone.utc).isoformat(),
        transcription='Codex visual check of original table; not independent human gold',
        excerpts=[TABLE,CONTEXT],original_text_path='evidence/stage7/fe4n_1962.txt')
    (ev/'feni_source.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2))
    key=read_key();rows=[];observations=[];calls=0
    for formula,tc,a in [('Fe4N',767,3.797),('Fe3.6Ni0.4N',695,3.788),('Fe3NiN',640,3.786)]:
        case=dict(formula=formula,excerpts=[TABLE,CONTEXT])
        payload=dict(model='intern-s1',temperature=0,max_tokens=2048,thinking_mode=False,messages=[
            dict(role='system',content=PROMPT+' Parenthesized nominal formulas in the table correspond to target_formula. Extract reported tc using the explicit ferromagnetic context, preserve ±10 K as uncertainty. A quote may be the whole contiguous table block, including headers. Retain nitrogen deficiency in sample_conditions, and do not invent a Tc measurement method.'),
            dict(role='user',content=json.dumps(case,ensure_ascii=False))])
        digest=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest();path=out/('intern_'+digest+'.json')
        row=dict(formula=formula,input_sha256=digest,source_pdf_sha256=sha)
        try:
            if path.exists():response=json.loads(path.read_text());row['cache_hit']=True
            else:
                calls+=1
                r=requests.post('https://chat.intern-ai.org.cn/api/v1/chat/completions',headers={'Authorization':'Bearer '+key},json=payload,timeout=(15,130))
                row['http_status']=r.status_code;r.raise_for_status();response=r.json()
                path.write_text(json.dumps(response,ensure_ascii=False,indent=2).replace(key,'[REDACTED]'));row['cache_hit']=False;time.sleep(2.1)
            text=response['choices'][0]['message']['content'].strip()
            if text.startswith('```'):text=re.sub(r'^```(?:json)?\s*|\s*```$','',text)
            result=json.loads(text);flags=check(result,case)
            row.update(status='schema_and_quote_valid',extraction=result,semantic_flags=flags,usage=response.get('usage'),returned_model=response.get('model'))
            # Independent deterministic comparison with the visually transcribed row; not inferred from API output.
            events=result['events']
            accepted=len(events)==1 and events[0]['type']=='curie' and events[0]['temperature_k']==tc and events[0]['uncertainty_k']==10 and events[0]['qualifier']=='reported_value'
            row['matches_source_row']=accepted
            if accepted:
                observation=dict(events[0],observation_id=formula+'_1962_TableII',lattice_a_angstrom=a,
                    lattice_uncertainty_angstrom=.002,doi=provenance['doi'],pdf_sha256=sha,pdf_page=2,printed_page=50,
                    source_group='Westinghouse_FeNiN_1962_TableII',uncertainty_interpretation='Reported ±10 K; not specified as standard deviation or confidence interval',
                    nominal_formula=True,nitrogen_deficiency=('less than 3%' if formula=='Fe4N' else '8%'),
                    original_transition_symbol='tc',historical_neel_wording_elsewhere=True,
                    experimental_temperature_method='Not specified in extracted source context',
                    prior_work_overlap='Table cites agreement with earlier work; do not count cited/reprinted rows as independent observations',
                    review_status='API extraction checked against Codex visual transcription',training_eligible=False)
                observations.append(observation)
        except (requests.RequestException,ValueError,KeyError,IndexError,TypeError) as e:
            row.update(status='failed',error_type=type(e).__name__)
        rows.append(row)
        (ROOT/'data/curation/stage7_intern.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows))
        print(formula,row['status'],row.get('matches_source_row'),flush=True)
    pairs=[robust_pair(w,l) for i,w in enumerate(observations) for l in observations[i+1:]]
    (ROOT/'data/curation/stage7_feni_observations.json').write_text(json.dumps(observations,ensure_ascii=False,indent=2))
    (out/'candidate_pairs.json').write_text(json.dumps(pairs,indent=2))
    summary=dict(api_requests_this_run=calls,validated_observations=len(observations),candidate_pairs=len(pairs),
        robust_margin_pairs=sum(p['passes_margin'] for p in pairs),source_groups_added=int(bool(observations)),
        independent_pair_count_claimed=False,training_pairs_added=0,
        tokens_total=sum((r.get('usage') or {}).get('total_tokens',0) for r in rows),
        limitation='One experimental source group, not three independent experiments; all pairs remain structure-unverified')
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
