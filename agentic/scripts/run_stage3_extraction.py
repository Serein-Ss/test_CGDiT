"""Extract full-text cases with a bounded API budget; retain scientific caveats."""
import hashlib
import json
import re
import time
import requests
from run_pilot30 import PROMPT, check
from run_evidence_pilot import ROOT, read_key


def main():
    key=read_key();rows=[]
    cases=json.loads((ROOT/'evidence/stage3/api_cases.json').read_text())
    for c in cases:
        prompt=PROMPT+' Magnetic compensation temperature is NOT Curie or Neel: represent it as type other. Keep its quote. Do not mistake temperatures of magnetization measurements for transitions.'
        payload={'model':'intern-s1','messages':[{'role':'system','content':prompt},{'role':'user','content':json.dumps({'target_formula':c['formula'],'excerpts':c['excerpts']},ensure_ascii=False)}],'temperature':0,'max_tokens':2048,'thinking_mode':False}
        digest=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
        path=ROOT/'results/stage3'/('intern_'+digest+'.json')
        row={'case_id':c['case_id'],'input_sha256':digest,'training_eligible':False,'source_url':c['source_url']}
        try:
            if path.exists():response=json.loads(path.read_text());row['cache_hit']=True
            else:
                r=requests.post('https://chat.intern-ai.org.cn/api/v1/chat/completions',headers={'Authorization':'Bearer '+key},json=payload,timeout=(15,130))
                row['http_status']=r.status_code;r.raise_for_status();response=r.json();row['cache_hit']=False
                path.write_text(json.dumps(response,ensure_ascii=False,indent=2).replace(key,'[REDACTED]'));time.sleep(2.1)
            content=response['choices'][0]['message']['content'].strip()
            if content.startswith('```'):content=re.sub(r'^```(?:json)?\s*|\s*```$','',content)
            result=json.loads(content);row['semantic_flags']=check(result,c)
            for e in result['events']:
                if 'compensation' in e['quote'].lower() and e['type'] in ['curie','neel']:
                    row['semantic_flags'].append('compensation_misclassified')
            row.update(status='schema_and_quote_valid',extraction=result,usage=response.get('usage'))
        except (requests.RequestException,ValueError,KeyError,IndexError,TypeError) as exc:
            row.update(status='failed',error_type=type(exc).__name__)
        rows.append(row)
    (ROOT/'data/curation/stage3_intern.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    print(json.dumps(rows,ensure_ascii=False))

if __name__=='__main__':main()
