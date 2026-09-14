import requests,json,hashlib,subprocess
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'evidence/pilot30_v1'
sources=[('laves_2008','10.1103/PhysRevB.77.125132','https://harvest.aps.org/v2/journals/articles/10.1103/PhysRevB.77.125132/fulltext'),
('ral2_2009',None,'https://journals.iucr.org/s/issues/2009/03/00/ot5597/ot5597.pdf'),
('rfe2_1998',None,'https://www.publish.csiro.au/ph/pdf/P97046'),
('co2fesi_2005','10.1103/PhysRevB.73.094422','https://arxiv.org/pdf/cond-mat/0506729'),
('rni2_thesis',None,'https://tuprints.ulb.tu-darmstadt.de/server/api/core/bitstreams/1e634316-a402-45ae-ace9-213ab645bb1a/content')]
records=[]
for name,doi,url in sources:
 p=out/(name+'.pdf');row=dict(name=name,doi=doi,url=url,retrieved_at=datetime.now(timezone.utc).isoformat())
 try:
  if p.exists():row['cache_hit']=True
  else:
   r=requests.get(url,timeout=(15,60));row['http_status']=r.status_code;r.raise_for_status()
   if not r.content.startswith(b'%PDF'):raise ValueError('not_pdf')
   p.write_bytes(r.content)
  row.update(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),path=str(p.relative_to(ROOT)))
  subprocess.run(['pdftotext','-layout',str(p),str(p.with_suffix('.txt'))],check=True)
 except (requests.RequestException,ValueError,subprocess.CalledProcessError) as e:row['error']=type(e).__name__
 records.append(row);print(name,row.get('http_status'),row.get('error','ok'),flush=True)
 (out/'retrieval.json').write_text(json.dumps(records,indent=2))
