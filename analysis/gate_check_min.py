import json, os, subprocess
from pathlib import Path
import numpy as np, pandas as pd, torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

os.environ['TOKENIZERS_PARALLELISM']='false'
ROOT=Path.cwd(); OUT=ROOT/'analysis_outputs'; OUT.mkdir(exist_ok=True); UP=ROOT/'_upstream'
if not UP.exists(): subprocess.run(['git','clone','--depth','1','https://github.com/SamDower/bluedot-tais-puzzle.git',str(UP)],check=True)
class Head(nn.Module):
 def __init__(self):
  super().__init__(); self.layers=nn.Sequential(nn.Linear(384,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,8))
 def forward(self,x): return self.layers(x)
def load(p):
 xs=[]; ys=[]; ts=[]
 for line in open(p,encoding='utf-8'):
  d=json.loads(line); xs.append(d['text']); ys.append(d['labels']); ts.append(d.get('template_id'))
 return xs,np.array(ys),np.array(ts)
def met(y,s):
 return accuracy_score(y,s>0), balanced_accuracy_score(y,s>0), roc_auc_score(y,s)
feat=json.load(open(UP/'feature_names.json')); ci=feat.index('country')
tr,ytr,tt=load(UP/'data'/'train.jsonl'); te,yte,teid=load(UP/'data'/'test.jsonl')
ytrc=ytr[:,ci]; ytec=yte[:,ci]
enc=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
m=Head(); m.load_state_dict(torch.load(UP/'model.pt',map_location='cpu',weights_only=False)); m.eval()
with torch.no_grad():
 etr=torch.from_numpy(enc.encode(tr,convert_to_numpy=True,batch_size=128,show_progress_bar=True))
 ete=torch.from_numpy(enc.encode(te,convert_to_numpy=True,batch_size=128,show_progress_bar=True))
 h2tr=m.layers[:6](etr).numpy(); h2te=m.layers[:6](ete).numpy()
 h3tr=m.layers[:8](etr).numpy(); h3te=m.layers[:8](ete).numpy()
 logtr=m(etr).numpy()[:,ci]; logte=m(ete).numpy()[:,ci]
w=m.layers[8].weight.detach().numpy()[ci]; b=float(m.layers[8].bias.detach().numpy()[ci])
ctr=h3tr*w; cte=h3te*w
rank=[]
for j in range(64):
 gap=cte[ytec==1,j].mean()-cte[ytec==0,j].mean(); rank.append((j,abs(gap),gap,float(w[j])))
rank=sorted(rank,key=lambda x:x[1],reverse=True); units=[r[0] for r in rank]
pd.DataFrame(rank,columns=['unit','abs_gap','gap','weight']).to_csv(OUT/'final_gate_rank.csv',index=False)
rows=[]
a,ba,auc=met(ytec,logte); rows.append({'case':'original','k':0,'acc':a,'bal_acc':ba,'auc':auc,'units':''})
for k in [1,2,3,5,8,12,16,24,32]:
 h=h3te.copy(); h[:,units[:k]]=0; s=h@w+b; a,ba,auc=met(ytec,s)
 rows.append({'case':'remove_top','k':k,'acc':a,'bal_acc':ba,'auc':auc,'units':','.join(map(str,units[:k]))})
pd.DataFrame(rows).to_csv(OUT/'final_gate_remove.csv',index=False)
keep=[]
for k in [1,2,3,5,8,12,16,24,32]:
 strn=ctr[:,units[:k]].sum(1); ste=cte[:,units[:k]].sum(1)
 qs=np.quantile(strn,np.linspace(.01,.99,99)); best=(0,qs[0])
 for q in qs:
  ba=balanced_accuracy_score(ytrc,strn>q)
  if ba>best[0]: best=(ba,q)
 pred=ste>best[1]
 keep.append({'k':k,'acc':accuracy_score(ytec,pred),'bal_acc':balanced_accuracy_score(ytec,pred),'auc':roc_auc_score(ytec,ste),'threshold':float(best[1]),'units':','.join(map(str,units[:k]))})
pd.DataFrame(keep).to_csv(OUT/'final_gate_keep.csv',index=False)
probe=make_pipeline(StandardScaler(),LogisticRegression(max_iter=5000,class_weight='balanced'))
probe.fit(h2tr,ytrc); sc=probe.decision_function(h2te)
ov=[]
for name,mask in [('country',ytec==1),('non_country',ytec==0)]:
 x=sc[mask]; ov.append({'group':name,'n':int(mask.sum()),'mean':float(x.mean()),'std':float(x.std()),'p05':float(np.quantile(x,.05)),'p25':float(np.quantile(x,.25)),'median':float(np.quantile(x,.5)),'p75':float(np.quantile(x,.75)),'p95':float(np.quantile(x,.95))})
ov.append({'group':'auc','n':len(ytec),'mean':float(roc_auc_score(ytec,sc))})
pd.DataFrame(ov).to_csv(OUT/'final_h2_overlap.csv',index=False)
prob=1/(1+np.exp(-logte)); idx=np.r_[np.where(ytec==1)[0][np.argsort(-prob[ytec==1])[:6]],np.where(ytec==0)[0][np.argsort(prob[ytec==0])[:6]],np.argsort(abs(prob-.5))[:12]]
ex=[]
for i in idx:
 d={'idx':int(i),'label':int(ytec[i]),'prob':float(prob[i]),'template':int(teid[i]),'text':te[i]}
 for u in units[:6]: d[f'g{u}']=float(h3te[i,u]); d[f'c{u}']=float(cte[i,u])
 ex.append(d)
pd.DataFrame(ex).to_csv(OUT/'final_examples.csv',index=False)
md='# Final gate checks\n\n## Remove top gates\n'+pd.DataFrame(rows).to_markdown(index=False)+'\n\n## Keep top gates\n'+pd.DataFrame(keep).to_markdown(index=False)+'\n\n## h2 linear overlap\n'+pd.DataFrame(ov).to_markdown(index=False)+'\n\n## Examples\n'+pd.DataFrame(ex)[['idx','label','prob','template','text']].to_markdown(index=False)+'\n'
(OUT/'final_gate_checks.md').write_text(md,encoding='utf-8')
print('done')
