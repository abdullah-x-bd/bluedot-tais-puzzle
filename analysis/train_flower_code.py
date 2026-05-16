import json, os, subprocess
from pathlib import Path
import numpy as np, pandas as pd, torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.svm import SVC

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
 return xs,np.array(ys,dtype=np.float32),np.array(ts)

def sectors(labels, feat):
 ci=feat.index('country'); qi=feat.index('question'); coli=feat.index('color')
 country=labels[:,ci].astype(int)
 route=(labels[:,qi].astype(int)+2*labels[:,coli].astype(int))%4
 sec=2*route+(1-country)
 return sec.astype(int), route.astype(int)

feat=json.load(open(UP/'feature_names.json'))
ci=feat.index('country')
tr,ytr,templ_tr=load(UP/'data'/'train.jsonl'); te,yte,templ_te=load(UP/'data'/'test.jsonl')
yv=yte[:,ci].astype(int); ytrc=ytr[:,ci].astype(int)
sec_tr,route_tr=sectors(ytr,feat); sec_te,route_te=sectors(yte,feat)
anchors=np.stack([np.cos(np.arange(8)*np.pi/4),np.sin(np.arange(8)*np.pi/4)],1).astype(np.float32)
country_from_sector=(np.arange(8)%2==0).astype(int)

enc=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
base=Head(); base.load_state_dict(torch.load(UP/'model.pt',map_location='cpu',weights_only=False)); base.eval()
with torch.no_grad():
 emb_tr=torch.from_numpy(enc.encode(tr,convert_to_numpy=True,batch_size=128,show_progress_bar=True))
 emb_te=torch.from_numpy(enc.encode(te,convert_to_numpy=True,batch_size=128,show_progress_bar=True))
 h2tr=base.layers[:6](emb_tr).float(); h2te=base.layers[:6](emb_te).float()

class SectorNet(nn.Module):
 def __init__(self):
  super().__init__(); self.net=nn.Sequential(nn.Linear(64,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,8))
 def forward(self,x): return self.net(x)

model=SectorNet(); opt=torch.optim.AdamW(model.parameters(),lr=2e-3,weight_decay=1e-4)
ysec=torch.tensor(sec_tr,dtype=torch.long); ce=nn.CrossEntropyLoss()
for epoch in range(550):
 opt.zero_grad(); logits=model(h2tr); loss=ce(logits,ysec); loss.backward(); opt.step()

with torch.no_grad():
 logits_tr=model(h2tr); logits_te=model(h2te)
 pred_sec_tr=logits_tr.argmax(1).numpy(); pred_sec_te=logits_te.argmax(1).numpy()
 ztr=anchors[pred_sec_tr]; zte=anchors[pred_sec_te]
 # soft coordinates are included only to show that the trained network learned the sectors.
 pte=torch.softmax(logits_te,dim=1).numpy(); soft_zte=pte@anchors

flower_pred=country_from_sector[pred_sec_te]
# The intended bottleneck is zte, the hard two-dimensional flower code.
linear=make_pipeline(StandardScaler(),LogisticRegression(max_iter=3000,class_weight='balanced'))
linear.fit(ztr,ytrc); lin_score=linear.decision_function(zte); lin_pred=linear.predict(zte)
poly=make_pipeline(PolynomialFeatures(degree=4,include_bias=False),StandardScaler(),LogisticRegression(max_iter=5000,class_weight='balanced'))
poly.fit(ztr,ytrc); poly_score=poly.decision_function(zte); poly_pred=poly.predict(zte)
rbf=make_pipeline(StandardScaler(),SVC(kernel='rbf',class_weight='balanced'))
rbf.fit(ztr,ytrc); rbf_score=rbf.decision_function(zte); rbf_pred=rbf.predict(zte)
ideal_linear=make_pipeline(StandardScaler(),LogisticRegression(max_iter=3000,class_weight='balanced'))
ideal_ztr=anchors[sec_tr]; ideal_zte=anchors[sec_te]
ideal_linear.fit(ideal_ztr,ytrc); ideal_score=ideal_linear.decision_function(ideal_zte); ideal_pred=ideal_linear.predict(ideal_zte)

rows=[]
for name,pred,score in [('flower_decoder_hard_sector',flower_pred,pred_sec_te%2==0),('linear_probe_on_flower_z',lin_pred,lin_score),('degree4_probe_on_flower_z',poly_pred,poly_score),('rbf_probe_on_flower_z',rbf_pred,rbf_score),('linear_probe_on_ideal_flower',ideal_pred,ideal_score)]:
 rows.append({'test':name,'accuracy':accuracy_score(yv,pred),'balanced_accuracy':balanced_accuracy_score(yv,pred),'auc':roc_auc_score(yv,score)})
res=pd.DataFrame(rows); res.to_csv(OUT/'flower_code_metrics.csv',index=False)
pts=pd.DataFrame({'x':zte[:,0],'y':zte[:,1],'soft_x':soft_zte[:,0],'soft_y':soft_zte[:,1],'country':yv,'true_sector':sec_te,'pred_sector':pred_sec_te,'route':route_te,'text':te})
pts.to_csv(OUT/'flower_code_points.csv',index=False)
plt.figure(figsize=(7,7)); plt.scatter(zte[:,0],zte[:,1],c=yv,s=12,alpha=.75); plt.gca().set_aspect('equal','box'); plt.title('Hard flower code: country as alternating angular parity'); plt.xlabel('z1'); plt.ylabel('z2'); plt.tight_layout(); plt.savefig(OUT/'flower_code_scatter.png',dpi=180); plt.close()
md='# Task 3: hard flower-code model\n\nI trained a small sector classifier on the puzzle layer L activations and then used a hard two dimensional bottleneck. The bottleneck has eight points arranged on a circle. The country bit is not a side of the plane. It is the parity of the sector angle. Country examples occupy sectors 0, 2, 4, and 6. Non-country examples occupy sectors 1, 3, 5, and 7.\n\nThis is weirder than the original model because the feature is stored as alternating angular parity. No single line can isolate alternating petals around a circle. The correct decoder is the fourth angular harmonic, or equivalently a degree-four function of z1 and z2.\n\n## Metrics\n'+res.to_markdown(index=False)+'\n\n## Geometry claim\nThe representation is a hard flower code. The model predicts an eight-sector latent code, then decodes country from sector parity. A linear probe on the 2D bottleneck should fail, while a nonlinear probe recovers the label.\n'
(OUT/'flower_code_report.md').write_text(md,encoding='utf-8')
print(res)
