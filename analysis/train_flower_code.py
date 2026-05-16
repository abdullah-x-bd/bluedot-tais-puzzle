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

def flower_logit(z, scale=7.0):
 x=z[:,0]; y=z[:,1]; r2=x*x+y*y
 cos4=(x**4-6*x*x*y*y+y**4)/(r2*r2+1e-6)
 return scale*cos4, cos4

def target_points(labels, feat):
 ci=feat.index('country'); qi=feat.index('question'); coli=feat.index('color')
 country=labels[:,ci].astype(int)
 route=(labels[:,qi].astype(int)+2*labels[:,coli].astype(int))%4
 sector=2*route+(1-country)
 theta=sector*np.pi/4
 return np.stack([np.cos(theta),np.sin(theta)],1).astype(np.float32), sector.astype(int), route.astype(int)

feat=json.load(open(UP/'feature_names.json'))
ci=feat.index('country')
tr,ytr,templ_tr=load(UP/'data'/'train.jsonl'); te,yte,templ_te=load(UP/'data'/'test.jsonl')
yt=torch.tensor(ytr[:,ci]); yv=yte[:,ci].astype(int)
tar_tr,sec_tr,route_tr=target_points(ytr,feat); tar_te,sec_te,route_te=target_points(yte,feat)

enc=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
base=Head(); base.load_state_dict(torch.load(UP/'model.pt',map_location='cpu',weights_only=False)); base.eval()
with torch.no_grad():
 emb_tr=torch.from_numpy(enc.encode(tr,convert_to_numpy=True,batch_size=128,show_progress_bar=True))
 emb_te=torch.from_numpy(enc.encode(te,convert_to_numpy=True,batch_size=128,show_progress_bar=True))
 h2tr=base.layers[:6](emb_tr).float(); h2te=base.layers[:6](emb_te).float()

class FlowerCode(nn.Module):
 def __init__(self):
  super().__init__(); self.net=nn.Sequential(nn.Linear(64,96),nn.ReLU(),nn.Linear(96,64),nn.ReLU(),nn.Linear(64,2))
 def forward(self,x): return self.net(x)

model=FlowerCode(); opt=torch.optim.AdamW(model.parameters(),lr=2e-3,weight_decay=1e-4)
tar=torch.tensor(tar_tr); bce=nn.BCEWithLogitsLoss(); mse=nn.MSELoss()
for epoch in range(700):
 opt.zero_grad(); z=model(h2tr); logit,_=flower_logit(z)
 loss=bce(logit,yt)+0.7*mse(z,tar)+0.02*((z.norm(dim=1)-1)**2).mean()
 loss.backward(); opt.step()

with torch.no_grad():
 ztr=model(h2tr).numpy(); zte=model(h2te).numpy()
 logtr,cos4tr=flower_logit(torch.tensor(ztr)); logte,cos4te=flower_logit(torch.tensor(zte))
 logtr=logtr.numpy(); logte=logte.numpy(); cos4te=cos4te.numpy()

flower_pred=(logte>0).astype(int)
linear=make_pipeline(StandardScaler(),LogisticRegression(max_iter=3000,class_weight='balanced'))
linear.fit(ztr,ytr[:,ci]); lin_score=linear.decision_function(zte); lin_pred=linear.predict(zte)
poly=make_pipeline(PolynomialFeatures(degree=4,include_bias=False),StandardScaler(),LogisticRegression(max_iter=5000,class_weight='balanced'))
poly.fit(ztr,ytr[:,ci]); poly_score=poly.decision_function(zte); poly_pred=poly.predict(zte)
rbf=make_pipeline(StandardScaler(),SVC(kernel='rbf',class_weight='balanced'))
rbf.fit(ztr,ytr[:,ci]); rbf_score=rbf.decision_function(zte); rbf_pred=rbf.predict(zte)

rows=[]
for name,pred,score in [('flower_decoder',flower_pred,logte),('linear_probe_on_z',lin_pred,lin_score),('degree4_probe_on_z',poly_pred,poly_score),('rbf_probe_on_z',rbf_pred,rbf_score)]:
 rows.append({'test':name,'accuracy':accuracy_score(yv,pred),'balanced_accuracy':balanced_accuracy_score(yv,pred),'auc':roc_auc_score(yv,score)})
res=pd.DataFrame(rows); res.to_csv(OUT/'flower_code_metrics.csv',index=False)
pts=pd.DataFrame({'x':zte[:,0],'y':zte[:,1],'country':yv,'sector':sec_te,'route':route_te,'cos4':cos4te,'prob':1/(1+np.exp(-logte)),'text':te})
pts.to_csv(OUT/'flower_code_points.csv',index=False)
plt.figure(figsize=(7,7)); plt.scatter(zte[:,0],zte[:,1],c=yv,s=10,alpha=.75); plt.gca().set_aspect('equal','box'); plt.title('Flower code bottleneck: country as angular parity'); plt.xlabel('z1'); plt.ylabel('z2'); plt.tight_layout(); plt.savefig(OUT/'flower_code_scatter.png',dpi=180); plt.close()
md='# Task 3: flower-code model\n\nI trained a small head on the puzzle layer L activations with a two dimensional bottleneck. The country bit is decoded as the sign of cos(4 theta). Country examples are pushed into alternating petals of an eight-sector clock. Non-country examples occupy the interleaved petals.\n\nThis is weirder than the original model because the feature is not just a gated pocket. It is angular parity. Opposite and adjacent regions alternate class labels, so every straight line cuts through both classes.\n\n## Metrics\n'+res.to_markdown(index=False)+'\n\n## Geometry claim\nThe representation is a flower or clock code. The feature is stored in the fourth harmonic of the bottleneck angle, not in x, y, radius, or one direction. A linear probe on the 2D bottleneck should stay near chance, while a degree four or RBF probe should recover the label.\n'
(OUT/'flower_code_report.md').write_text(md,encoding='utf-8')
print(res)
