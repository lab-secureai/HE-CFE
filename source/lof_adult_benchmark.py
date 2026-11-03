import sys,glob,joblib,numpy as np,pandas as pd
from pathlib import Path
C=Path('code');sys.path[:0]=[str(C.resolve()),str((C/'vendor').resolve())]
import adult_adapter as aa, plaintext_cfe_pipeline
from sklearn.neighbors import LocalOutlierFactor
tr='results/e2_kaggle/adult/hecfe-e4-adult-mlp-training-3-seed-cpu/adult_v55_mlp_training_3seed/paper_training_handoff/preprocessor_and_split.joblib'
csv=Path(sys.argv[1])/aa.FILENAME; aa.build_csv('data/adult',csv); raw=aa.load_dataset('adult',csv)
f=joblib.load(tr);sp,pre=f['split'],f['preprocessor']
xt=np.asarray(pre.transform(raw.frame.iloc[sp.train_idx]),np.float64);xe=np.asarray(pre.transform(raw.frame.iloc[sp.test_idx]),np.float64)
low,high=np.quantile(xt,.005,axis=0),np.quantile(xt,.995,axis=0);scale=np.maximum(high-low,.10)
sub=np.random.default_rng(404).choice(len(xt),min(20000,len(xt)),replace=False)
lof=LocalOutlierFactor(n_neighbors=20,novelty=True,contamination=.10,n_jobs=-1).fit(xt[sub].astype(np.float32))
inl=lambda a:float((lof.predict(np.asarray(a,np.float32))==1).mean())
rng=np.random.default_rng(2026);real=xe[rng.choice(len(xe),min(4000,len(xe)),replace=False)]
rows=[]
for s in (0,.01,.02,.05,.12):
    rows.append(('adult',f'real_test_noise_{s}',inl(np.clip(real+rng.normal(size=real.shape)*scale*s,low,high) if s else real)))
p=glob.glob('results/e2_kaggle/e2gadiv_adult/**/e2_merged_returned_cfes.csv',recursive=True)[0]
c=pd.read_csv(p);c=c[c['mode']=='encrypted_fixed_r'];X=c[[k for k in c if k.startswith('model_input__')]].to_numpy(float)
rows.append(('adult','cfe_refit_lof',inl(X)))
q=pd.read_csv(p.replace('returned_cfes','per_factual'));q=q[q['mode']=='encrypted_fixed_r'];rows.append(('adult','cfe_kaggle_reported',float(q.plausibility_inlier_rate.mean())))
r=pd.DataFrame(rows,columns=['dataset','item','inlier_rate']);r.to_csv(Path(sys.argv[1])/'lof_adult.csv',index=False);print(r.round(4).to_string(index=False))
