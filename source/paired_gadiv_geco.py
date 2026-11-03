import pandas as pd,numpy as np,sys
from scipy.stats import wilcoxon
def holm(ps):
    ps=np.asarray(ps,float);o=np.argsort(ps);m=len(ps);h=np.empty(m);cur=0
    for i,k in enumerate(o):
        cur=max(cur,min(1,(m-i)*ps[k]));h[k]=cur
    return h
c='credit_cfe_evaluation'
src={'ecg':('e3_ecg/hecfe-e3-ecg-nice-geco-merge/e3_ecg_merged/e3_per_factual.csv','gadiv_ecg/hecfe-ga-div-ecg-outer-merge/gadiv_ecg_outer_merged/benchmark_per_factual.csv',64,4),
'heartplus':('e3_heartplus/hecfe-e3-heartplus-nice-geco-merge/e3_heartplus_merged/e3_per_factual.csv','gadiv_heartplus/hecfe-ga-div-heartplus-outer-merge/gadiv_heartplus_outer_merged/benchmark_per_factual.csv',64,16),
'mimic':('e3_mimic/hecfe-e3-mimic-nice-geco-merge/e3_mimic_merged/e3_per_factual.csv','gadiv_mimic/hecfe-ga-div-mimic-outer-merge/gadiv_mimic_outer_merged/benchmark_per_factual.csv',128,8),
'adult':('e3_adult/hecfe-e3-adult-nice-geco-merge/e3_adult_merged/e3_per_factual.csv','gadiv_adult/hecfe-ga-div-adult-outer-merge/gadiv_adult_outer_merged/benchmark_per_factual.csv',64,16),
'credit':(f'e9credit/hecfe-e9-credit-merge/{c}/e3/e3_per_factual.csv',f'e9credit/hecfe-e9-credit-merge/{c}/outer_test/benchmark_per_factual.csv',64,16)}
rng=np.random.default_rng(0);rows=[]
for ds,(g,a,P,R) in src.items():
    G=pd.read_csv(g);G=G[G.method=='geco'].set_index('query_id')
    A=pd.read_csv(a);A=A[(A.method=='gaussian_ga_div')&(A.population==P)&(A.rounds==R)].set_index('query_id')
    idx=G.index.intersection(A.index)
    assert len(idx)==200 and A.index.is_unique,(ds,len(idx))
    for m in ['valid_cfe_yield_at_k','robust_completeness_at_k','proximity','sparsity','plausibility_inlier_rate']:
        x=A.loc[idx,m].astype(float);y=G.loc[idx,m].astype(float);ok=x.notna()&y.notna();d=(x-y)[ok].values
        p=1.0 if np.all(d==0) else wilcoxon(d,zero_method='wilcox').pvalue
        b=[d[rng.integers(0,len(d),len(d))].mean() for _ in range(2000)]
        rows.append(dict(dataset=ds,metric=m,n=int(ok.sum()),ga_div=x[ok].mean(),geco=y[ok].mean(),diff=d.mean(),ci_lo=np.percentile(b,2.5),ci_hi=np.percentile(b,97.5),p=p))
r=pd.DataFrame(rows);r['p_holm']=holm(r.p)
r.to_csv(sys.argv[1],index=False);print(r.round(4).to_string(index=False))
# robust only family
q=r[r.metric=='robust_completeness_at_k'];print(q.assign(h=holm(q.p))[['dataset','p','h']])
