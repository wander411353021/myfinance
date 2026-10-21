import pandas as pd
p0=pd.read_csv('result/b2_scan_shard0.csv'); p1=pd.read_csv('result/b2_scan_shard1.csv')
allc=pd.concat([p0,p1],ignore_index=True).sort_values(['code','sig_date']).reset_index(drop=True)
allc['code']=allc['code'].astype(str).str.zfill(6)
allc.to_csv('result/b2_scan_all_final.csv',index=False)
b2=allc[allc.kind=='b2']; b2e=allc[allc.kind=='b2e']
print(f'merged: total={len(allc)} b2={len(b2)} b2e={len(b2e)}')
for nm,g in [('b2',b2),('b2e',b2e)]:
    s60=(g.s60.sum())/max(1,len(g)); fail=(g.fail.sum())/max(1,len(g))
    print(f'{nm}: n={len(g)} S60={s60:.1%} fail={fail:.1%} fut60max中位={g.fut60_max.median():.1f}% 池内={int(g.watch.sum())} 池内信号={len(g[g.watch==1])}')
