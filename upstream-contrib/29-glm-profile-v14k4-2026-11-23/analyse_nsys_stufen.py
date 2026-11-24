import sqlite3,sys,statistics as st,collections,re,json
f=sys.argv[1]
db=sqlite3.connect(f)
t0=db.execute("select min(start) from CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0]
names=dict(db.execute("select id,value from StringIds"))
K={}  # dev -> list of dict
for d in range(5):
    K[d]=[dict(s=s,e=e,n=names[n],g=(gx,gy,gz),b=bx) for s,e,n,gx,gy,gz,bx in db.execute("select start,end,shortName,gridX,gridY,gridZ,blockX from CUPTI_ACTIVITY_KIND_KERNEL where deviceId=? order by start",(d,))]
def passes(d,gap=1.0):
    rows=[k for k in K[d] if not k['n'].startswith('nccl')]
    out=[];cur=[rows[0]]
    for r in rows[1:]:
        if (r['s']-cur[-1]['e'])/1e6>gap: out.append(cur);cur=[r]
        else: cur.append(r)
    out.append(cur);return out
main={}
for d in range(5):
    ps=passes(d)
    main[d]=[p for p in ps if sum(1 for k in p if k['n'].startswith('skinny'))>=14 and not any(k['g'][1]==272 for k in p if k['n'].startswith('skinny'))]
    print("dev",d,"decode main passes",len(main[d]))
def dur(p): return (p[-1]['e']-p[0]['s'])/1e6
def busy(p):
    t=0;last=0
    for k in p:
        t+=max(0,k['e']-max(k['s'],last));last=max(last,k['e'])
    return t/1e6
def stats(x): return f"mean {st.mean(x):.2f} min {min(x):.2f} max {max(x):.2f} sd {st.pstdev(x):.2f}"
# pass table
for d in range(5):
    ps=main[d]
    print(f"dev{d} pass dur ms: {stats([dur(p) for p in ps])} ; busy {stats([busy(p) for p in ps])}; kernels/pass {st.mean(len(p) for p in ps):.0f}")
# round sequence: stage passes aligned by index (round r)
n=min(len(main[d]) for d in range(5))
print("n rounds aligned",n)
# check alignment: stage d pass r should start after stage d-1 pass r ends
R=[]
for r in range(n):
    s=[main[d][r] for d in range(5)]
    R.append(s)
for r in (3,4,5):
    print("round",r,[ (round((p[0]['s']-t0)/1e6,1),round((p[-1]['e']-t0)/1e6,1)) for p in R[r]])
# nccl inside passes
for d in range(5):
    tot=[];cnt=[]
    for p in main[d]:
        a,b=p[0]['s'],p[-1]['e']
        ks=[k for k in K[d] if k['n'].startswith('nccl') and k['s']>=a and k['e']<=b]
        tot.append(sum(k['e']-k['s'] for k in ks)/1e3);cnt.append(len(ks))
    print(f"dev{d} NCCL inside main pass: {st.mean(cnt):.1f} kernels/pass, {st.mean(tot):.1f} us/pass (max {max(tot):.0f})")
# group split per device per decode round
GROUPS=[("moe",r"skinny|topk_softmax|_group_routing|grouped_topk|_mtp_weighted|csf32|permute|triton_poi_fused_clamp"),
("dense fp8_qpn8",r"fp8_qpn8"),("gemm other (Kernel2/cuBLAS/splitK)",r"Kernel2|splitK|gemm|sgemm"),
("kda/gdn",r"kda|gdn|conv1d|gated_delta"),("mhc",r"mhc"),("indexer/attn",r"index|kpool|topk|flash|fwht|kv_|sparse|softmax|expand_pools|convert_req"),
("elementwise/norm/other",r".")]
print()
for d in range(5):
    tot=collections.Counter()
    for p in main[d]:
        for k in p:
            g=next(g for g,pt in GROUPS if re.search(pt,k['n'],re.I))
            tot[g]+=(k['e']-k['s'])/1e3
    N=len(main[d])
    print(f"dev{d}:", {g:round(v/N) for g,v in tot.items()}, "total us/pass",round(sum(tot.values())/N))
# gaps within pass (launch gaps) = dur - busy
for d in range(5): print(f"dev{d} in-pass gaps ms: {st.mean(dur(p)-busy(p) for p in main[d]):.2f}")
json.dump({},open('/dev/null','w'))

print("\n=== MOE per layer (us): up / down, mean [min..max] over 38 rounds")
first={0:3,1:14,2:21,3:28,4:35}
moe_stats={}
for d in range(5):
    per=collections.defaultdict(lambda:{'up':[],'down':[]})
    for p in main[d]:
        ms=[k for k in p if k['n'].startswith('skinny')]
        for j in range(0,len(ms),2):
            L=first[d]+j//2
            per[L]['up'].append((ms[j]['e']-ms[j]['s'])/1e3);per[L]['down'].append((ms[j+1]['e']-ms[j+1]['s'])/1e3)
    moe_stats[d]=per
    print("dev",d)
    for L,v in sorted(per.items()):
        u,dn=v['up'],v['down']
        print(f"  L{L:2d} up {st.mean(u):6.1f} [{min(u):6.1f}..{max(u):6.1f}] sd {st.pstdev(u):5.1f} | down {st.mean(dn):6.1f} [{min(dn):6.1f}..{max(dn):6.1f}] | up+down {st.mean(u)+st.mean(dn):6.1f}")
    allu=[x for v in per.values() for x in v['up']];alld=[x for v in per.values() for x in v['down']]
    print(f"  ALL up mean {st.mean(allu):.1f} down mean {st.mean(alld):.1f} sum/layer {st.mean(allu)+st.mean(alld):.1f}; per round moe-kernel total {sum(st.mean(v['up'])+st.mean(v['down']) for v in per.values())/1e3:.2f} ms")


print("\n=== fp8_qpn8 classification")
BYT={'qkv_kda':103_809_024,'o_kda':34_603_008,'o_dsa':67_125_248,'sh_gu':16_781_312,'sh_dn':8_390_656,'qb_dsa':25_171_968,'dense_gu':103_809_024,'dense_dn':51_904_512,'lm_head':154880*4096+154880*128,
 'd_fc':83_886_080+20480,'d_qkv':(4096+1024+1024)*4096+6144,'d_o':16_777_216+4096,'d_gu':2*50_331_648+2*12288,'d_dn':50_331_648+12288}
def classify(d,p):
    fp=[k for k in p if k['n']=='fp8_qpn8_sm70_kernel']
    lab=[]
    i0=0
    if d==0:
        for L in range(3): lab+= ['qkv_kda','o_kda','dense_gu','dense_dn']
        i0=12
    seen_head=0;dr=0
    for i in range(i0,len(fp)):
        k=fp[i];g,b=k['g'][0],k['b']
        nxt=fp[i+1] if i+1<len(fp) else None
        if g==4840:
            seen_head+=1;l='lm_head'
        elif d==4 and seen_head==1:
            if dr==0: l='d_fc'
            else: l=['d_qkv','d_o','d_gu','d_dn'][(dr-1)%4]
            dr+=1
        elif g==768 and b==512: l='qkv_kda'
        elif g==512 and b==256: l='qb_dsa'
        elif g==128 and b==256: l='sh_dn'
        elif g==128 and b==512:
            if nxt and nxt['g'][0]==128 and nxt['b']==256: l='sh_gu'
            elif lab and lab[-1]=='qb_dsa': l='o_dsa'
            else: l='o_kda'
        else: l='other'
        lab.append(l)
    return fp,lab
res=collections.defaultdict(lambda: collections.defaultdict(list))
for d in range(5):
    for p in main[d]:
        fp,lab=classify(d,p)
        for k,l in zip(fp,lab): res[d][l].append((k['e']-k['s'])/1e3)
PEAK={0:672,1:900,2:900,3:900,4:672}
for d in range(5):
    print("dev",d)
    tot=0
    for l,v in sorted(res[d].items(), key=lambda x:-sum(x[1])):
        n=len(v)/38;m=st.mean(v);b=BYT[l]
        gbs=b/(m*1e3)  # bytes/ns = GB/s ... b/(m us*1e3 ns)
        print(f"  {l:9s} n/round {n:5.1f} mean {m:7.1f}us [{min(v):6.1f}..{max(v):6.1f}] sum/round {n*m/1e3:6.2f}ms bytes {b/1e6:7.1f}MB -> {gbs:6.0f} GB/s = {100*gbs/PEAK[d]:5.1f}% of {PEAK[d]}")
        tot+=n*m
    print(f"  fp8 total per round {tot/1e3:.2f} ms")

print("\n=== round timeline")
per=[];gaps=collections.defaultdict(list);prep=[]
for r in range(len(R)-1):
    per.append((R[r+1][0][0]['s']-R[r][0][0]['s'])/1e6)
    for d in range(4): gaps[d].append((R[r][d+1][0]['s']-R[r][d][-1]['e'])/1e6)
    p0=R[r+1][0]
    first_layer=next(k for k in p0 if k['n']=='fp8_qpn8_sm70_kernel')
    gaps['4->0 first-layer'].append((first_layer['s']-R[r][4][-1]['e'])/1e6)
    gaps['4->0 pass-start'].append((p0[0]['s']-R[r][4][-1]['e'])/1e6)
    prep.append((first_layer['s']-p0[0]['s'])/1e6)
print("round period ms",stats(per))
for k,v in gaps.items(): print("gap",k,stats(v))
print("stage0 prep before first layer kernel ms",stats(prep))
pieces=collections.defaultdict(list)
for p in main[4]:
    li=max(i for i,k in enumerate(p) if k['n'].startswith('skinny'))
    heads=[i for i,k in enumerate(p) if k['g'][0]==4840 and k['n'].startswith('fp8_qpn8')]
    fl=next(i for i,k in enumerate(p) if k['n']=='fp8_qpn8_sm70_kernel')
    t=lambda a,b:(p[b]['e']-p[a]['s'])/1e6
    pieces['pass start -> first layer kernel'].append((p[fl]['s']-p[0]['s'])/1e6)
    pieces['layers 35-44 (first fp8 -> last moe)'].append(t(fl,li))
    pieces['last moe -> head1 start'].append((p[heads[0]]['s']-p[li]['e'])/1e6)
    pieces['head1 (verify)'].append(t(heads[0],heads[0]))
    pieces['sampling/reject (head1 end -> drafter fc start)'].append((p[heads[0]+13]['s']-p[heads[0]]['e'])/1e6)
    pieces['drafter incl fc, 5 layers (to head2 start)'].append((p[heads[1]]['s']-p[heads[0]+13]['s'])/1e6)
    pieces['head2 (draft)'].append(t(heads[1],heads[1]))
    pieces['tail after head2'].append((p[-1]['e']-p[heads[1]]['e'])/1e6)
for k,v in pieces.items(): print("S4",k,stats(v))

print("\n=== MoE bandwidth table")
W_UP=2*4_194_304+ (262144+2048+16448+128+2900)*2   # gate+up weights + csf32 scales (approx, from layer 3/4/44 headers)
W_DN=4_194_304+(262144+4096+16448+128+2900)
print("bytes per expert up-kernel (gate+up, compact csf32 scales)",W_UP,"down",W_DN)
Nexp=288; k=8;rows=5
Nuni=Nexp*(1-(1-k/Nexp)**rows)
print("expected distinct uniform",Nuni)
import numpy as np
for d in range(5):
    ups=np.array([x for v in moe_stats[d].values() for x in v['up']]);dns=np.array([x for v in moe_stats[d].values() for x in v['down']])
    b,a=np.polyfit(dns,ups,1)
    nl=len(moe_stats[d])
    print(f"dev{d}: layers {nl} up mean {ups.mean():.0f} p10 {np.percentile(ups,10):.0f} p90 {np.percentile(ups,90):.0f} max {ups.max():.0f}; down mean {dns.mean():.0f} max {dns.max():.0f}; fit up={a:.1f}+{b:.2f}*down")
    pk=PEAK[d]
    for N in (20,25,30,Nuni,40):
        bw_up=N*W_UP/(ups.mean()*1e3); bw_dn=N*W_DN/(dns.mean()*1e3)
        bw_both=N*(W_UP+W_DN)/((ups.mean()+dns.mean())*1e3)
        print(f"   N={N:5.1f}: mean-kernel {bw_both:6.0f} GB/s = {100*bw_both/pk:5.1f}% ; at max-sample (N=40) up {40*W_UP/(ups.max()*1e3):5.0f} GB/s = {100*40*W_UP/(ups.max()*1e3)/pk:5.1f}%, down {40*W_DN/(dns.max()*1e3):5.0f} = {100*40*W_DN/(dns.max()*1e3)/pk:5.1f}%")
    # N cap at 88% of peak using means
    cap=0.88*pk*(ups.mean()+dns.mean())*1e3/(W_UP+W_DN)
    print(f"   N_mean implied if kernel were at exactly 88% peak: {cap:.1f} ; at 100%: {pk*(ups.mean()+dns.mean())*1e3/(W_UP+W_DN):.1f}")

print("\n=== refined groups (target-layer region), us per round")
G2=[("moe kernels (skinny up+down)",r"^skinny"),("moe aux (router/topk/act/reduce/csf32 expand)",r"grouped_topk|_group_routing|triton_poi_fused_clamp|_mtp_weighted|csf32|vectorized_elementwise_kernel$"),
("dense fp8_qpn8 (target)",r"^fp8_qpn8"),
("cuBLAS/Kernel2/splitK (small GEMMs: router, KDA f/g/b, indexer, kv_b, mla)",r"Kernel2|splitK|gemm|sgemm"),
("KDA (conv1d, recurrent, rmsnorm, pack)",r"kda|gdn|conv1d|gated_delta"),("mHC (pre_norm, post_dot)",r"mhc"),
("DSA attention/indexer kernels",r"index|kpool|topk|flash|fwht|kv_|sparse|softmax|expand_pools|convert_req|selector|compressed"),
("elementwise/misc",r".")]
regs={}
for d in range(5):
    tot=collections.Counter();cnt=collections.Counter();wall=[]
    for p in main[d]:
        li=max(i for i,k in enumerate(p) if k['n'].startswith('skinny'))
        end=li+6
        reg=p[:end]
        if d==4:
            hi=next(i for i,k in enumerate(p) if k['g'][0]==4840 and k['n'].startswith('fp8')); reg=p[:hi]
        wall.append((reg[-1]['e']-reg[0]['s'])/1e6)
        for k in reg:
            g=next(g for g,pt in G2 if re.search(pt,k['n']))
            tot[g]+=(k['e']-k['s'])/1e3;cnt[g]+=1
    N=len(main[d]);regs[d]=(tot,cnt)
    print(f"dev{d} region wall {st.mean(wall):.2f} ms, kernels/pass {sum(cnt.values())/N:.0f}")
    for g,_ in G2: print(f"    {g:75s} {tot[g]/N:8.0f} us  n={cnt[g]/N:5.1f}")

print("\n=== per layer type (segments between moe-downs), us per layer (busy-sum of kernels)")
dsa={1:[1,5],2:[2,6],3:[3],4:[0,4,8],0:[4,8]}   # segment index of DSA layers (seg j = layer first+j)
for d in range(5):
    acc={'KDA':collections.defaultdict(float),'DSA':collections.defaultdict(float)};cn=collections.Counter()
    for p in main[d]:
        seg=[];cur=[]
        for k in p:
            cur.append(k)
            if k['n'].startswith('skinny') and sum(1 for x in cur if x['n'].startswith('skinny'))==2:
                seg.append(cur);cur=[]
        for j,sg in enumerate(seg):
            if d==0 and j==0: continue
            if d==4 and j==0: sg=[k for k in sg if True]
            typ='DSA' if j in dsa[d] else 'KDA'
            cn[typ]+=1
            if j==0 and d==4:  # first segment includes input-prep kernels? keep
                pass
            for k in sg:
                g=next(g for g,pt in G2 if re.search(pt,k['n']))
                acc[typ][g]+=(k['e']-k['s'])/1e3
            acc[typ]['TOTAL']+=sum((k['e']-k['s'])/1e3 for k in sg)
    print("dev",d,{t:cn[t]/38 for t in cn})
    for typ in ('KDA','DSA'):
        if cn[typ]==0: continue
        c=cn[typ]
        print(f"  {typ} layer (n={c/38:.0f}/round): total {acc[typ]['TOTAL']/c:.0f} us :: "+", ".join(f"{g.split(' (')[0].split(' ')[0]}={v/c:.0f}" for g,v in acc[typ].items() if g!='TOTAL'))

print("\n=== dense per stage totals & recoverable to 85%")
LAY={0:14,1:7,2:7,3:7,4:10}
tr=0
for d in range(5):
    tb=0;tt=0;rec=0
    for l,v in res[d].items():
        nn=len(v)/38;m=st.mean(v);b=BYT[l]
        tb+=nn*b;tt+=nn*m
        ideal=b/(0.85*PEAK[d]*1e3)
        r=nn*max(0,m-ideal);rec+=r
    print(f"dev{d}: fp8 time/round {tt/1e3:.2f} ms, bytes/round {tb/1e6:.0f} MB -> {tb/(tt*1e3):.0f} GB/s ({100*tb/(tt*1e3)/PEAK[d]:.1f}%), per layer {tt/LAY[d]/1e3:.2f} ms, bytes/layer {tb/LAY[d]/1e6:.0f} MB; recoverable to 85%: {rec/1e3:.2f} ms")
    tr+=rec
print("total dense recoverable",tr/1e3)
# per class recoverable
for d in range(5):
    for l,v in sorted(res[d].items()):
        nn=len(v)/38;m=st.mean(v);b=BYT[l]; ideal=b/(0.85*PEAK[d]*1e3)
        print(f"  dev{d} {l:9s} mean {m:6.1f} ideal85 {ideal:6.1f} us  rec/round {nn*max(0,m-ideal)/1e3:.3f} ms")
