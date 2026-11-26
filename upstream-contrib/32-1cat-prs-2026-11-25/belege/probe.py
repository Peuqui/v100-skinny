import subprocess,sys
R='/home/mp/Projekte/vllm-research/1Cat-vLLM'
def conflicts(a,b):
    p=subprocess.run(['git','merge-tree','--write-tree','--name-only','--no-messages',a,b],capture_output=True,text=True,cwd=R)
    lines=p.stdout.strip().split('\n')
    if p.returncode==0: return set()
    # first line tree oid, then conflicted file names (until blank line)
    return set(l for l in lines[1:] if l and '/' in l or l.endswith('.md'))
def hunks(a,b,f):
    p=subprocess.run(['git','merge-tree','--write-tree',a,b],capture_output=True,text=True,cwd=R)
    return p.stdout
branches={'adaptive-verify':'pr-adaptive-verify','pp-pack':'pr-pp-pack-handover','uva':'pr-uva-shared-host-memory','cutlass':'pr-cutlass-int8-build-query-v2','kda':'pr-glm5next-kda-decode-glue'}
local={'#1221':'pr-mrv2-dspark-r1125','#1278':'pr-turing-kernels-r1125','#1282':'pr-triton-sm7x-r1125','#1347':'pr-dsv4-longctx-decode-r1125','#1286':'pr-glm53-r1125','#1348':'pr-glm53-prefill-head-r1125'}
other={'#1379':'refs/probe/pr1379','#1260':'refs/probe/pr1260','#1371':'refs/probe/pr1371','#1589':'refs/probe/pr1589','#1551':'refs/probe/pr1551','#1243':'refs/probe/pr1243','#1031':'refs/probe/pr1031','#1389':'refs/probe/pr1389','#1388':'refs/probe/pr1388','#1377':'refs/probe/pr1377','#1376':'refs/probe/pr1376','#1374':'refs/probe/pr1374','#1370':'refs/probe/pr1370','#822':'refs/probe/pr822','#1587':'refs/probe/pr1587','#1102':'refs/probe/pr1102'}
base='73da3445d'
for k,b in branches.items():
    print('=====',k)
    for name,ref in {**local,**other}.items():
        c_base=conflicts(base,ref); c_b=conflicts(b,ref)
        add=sorted(c_b-c_base)
        print(f'  {name:6} main-conflicts {len(c_base):3d}  branch-conflicts {len(c_b):3d}  added: {add}')
