import sys,re
src,dst,header=sys.argv[1],sys.argv[2],sys.argv[3]
lines=open(src,encoding='utf-8',errors='replace').read().splitlines()
cues=[];cur=None
for ln in lines:
    m=re.match(r'\s*(\d\d:\d\d:\d\d)\.\d+\s+-->',ln)
    if m:
        cur={'t':m.group(1),'txt':[]};cues.append(cur);continue
    if cur is not None:
        if ln.strip()=='' : cur=None;continue
        cur['txt'].append(ln.strip())
out=[];prev_spk=None;para=None;prev_line=None
for c in cues:
    t=' '.join(c['txt']).strip()
    if not t or t==prev_line: continue
    prev_line=t
    m=re.match(r'^([^:]{1,60}):\s(.*)$',t)
    spk,body=(m.group(1),m.group(2)) if m else (prev_spk,t)
    if spk!=prev_spk or para is None or len(para[2])>1200:
        if para: out.append(para)
        para=[c['t'],spk,body]
    else:
        para[2]+=' '+body
    prev_spk=spk
if para: out.append(para)
with open(dst,'w',encoding='utf-8') as f:
    f.write(header+'\n\n')
    for t,s,b in out:
        f.write(f'[{t}] {s}: {b}\n\n')
print(dst,len(out),'paragraphs')
