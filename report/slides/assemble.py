import re
src=open("build_deck_v1.js").read()
new=open("new_blocks.js").read()
new=new.replace("Learned models are 5 to 10 times better than physics alone at 1 s.","Learned models are 4 to 5 times better than physics alone at 1 s in distribution.")
new=new.replace("OOD direction is easy for every learned model; OOD magnitude is hard for all of them","OOD direction errors are lower than in distribution; OOD magnitude is hard for all models")
new=new.replace("the error at 1 s is about 2.5 to 3 times the in-distribution value","the error at 1 s is 1.7 to 2.8 times the in-distribution value")
new=new.replace("wind-range augmentation (done in v7 data)","wider training wind (done in the v7 data)")
new=new.replace("{ x: 0.6, y: 5.5, w: 5.6, h: 1.2, fontSize: 13 });","{ x: 0.6, y: 5.5, w: 5.0, h: 1.3, fontSize: 13 });")
nb={m.group(1):m.group(2) for m in re.finditer(r"//@@(\w+)\n(\{.*?\n\}\n)(?=//@@|\Z)",new,re.S)}
i0=src.index("// ---------------------------------------------------------------- 1 Title")
i1=src.index("(async () =>")
header,region,tail=src[:i0],src[i0:i1],src[i1:]
region=re.sub(r'\nsection\("[^"]*"\);','',region)
region=re.sub(r'\n// -{10,}[^\n]*','',region)
blocks=[b.strip("\n") for b in re.split(r'\n(?=\{\n  const s = )',region.strip("\n")) if b.strip()]
keys={"title":"M.Tech thesis","problem":"Recursive prediction drifts","approach":"Approach: simulate","evolved":"How the project evolved","audit":"What did not work: the first","infra":"Infrastructure: one shared","sim":"Simulation platform","valid":"Physics validation","pilot":"Pilot v0.5 / v0.6: 147","problems":"Problems found while","protocol":"Training and evaluation protocol","pilotres":"Pilot results","diag":"Diagnostics: variants","v7wind":"Pilot v7: wide","v7size":"Pilot v7 dataset","gen":"Data generation cost","quality":"v7 quality report","train":"Training curves on v7","results":"Stage-1 results on v7","traj":"Open-loop rollouts","diagnosis":"Why B1 and the oracle","kv":"Known-vehicle physics prior","worked":"What worked, what did not","limits":"Limitations and threats","future":"Future work","summary":"T(s, \"Summary\""}
eb={}
for k,sub in keys.items():
    c=[b for b in blocks if sub in b]; assert len(c)==1,(k,len(c)); eb[k]=c[0]
order=[("sec","Opening"),"E:title","N:outline",
("sec","Problem and literature"),"E:problem","N:problem","N:dynamics","N:ch1","N:ch2","N:ch3","N:lit1","N:lit2","N:gap","N:rq",
("sec","Approach and model"),"E:approach","N:state","N:arch","N:fphys","N:resid","N:imag","N:loss","N:baselines","N:decisions",
("sec","How the project evolved"),"E:evolved","E:audit",
("sec","Infrastructure and data generation"),"E:infra","N:software","E:sim","N:drone","N:aero","N:wind","N:scen","N:schema","E:valid","E:problems","E:pilot",
("sec","Pilot experiments"),"E:protocol","E:pilotres","E:diag","N:cv",
("sec","Pilot v7"),"E:v7wind","E:v7size","N:v7veh","E:gen","E:quality",
("sec","Stage-1 results"),"E:train","E:results","N:heat","N:horizon","E:traj","E:diagnosis","E:kv","N:kvtrain2","N:kvres","N:kvheat",
("sec","Lessons and next steps"),"E:worked","E:limits","E:future","N:status","E:summary",
("sec","Backup"),"N:qa","N:refs"]
def sub(k,a,b):
    assert a in eb[k],(k,a[:40]); eb[k]=eb[k].replace(a,b)
sub("summary","Known-vehicle training of 9 models on v7 is in progress; evaluation on all roles, seeds and the reliable-horizon experiment follow.","With the known-vehicle prior B3 beats B2 on 6 of 7 test roles (not on OOD wind magnitude); next: seeds, tests of why the residual does not extrapolate, the reliable-horizon experiment.")
sub("summary","The physics prior helps in distribution (B3 0.036 m vs B2 0.105 m, pilot), not yet on strong OOD wind, where a constant-velocity baseline wins. Causes: vehicle mismatch (established), residual extrapolation (hypothesis).","The physics prior helps in distribution and on most OOD roles (v7: B3 0.164 m vs B2 0.220 m on test ID) but not on strong OOD wind magnitude (B3 0.462, B2 0.386, constant velocity 0.282). Causes: vehicle mismatch (established), residual extrapolation (hypothesis).")
sub("worked","Known-vehicle prior: removes the parameter mismatch (oracle 0.41 → 0.02 m)","Known-vehicle prior: removes the parameter mismatch (oracle 0.41 → 0.02 m); B3 0.164 vs B2 0.220 on test ID, 0.270 vs 0.292 on OOD physics")
sub("future","Finish known-vehicle training; evaluate on all roles","Seeds for the known-vehicle runs (training and evaluation done for seed 0)")
out=[];n=0
for o in order:
    if isinstance(o,tuple): out.append('section("%s");'%o[1]); continue
    t,k=o.split(":"); out.append(eb[k] if t=="E" else nb[k]); n+=1
print("slides",n)
arrow='''
function arrow(s, x1, y1, x2, y2, color) {
  s.addShape(pres.shapes.LINE, { x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1), h: Math.abs(y2 - y1), flipH: x2 < x1, flipV: y2 < y1, line: { color: color || INK, width: 2, endArrowType: "triangle" } });
}
'''
open("build_deck.js","w").write(header+arrow+"\n"+"\n".join(out)+"\n\n"+tail)
