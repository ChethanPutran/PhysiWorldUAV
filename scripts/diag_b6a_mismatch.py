import numpy as np, torch
from physiworld_uav.data.dataset import ModelReadyDataset
from physiworld_uav.data.dataset_builder import BuilderConfig, dataset_decimation, load_model_episodes
from physiworld_uav.evaluation.evaluator import collect_errors
from physiworld_uav.models.world_model import build_model
D="data/datasets/physiworld_pilot_v7_v0.7.0"
b=BuilderConfig(decimation=dataset_decimation(D),history=10,horizon=20,obs_mode="instant")
eps=load_model_episodes(D,b,roles=["test_id"])
m=build_model("b6a_future",dt=0.05)
E=collect_errors(m,ModelReadyDataset(eps,b,oracle=True),horizon=20)
pe=E["pe"][:,-1]; ids=E["eids"]
import json
MAN={}
for l in open(D+"/manifest.jsonl"):
    q=json.loads(l); MAN[q["episode_id"]]=q["domain"]["factors"]
np.savez("/workspace/PhysiWorldUAV/outputs/eval/b6a_v7_test_id_err.npz",pe=E["pe"],eids=E["eids"])
rows=[]
for e in eps:
    s=pe[ids==e.episode_id]
    f=MAN[e.episode_id]
    rows.append((np.sqrt((s**2).mean()),f["mass"],f["force_scale"],f.get("motor_time_constant",1),f.get("km_kf_ratio",1)))
r=np.array(rows)
print("n",len(r),"overall rms",np.sqrt((r[:,0]**2).mean()))
for j,n in enumerate(["mass","force_scale","motor_tc","km_kf"],1):
    print(n,"range",r[:,j].min(),r[:,j].max(),"corr with err",np.corrcoef(r[:,j],r[:,0])[0,1])
# thrust-accel mismatch proxy: force_scale/mass-1 
x=r[:,2]/r[:,1]-1
print("corr err vs |fs/mass-1|",np.corrcoef(np.abs(x),r[:,0])[0,1])
for lo,hi in [(0,.03),(.03,.06),(.06,.1),(.1,1)]:
    k=(np.abs(x)>=lo)&(np.abs(x)<hi); print(f"|fs/mass-1| in [{lo},{hi}): n={k.sum()} rms err {np.sqrt((r[k,0]**2).mean()) if k.any() else float('nan'):.3f}")
