"""Independent DOP853 refits and a halved-step ensemble check."""
import json,csv
import numpy as np
from scipy.optimize import least_squares
from scripts.common.still_air import Motion,Physics,Settings,SCALES,position_residuals
from scripts.part05 import part5_solver as s
from scripts.part05.run_part5 import require_published_ensemble

from scripts.common.paths import data_dir

DATA = data_dir(5)
summary=json.loads((DATA/'monte_carlo_summary.json').read_text())
require_published_ensemble(summary['n'])
rows=[]
with np.load(DATA/'ensembles.npz') as arrays:
    for j,condition in enumerate(summary['ensemble_conditions']):
        times=arrays[f'{j}_times'];positions=arrays[f'{j}_positions']
        for model,wind in [('M-DM/calm',(0.,0.,0.)),('M-DM/wind',(2.,0.,0.))]:
            parameters=arrays[f'{j}_{model}_parameters'];basis=arrays[f'{j}_{model}_basis'];residuals=arrays[f'{j}_{model}_residual'];active=arrays[f'{j}_{model}_active']
            indices=np.unique(np.r_[[0,100,1000],np.flatnonzero(active)[:3],np.argmax(np.mean(residuals**2,axis=(1,2)))])
            for index in indices:
                p=parameters[index];b=basis[index];physics=Physics(wind=wind)
                def residual(x):
                    current=x*SCALES;omega=b@np.r_[0.,current[4:]]
                    return position_residuals(times[index],positions[index],Motion(current[:3],current[3],omega),physics,Settings()).ravel()
                refined=least_squares(residual,np.clip(p/SCALES,s.LOW/SCALES,s.HIGH/SCALES),bounds=(s.LOW/SCALES,s.HIGH/SCALES),ftol=1e-12,xtol=1e-12,gtol=1e-13,max_nfev=100)
                before=float(np.sum(residuals[index]**2));after=float(np.sum(refined.fun**2))
                assert before-after<1e-10,(condition,model,index,before,after)
                rows.append(dict(condition=condition,model=model,sample=int(index),batch_sse_m2=before,dop853_sse_m2=after,improvement_m2=before-after,max_parameter_change=float(np.max(abs(p-refined.x*SCALES)))))
    key='position/gaussian/0.005';j=summary['ensemble_conditions'].index(key);model='M-DM/wind'
    indices=np.unique(np.r_[np.linspace(0,3999,200,dtype=int),np.flatnonzero(arrays[f'{j}_{model}_active'])])
    fit=s.fit_batch(arrays[f'{j}_positions'][indices],arrays[f'{j}_times'][indices],(2.,0.,0.),nstep=20)
    prediction=s.predict_batch(fit['parameters'],fit['basis'],(2.,0.,0.),step=.002)
    original=arrays[f'{j}_{model}_goal'][indices]
    delta=float(np.max(np.abs(prediction['goal']-original)))
    assert delta<1e-5,delta
with (DATA/'independent_refits.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
result=dict(independent_refits=len(rows),max_objective_improvement_m2=max(x['improvement_m2'] for x in rows),step_halving_cases=len(indices),max_step_halving_event_error=delta)
(DATA/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
