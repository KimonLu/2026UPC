"""Reproduce all outline Cases 5A--5F; every Monte Carlo draw is retained."""
import argparse
import csv
import json
import platform
import time
from dataclasses import replace
import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.stats import gaussian_kde
from scripts.common.still_air import Physics, Motion, Settings, simulate, spin_basis
from scripts.part05 import part5_solver as s

from scripts.common.paths import data_dir

DATA = data_dir(5)
SEED=20261107
MODELS=('B-G','B-CA','M-DM/calm','M-DM/wind')


def require_published_ensemble(n):
    """The paper and full audit are for the prescribed 4000-draw experiment."""
    if n != 4000:
        raise ValueError('Publication requires the 4000-sample ensemble. Run scripts/run_part5.py --samples 4000 before generating the paper or its full numerical audit.')


def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def summarize(prediction,fit=None,n=None):
    n=len(prediction['event']) if n is None else n
    event=prediction['event'][:n]; goal=prediction['goal'][:n]
    reached=np.isfinite(goal).all(axis=1)
    result={'n':n,'events':{kind:int(sum(event==kind)) for kind in s.EVENTS},'n_crossings':int(sum(reached))}
    if fit is not None:
        result.update(expanded=int(sum(fit['expanded'][:n])),active_bound=int(sum(fit['active'][:n])),fallback=int(sum(fit['fallback'][:n])),max_fit_rms_m=float(np.max(np.sqrt(np.mean(fit['residual'][:n]**2,axis=(1,2))))))
        residual_rms=np.sqrt(np.mean(fit['residual'][:n]**2,axis=(1,2)))
        result['median_fit_rms_m']=float(np.median(residual_rms))
        result['q95_fit_rms_m']=float(np.quantile(residual_rms,.95))
    points=goal[reached]
    if len(points):
        result.update(mean=points.mean(0).tolist(),sd=points.std(0,ddof=1).tolist() if len(points)>1 else [0.,0.,0.],q025=np.quantile(points,.025,axis=0).tolist(),q975=np.quantile(points,.975,axis=0).tolist())
    if len(points)>6 and np.max(np.std(points[:,1:],axis=0))>1e-9:
        yz=points[:,1:];cov=np.cov(yz.T);mean=yz.mean(0)
        eigenvalues,eigenvectors=np.linalg.eigh(cov)
        retained=eigenvalues>max(1e-24,eigenvalues[-1]*1e-12)
        dimension=int(sum(retained))
        projected=(yz-mean)@eigenvectors[:,retained]
        distance=np.sum(projected**2/eigenvalues[retained],axis=1)
        q=float(np.quantile(distance,.95))
        area=float(np.pi*q*np.sqrt(np.prod(eigenvalues))) if dimension==2 else 0.
        result.update(region_dimension=dimension,covariance_yz=cov.tolist(),ellipse_radius_squared=q,ellipse_area_m2=area,ellipse_sample_fraction=float(np.mean(distance<=q)))
    else: result['ellipse_area_m2']=0.
    return result


def kde_mode(points):
    yz=points[np.isfinite(points).all(axis=1),1:]
    kde=gaussian_kde(yz.T,bw_method='scott')
    low,high=np.quantile(yz,[.001,.999],axis=0)
    yy,zz=np.meshgrid(np.linspace(low[0],high[0],65),np.linspace(low[1],high[1],65))
    grid=np.column_stack([yy.ravel(),zz.ravel()]);density=kde(grid.T)
    candidates=np.argsort(density)[-8:]
    fits=[minimize(lambda x:-float(kde(x[:,None])[0]),grid[i],method='BFGS',options={'gtol':1e-7}) for i in candidates]
    best=min(fits,key=lambda f:f.fun)
    return dict(y_m=float(best.x[0]),z_m=float(best.x[1]),bandwidth_factor=float(kde.factor),conditioning='all successful goal-plane crossings; not restricted to goals')


def compute_mc(n=4000):
    rng=np.random.default_rng(SEED)
    standard_position=rng.normal(size=(n,2,3))
    uniform_position=rng.uniform(-.005,.005,(n,2,3))
    independent_time=rng.normal(size=(n,2)); common_time=rng.normal(size=(n,1))
    settings=[('position','gaussian',sigma) for sigma in [.001,.005,.010,.005/np.sqrt(3)]]+[('position','uniform',.005)]
    settings += [('time',kind,sigma) for kind in ('independent','common') for sigma in [.0005,.001]]
    summaries={}; ensembles={}; sample_rows=[]; stability=[]; checks=[]
    deterministic={}
    for model in MODELS:
        if model.startswith('B-'): pred=s.baseline(s.OBS[None],s.TIMES[None],model)
        else:
            wind=(2.,0.,0.) if model.endswith('wind') else (0.,0.,0.)
            fit=s.fit_batch(s.OBS[None],s.TIMES[None],wind)
            pred=s.predict_batch(fit['parameters'],fit['basis'],wind)
        deterministic[model]=summarize(pred)
    for family,kind,scale in settings:
        positions=np.broadcast_to(s.OBS,(n,3,3)).copy();times=np.broadcast_to(s.TIMES,(n,3)).copy()
        if family=='position': positions[:,1:]+=scale*standard_position if kind=='gaussian' else uniform_position
        else: times[:,1:]+=scale*(independent_time if kind=='independent' else common_time)
        rejected=0
        invalid=(times[:,1]<=0)|(times[:,2]<=times[:,1])
        while invalid.any():
            rejected+=int(sum(invalid)); count=sum(invalid)
            perturb=rng.normal(size=(count,2 if kind=='independent' else 1))*scale
            times[invalid,1:]=s.TIMES[1:]+perturb
            invalid=(times[:,1]<=0)|(times[:,2]<=times[:,1])
        condition=f'{family}/{kind}/{scale:g}'
        ensembles[condition]={'positions':positions,'times':times}
        for model in MODELS:
            start=time.perf_counter();fit=None
            if model.startswith('B-'): prediction=s.baseline(positions,times,model)
            else:
                wind=(2.,0.,0.) if model.endswith('wind') else (0.,0.,0.)
                fit=s.fit_batch(positions,times,wind)
                prediction=s.predict_batch(fit['parameters'],fit['basis'],wind)
                prediction['event'][~fit['success']]='fit_failure';prediction['goal'][~fit['success']]=np.nan
                # Independent checks include extrema, active bounds and ordinary samples.
                selected=np.unique(np.r_[np.linspace(0,n-1,min(n,12),dtype=int),np.argmin(fit['parameters'],axis=0),np.argmax(fit['parameters'],axis=0),np.flatnonzero(fit['active'])[:4]])
                max_goal=0.;max_residual=0.
                for index in selected:
                    p=fit['parameters'][index];omega=fit['basis'][index]@np.r_[0.,p[4:]]
                    ref=simulate(s.R0,Motion(p[:3],p[3],omega),Physics(wind=wind),Settings())
                    if ref.event!=prediction['event'][index]: raise AssertionError((condition,model,index,'event mismatch'))
                    if ref.event.startswith('goal_'):
                        max_goal=max(max_goal,float(np.max(np.abs(prediction['goal'][index]-[ref.time,*ref.states[-1,1:3]]))))
                    original=ref.sample(times[index,1:])[:,:3]-positions[index,1:]
                    max_residual=max(max_residual,float(np.max(np.abs(original-fit['residual'][index]))))
                if max_goal>1e-5 or max_residual>1e-7: raise AssertionError((max_goal,max_residual))
                checks.append(dict(condition=condition,model=model,n_checked=len(selected),max_event_error=max_goal,max_observation_error=max_residual))
            key=condition+'/'+model
            summary=summarize(prediction,fit);summary['invalid_time_draws']=rejected
            prefixes={str(size):summarize(prediction,fit,size) for size in [200,2000,4000] if size<=n}
            summary['prefixes']=prefixes
            if family=='position' and kind=='gaussian' and scale==.005 and model=='M-DM/wind':
                summary['kde_mode']=kde_mode(prediction['goal'])
                # Bandwidth sensitivity is reported separately rather than extra precision.
                yz=prediction['goal'][np.isfinite(prediction['goal']).all(axis=1),1:]
                summary['bandwidth_mode_check']=[]
                for mult in [.8,1.2]:
                    kde=gaussian_kde(yz.T,bw_method=lambda x:mult*x.scotts_factor())
                    initial=np.array([summary['kde_mode']['y_m'],summary['kde_mode']['z_m']])
                    optimum=minimize(lambda x:-float(kde(x[:,None])[0]),initial,method='BFGS')
                    summary['bandwidth_mode_check'].append(dict(multiplier=mult,y_m=float(optimum.x[0]),z_m=float(optimum.x[1])))
            summaries[key]=summary
            if n>=4000:
                first=prefixes['2000'];full=prefixes['4000']
                se=np.array(full['sd'])/np.sqrt(full['n_crossings'])
                relative_area=(full['ellipse_area_m2']/first['ellipse_area_m2']-1) if first['ellipse_area_m2']>0 else 0. if full['ellipse_area_m2']==0 else None
                stability.append(dict(condition=condition,model=model,mean_shift_t_s=full['mean'][0]-first['mean'][0],mean_shift_y_m=full['mean'][1]-first['mean'][1],mean_shift_z_m=full['mean'][2]-first['mean'][2],area_relative_change=relative_area,mc_se_t_s=se[0],mc_se_y_m=se[1],mc_se_z_m=se[2]))
            ensembles[condition][model+'_goal']=prediction['goal'];ensembles[condition][model+'_event']=prediction['event']
            if fit is not None:
                ensembles[condition][model+'_parameters']=fit['parameters'];ensembles[condition][model+'_basis']=fit['basis'];ensembles[condition][model+'_active']=fit['active'];ensembles[condition][model+'_expanded']=fit['expanded'];ensembles[condition][model+'_residual']=fit['residual'];ensembles[condition][model+'_fit_success']=fit['success']
            for i in range(n):
                row=dict(condition=condition,model=model,sample=i,event=prediction['event'][i],t_g_s=prediction['goal'][i,0],y_g_m=prediction['goal'][i,1],z_g_m=prediction['goal'][i,2],expanded=False if fit is None else fit['expanded'][i],active_bound=False if fit is None else fit['active'][i],rms_m=float(np.sqrt(np.mean((prediction if fit is None else fit)['residual'][i]**2))))
                sample_rows.append(row)
            print(key,'sec',round(time.perf_counter()-start,2),'events',summary['events'],'active',summary.get('active_bound',0),flush=True)
    packed={}
    for j,(key,values) in enumerate(ensembles.items()):
        for name,array in values.items(): packed[f'{j}_{name}']=array
    np.savez_compressed(DATA/'ensembles.npz',**packed)
    write_csv(DATA/'monte_carlo_samples.csv',sample_rows)
    if stability: write_csv(DATA/'sample_stability.csv',stability)
    write_csv(DATA/'numerical_checks.csv',checks)
    return dict(deterministic=deterministic,summaries=summaries,ensemble_conditions=list(ensembles),n=n)


def compute_scenarios():
    physics=Physics(wind=(2.,0.,0.));reference=s.fit_general(physics);p0=reference['parameters'];basis=reference['basis']
    reference_flight=s.general_flight(p0,basis,physics);g0=np.array(reference_flight['goal'])
    rows=[]
    def record(family,name,value,fit,phys=physics,**options):
        flight=s.general_flight(fit['parameters'],fit['basis'],phys,**options)
        residual=fit['residual']; p=fit['parameters'];goal=flight['goal']
        row=dict(family=family,parameter=name,value=value,event=flight['event'],rms_mm=1000*float(np.sqrt(np.mean(residual**2))),t_g_s=None if goal is None else goal[0],y_g_m=None if goal is None else goal[1],z_g_m=None if goal is None else goal[2],delta_y_m=None if goal is None else goal[1]-g0[1],delta_z_m=None if goal is None else goal[2]-g0[2],delta_t_s=None if goal is None else goal[0]-g0[0],cd=p[3],v0x=p[0],v0y=p[1],v0z=p[2],omega2=p[4],omega3=p[5],spin_start_rad_s=float(np.linalg.norm(p[4:])),fit_success=bool(fit['success']),rho_cd_over_m=phys.density*p[3]/phys.mass)
        rows.append(row);return row
    property_values={'mass':[.41,.43,.45],'density':[1.1,1.2,1.3],'viscosity':[1.70e-5,1.81e-5,1.90e-5],'cd':[.1,.2,.3,.5],'k_l':[.5,1.,1.5],'gravity':[9.78,9.81,9.83],'wind_z':[-1.,0.,1.]}
    for name,values in property_values.items():
        for value in values:
            phys=replace(physics,**{name:value}) if name in ('mass','density','viscosity','gravity') else physics
            if name=='wind_z':phys=replace(physics,wind=(2.,0.,value))
            options={'fixed_cd':value} if name=='cd' else {'k_l':value} if name=='k_l' else {}
            fit=s.fit_general(phys,guess=p0,**options)
            row=record('5C',name,value,fit,phys,k_l=value if name=='k_l' else 1.)
            row['effective_gravity']=phys.effective_gravity
    for direction in [(0.,2.,0.),(0.,-2.,0.),(2.,0.,0.)]:
        for start in [.15,.25]:
            for duration in [.02,.05,.1]:
                row=record('5D','gust',str(direction),reference,gust=(start,duration,direction))
                row.update(start_s=start,duration_s=duration)
    no_spin=s.fit_general(physics,fit_spin=False,guess=p0)
    wake_reference=s.general_flight(no_spin['parameters'],no_spin['basis'],physics)
    for amplitude,frequency,phase in [(0.,5.,0.)]+[(a,f,phase) for a in [1.,3.] for f in [5.,10.] for phase in [0.,np.pi/2,np.pi,3*np.pi/2]]:
        wake=(amplitude,frequency,phase)
        fit=s.fit_general(physics,wake=wake,fit_spin=False,guess=no_spin['parameters'])
        row=record('5E','wake',amplitude,fit,wake=wake)
        row.update(frequency_hz=frequency,phase_rad=phase,delta_y_from_drag_m=row['y_g_m']-wake_reference['goal'][1],delta_z_from_drag_m=row['z_g_m']-wake_reference['goal'][2])
    for name,axis in [('z',[0.,0.,1.]),('y',[0.,1.,0.])]:
        for degrees in [0.,-30.,-10.,10.,30.]:
            rate=np.deg2rad(degrees)
            fit=s.fit_general(physics,axis=axis,rate=rate,guess=p0)
            row=record('5F','axis_'+name,degrees,fit,axis=axis,rate=rate)
            omega=fit['basis']@np.r_[0.,fit['parameters'][4:]]
            end=s.rotate_spin(omega,axis,rate,row['t_g_s'])
            row['axis_angle_deg']=float(np.rad2deg(np.arccos(np.clip(np.dot(omega,end)/np.linalg.norm(omega)**2,-1,1))))
    columns=list(dict.fromkeys(key for row in rows for key in row))
    write_csv(DATA/'scenarios.csv',[{key:row.get(key) for key in columns} for row in rows])
    return dict(reference_goal=g0.tolist(),reference_parameters=p0.tolist(),rows=rows)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--samples',type=int,default=4000);parser.add_argument('--scenarios-only',action='store_true');args=parser.parse_args()
    DATA.mkdir(parents=True,exist_ok=True)
    config=dict(seed=SEED,n=args.samples,position_sigma_m=[0.,.001,.005,.01],rounding_half_width_m=.005,time_sigma_s=[.0005,.001],nominal_times=s.TIMES.tolist(),positions=s.OBS.tolist(),models=MODELS,main_wind_m_s=[2.,0.,0.],observation_rk4_steps_per_interval=10,prediction_rk4_step_s=.004,expanded_bounds=[s.LOW.tolist(),s.HIGH.tolist()],region='empirical Mahalanobis 95% ellipse conditional on successful goal-plane crossing; not a calibrated coverage claim',parameter_sampling='none; 5C--5F are prescribed scenarios, not random priors')
    config.update(gaussian_rounding_matched_sigma_m=.005/np.sqrt(3),published_n=4000)
    (DATA/'config.json').write_text(json.dumps(config,indent=2)+'\n',encoding='utf-8')
    if not args.scenarios_only:
        mc=compute_mc(args.samples)
        (DATA/'monte_carlo_summary.json').write_text(json.dumps(mc,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    scenarios=compute_scenarios()
    (DATA/'scenario_summary.json').write_text(json.dumps(scenarios,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    (DATA/'runtime.json').write_text(json.dumps(dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__),indent=2)+'\n',encoding='utf-8')
    print('Complete.',flush=True)


if __name__=='__main__':main()
