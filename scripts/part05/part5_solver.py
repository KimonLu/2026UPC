"""Batched SI-unit inverse calculations and first-event predictions for Part 5."""
from dataclasses import replace
from math import pi
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from scripts.common.still_air import Physics, Motion, Settings, spin_basis, PARAMETERS, SCALES

R0 = np.array([0., 0., .11])
OBS = np.array([R0, [1.5,.45,.44], [2.96,.86,.74]])
TIMES = np.array([0.,.05,.10])
LOW = np.array([5.,-30.,.01,.01,-150.,-150.])
HIGH = np.array([55.,30.,30.,1.,150.,150.])
ORIG_LOW = np.array([10.,-20.,.1,.05,-100.,-100.])
ORIG_HIGH = np.array([45.,20.,20.,.6,100.,100.])
EVENTS = ('goal_inside','goal_outside','ground_first','fit_failure','numerical_failure','timeout')


def basis_batch(positions):
    e1 = positions[:,2]-positions[:,0]
    e1 /= np.linalg.norm(e1,axis=1)[:,None]
    reference=np.tile([0.,0.,1.],(len(e1),1))
    reference[np.abs(e1[:,2])>.99]=[1.,0.,0.]
    e2=np.cross(reference,e1)
    e2/=np.linalg.norm(e2,axis=1)[:,None]
    return np.stack([e1,e2,np.cross(e1,e2)],axis=2)


def coefficients(parameters,basis,wind,physics=Physics()):
    factor=physics.density*physics.area/(2*physics.mass)
    omega=np.einsum('nij,nj->ni',basis,np.column_stack([np.zeros(len(parameters)),parameters[:,4:]]))
    return factor*parameters[:,3],factor*physics.radius*omega,np.asarray(wind),physics.effective_gravity


def rhs_batch(state,coef):
    drag,lift,wind,gravity=coef
    u=state[:,3:]-wind
    acceleration=-drag[:,None]*np.linalg.norm(u,axis=1)[:,None]*u+np.cross(lift,u)
    acceleration[:,2]-=gravity
    return np.column_stack([state[:,3:],acceleration])


def rk4(state,dt,coef):
    dt=np.asarray(dt)
    if dt.ndim: dt=dt[:,None]
    k1=rhs_batch(state,coef)
    k2=rhs_batch(state+dt*k1/2,coef)
    k3=rhs_batch(state+dt*k2/2,coef)
    k4=rhs_batch(state+dt*k3,coef)
    return state+dt*(k1+2*k2+2*k3+k4)/6


def positions_batch(parameters,basis,times,wind,nstep=10):
    state=np.column_stack([np.broadcast_to(R0,(len(parameters),3)),parameters[:,:3]])
    coef=coefficients(parameters,basis,wind)
    output=[]
    for interval in (1,2):
        dt=(times[:,interval]-times[:,interval-1])/nstep
        for _ in range(nstep): state=rk4(state,dt,coef)
        output.append(state[:,:3].copy())
    return np.stack(output,axis=1)


def residual_jacobian(parameters,basis,times,wind,target,nstep):
    n=len(parameters)
    samples=np.repeat(parameters[:,None,:],7,axis=1)
    eps=2e-5
    for j in range(6): samples[:,j+1,j]+=eps*SCALES[j]
    predicted=positions_batch(samples.reshape(-1,6),np.repeat(basis,7,axis=0),np.repeat(times,7,axis=0),wind,nstep).reshape(n,7,6)
    residual=predicted[:,0]-target.reshape(n,6)
    jac=np.transpose((predicted[:,1:]-predicted[:,:1])/eps,(0,2,1))
    return residual,jac


def fit_batch(positions,times,wind=(0.,0.,0.),nstep=10,guess=None):
    positions=np.asarray(positions,float); times=np.asarray(times,float)
    n=len(positions); basis=basis_batch(positions)
    if np.any(times[:,1]<=0) or np.any(times[:,2]<=times[:,1]):
        raise ValueError('Require ordered positive observation times')
    if guess is None:
        guess=np.array([30.4169588019885,9.4186784579097,6.907937547899,.4323579369902,-11.22804621035,-60.97793411028])
        if np.linalg.norm(wind)==0:
            guess=np.array([30.4154362929875,9.4173315062224,6.9073033009944,.3789778889117,-9.133817420207,-59.392229708203])
    p=np.broadcast_to(guess,(n,6)).copy()
    target=positions[:,1:]
    for _ in range(8):
        residual,jac=residual_jacobian(p,basis,times,wind,target,nstep)
        if np.max(np.abs(residual))<2e-11: break
        delta=np.linalg.solve(jac,-residual[...,None])[...,0]
        delta=np.clip(delta,-2.,2.)
        p+=delta*SCALES
    unconstrained=p.copy()
    original_outside=np.any((p<ORIG_LOW)|(p>ORIG_HIGH),axis=1)
    bounded=np.any((p<LOW)|(p>HIGH)|~np.isfinite(p),axis=1)
    success=np.ones(n,bool); fallback=np.zeros(n,bool)
    indices=np.flatnonzero(bounded)
    if len(indices):
        x=np.clip(p[indices],LOW,HIGH)/SCALES
        lo=LOW/SCALES; hi=HIGH/SCALES
        lam=np.full(len(indices),1e-8)
        for _ in range(45):
            res,j=residual_jacobian(x*SCALES,basis[indices],times[indices],wind,target[indices],nstep)
            gradient=np.einsum('nki,nk->ni',j,res)
            fixed=((x<=lo+1e-10)&(gradient>0))|((x>=hi-1e-10)&(gradient<0))
            free=~fixed
            normal=np.einsum('nki,nkj->nij',j,j)*free[:,:,None]*free[:,None,:]
            normal+=np.eye(6)[None]*(lam[:,None]+fixed)[:,None,:]
            delta=np.linalg.solve(normal,(-gradient*free)[...,None])[...,0]
            candidate=np.clip(x+delta,lo,hi)
            candres=positions_batch(candidate*SCALES,basis[indices],times[indices],wind,nstep).reshape(-1,6)-target[indices].reshape(-1,6)
            accept=np.sum(candres**2,axis=1)<=np.sum(res**2,axis=1)+1e-24
            x=np.where(accept[:,None],candidate,x)
            lam=np.where(accept,np.maximum(1e-12,lam/3),np.minimum(1e2,lam*10))
            if np.max(np.abs(gradient*free))<1e-10: break
        p[indices]=x*SCALES
        res,j=residual_jacobian(p[indices],basis[indices],times[indices],wind,target[indices],nstep)
        grad=np.einsum('nki,nk->ni',j,res)
        fixed=((x<=lo+1e-8)&(grad>0))|((x>=hi-1e-8)&(grad<0))
        unresolved=np.max(np.abs(grad*(~fixed)),axis=1)>2e-8
        for local in np.flatnonzero(unresolved):
            i=indices[local]; fallback[i]=True
            def residual1(scaled):
                return (positions_batch((scaled*SCALES)[None],basis[i:i+1],times[i:i+1],wind,nstep)[0]-target[i]).ravel()
            result=least_squares(residual1,p[i]/SCALES,bounds=(lo,hi),ftol=1e-11,xtol=1e-11,gtol=1e-11,max_nfev=100)
            p[i]=result.x*SCALES; success[i]=result.success
    residual=positions_batch(p,basis,times,wind,nstep)-target
    active=np.any(np.isclose(p,LOW,atol=2e-6)|np.isclose(p,HIGH,atol=2e-6),axis=1)
    success &= np.isfinite(p).all(axis=1)&np.isfinite(residual).all(axis=(1,2))
    return dict(parameters=p,basis=basis,residual=residual,success=success,expanded=original_outside,active=active,fallback=fallback,unconstrained=unconstrained)


def hermite(old,new,dt,fraction):
    s=np.asarray(fraction)[:,None]
    return (2*s**3-3*s**2+1)*old[:,:3]+(s**3-2*s**2+s)*dt*old[:,3:]+(-2*s**3+3*s**2)*new[:,:3]+(s**3-s**2)*dt*new[:,3:]


def root_fraction(old,new,dt,axis,target):
    lo=np.zeros(len(old)); hi=np.ones(len(old))
    increasing=new[:,axis]>old[:,axis]
    for _ in range(32):
        mid=(lo+hi)/2
        value=hermite(old,new,dt,mid)[:,axis]
        below=(value<target)==increasing
        lo=np.where(below,mid,lo); hi=np.where(below,hi,mid)
    return (lo+hi)/2


def predict_batch(parameters,basis,wind=(0.,0.,0.),step=.004):
    n=len(parameters); coef=coefficients(parameters,basis,wind)
    state=np.column_stack([np.broadcast_to(R0,(n,3)),parameters[:,:3]])
    event=np.full(n,'timeout',dtype='<U20'); goal=np.full((n,3),np.nan); stop=np.full(n,2.)
    done=state[:,5]<=0; event[done]='ground_first'; stop[done]=0
    for k in range(round(2/step)):
        if done.all(): break
        new=rk4(state,step,coef)
        crossing=(~done)&(state[:,0]<11)&(new[:,0]>=11)
        ground=(~done)&(state[:,2]>=.11)&(new[:,2]<.11)
        picked=np.flatnonzero(crossing|ground)
        if len(picked):
            fg=np.full(len(picked),np.inf); fb=fg.copy()
            a=crossing[picked]; b=ground[picked]
            if a.any(): fg[a]=root_fraction(state[picked[a]],new[picked[a]],step,0,11.)
            if b.any(): fb[b]=root_fraction(state[picked[b]],new[picked[b]],step,2,.11)
            reach=fg<fb
            contact=picked[~reach]; event[contact]='ground_first'
            stop[picked]=k*step+np.minimum(fg,fb)*step
            if reach.any():
                indices=picked[reach]
                point=hermite(state[indices],new[indices],step,fg[reach])
                goal[indices]=np.column_stack([stop[indices],point[:,1:3]])
                inside=(np.abs(point[:,1])<3.66)&(point[:,2]>0)&(point[:,2]<2.44)
                event[indices]=np.where(inside,'goal_inside','goal_outside')
            done[picked]=True
        failure=(~done)&~np.isfinite(new).all(axis=1)
        event[failure]='numerical_failure'; done[failure]=True
        # Freeze completed trajectories to avoid irrelevant long-time overflow.
        state=np.where(done[:,None],state,new)
    return dict(event=event,goal=goal,stop_time=stop)


def baseline(positions,times,model):
    positions=np.asarray(positions); times=np.asarray(times)
    d=positions[:,1:]-positions[:,:1]; t=times[:,1:]
    if model=='B-G':
        a=np.broadcast_to([0.,0.,-9.81],(len(d),3)).copy()
        v=np.sum(t[:,:,None]*(d-.5*t[:,:,None]**2*a[:,None]),axis=1)/np.sum(t**2,axis=1)[:,None]
    else:
        a=2*(d[:,1]/t[:,1,None]-d[:,0]/t[:,0,None])/(t[:,1]-t[:,0])[:,None]
        v=d[:,0]/t[:,0,None]-.5*a*t[:,0,None]
    vx=v[:,0]; ax=a[:,0]; disc=vx**2+22*ax
    tg=np.full(len(v),np.inf)
    straight=np.abs(ax)<1e-10; tg[straight&(vx>0)]=11/vx[straight&(vx>0)]
    curved=(~straight)&(disc>0)
    tg[curved]=22/(vx[curved]+np.sqrt(disc[curved]))
    tg[tg<=0]=np.inf
    tc=np.full(len(v),np.inf)
    downward=v[:,2]<=0; tc[downward]=0
    falling=(a[:,2]<0)&(~downward); tc[falling]=-2*v[falling,2]/a[falling,2]
    ground=tc<=tg; reached=np.isfinite(tg)&(~ground)
    event=np.full(len(v),'timeout',dtype='<U20'); event[ground]='ground_first'
    goal=np.full((len(v),3),np.nan)
    point=R0+tg[reached,None]*v[reached]+.5*tg[reached,None]**2*a[reached]
    inside=(np.abs(point[:,1])<3.66)&(point[:,2]>0)&(point[:,2]<2.44)
    event[reached]=np.where(inside,'goal_inside','goal_outside')
    goal[reached]=np.column_stack([tg[reached],point[:,1:3]])
    residual=R0+t[:,:,None]*v[:,None]+.5*t[:,:,None]**2*a[:,None]-positions[:,1:]
    return dict(event=event,goal=goal,velocity=v,acceleration=a,residual=residual,stop_time=np.minimum(tc,tg))


def rotate_spin(omega,axis,rate,time):
    omega=np.asarray(omega); axis=np.asarray(axis)
    angle=rate*time
    return omega*np.cos(angle)+np.cross(axis,omega)*np.sin(angle)+axis*np.dot(axis,omega)*(1-np.cos(angle))


def make_rhs(parameters,basis,physics=Physics(),k_l=1.,wake=None,axis=None,rate=0.,wind=None):
    omega=basis@np.r_[0.,parameters[4:]]
    air=np.array(physics.wind if wind is None else wind)
    factor=physics.density*physics.area/(2*physics.mass)
    def rhs(t,state):
        u=state[3:]-air
        spin=omega if axis is None else rotate_spin(omega,axis,rate,t)
        acc=-factor*parameters[3]*np.linalg.norm(u)*u+factor*k_l*physics.radius*np.cross(spin,u)
        acc[2]-=physics.effective_gravity
        if wake is not None:
            amplitude,frequency,phase=wake
            acc[1]+=amplitude*np.sin(2*pi*frequency*t+phase)
        return np.r_[state[3:],acc]
    return rhs


def general_flight(parameters,basis,physics=Physics(),k_l=1.,wake=None,axis=None,rate=0.,gust=None,max_step=.01,rtol=1e-10):
    state=np.r_[R0,parameters[:3]]; start=0.
    if state[5]<=0: return dict(event='ground_first',goal=None,stop_time=0.)
    edges=[0.,2.]
    if gust is not None: edges=sorted(set([0.,gust[0],gust[0]+gust[1],2.]))
    for left,right in zip(edges[:-1],edges[1:]):
        wind=np.array(physics.wind,float)
        if gust is not None and gust[0]<=left<gust[0]+gust[1]-1e-12: wind+=gust[2]
        def goal(t,y): return y[0]-11
        def ground(t,y): return np.spacing(.11) if t==0 else y[2]-.11
        goal.direction=1; ground.direction=-1; goal.terminal=ground.terminal=True
        sol=solve_ivp(make_rhs(parameters,basis,physics,k_l,wake,axis,rate,wind),(left,right),state,method='DOP853',rtol=rtol,atol=rtol*.01,max_step=max_step,events=(goal,ground))
        state=sol.y[:,-1]; start=float(sol.t[-1])
        if not sol.success: return dict(event='numerical_failure',goal=None,stop_time=start)
        if len(sol.t_events[1]): return dict(event='ground_first',goal=None,stop_time=start)
        if len(sol.t_events[0]):
            event='goal_inside' if abs(state[1])<3.66 and 0<state[2]<2.44 else 'goal_outside'
            return dict(event=event,goal=[start,float(state[1]),float(state[2])],stop_time=start)
    return dict(event='timeout',goal=None,stop_time=start)


def fit_general(physics=Physics(wind=(2.,0.,0.)),fixed_cd=None,k_l=1.,wake=None,axis=None,rate=0.,guess=None,fit_spin=True):
    basis=spin_basis(OBS[2]-OBS[0])
    if guess is None: guess=np.array([30.4169588,9.41867846,6.90793755,.432357937,-11.2280462,-60.9779341])
    free=np.array([True,True,True,fixed_cd is None,fit_spin,fit_spin])
    fixed=guess.copy()
    if fixed_cd is not None: fixed[3]=fixed_cd
    if not fit_spin: fixed[4:]=0.
    def unpack(x):
        p=fixed.copy(); p[free]=x*SCALES[free]; return p
    def residual(x):
        p=unpack(x)
        sol=solve_ivp(make_rhs(p,basis,physics,k_l,wake,axis,rate),(0.,.1),np.r_[R0,p[:3]],t_eval=TIMES[1:],method='DOP853',rtol=1e-10,atol=1e-12,max_step=.005)
        return (sol.y[:3].T-OBS[1:]).ravel()
    fit=least_squares(residual,np.clip(guess[free],LOW[free],HIGH[free])/SCALES[free],bounds=(LOW[free]/SCALES[free],HIGH[free]/SCALES[free]),ftol=1e-12,xtol=1e-12,gtol=1e-13,max_nfev=150)
    p=unpack(fit.x)
    return dict(parameters=p,basis=basis,residual=fit.fun.reshape(2,3),success=fit.success,active_mask=fit.active_mask)
