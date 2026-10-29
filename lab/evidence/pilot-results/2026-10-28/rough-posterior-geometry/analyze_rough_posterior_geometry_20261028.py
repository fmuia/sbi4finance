"""Describe existing samples; no posterior/emulator fitting or model calls."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from scipy.special import logit
from scipy.stats import skew,kurtosis,norm,spearmanr,chi2

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def marginal(a,bounds):
 rows={}
 for j,name in enumerate(('H','eta','rho')):
  v=a[:,j];lo,hi=bounds[j];sd=v.std(ddof=1)
  rows[name]={'mean':float(v.mean()),'sd':float(sd),'skewness':float(skew(v,bias=False)),
   'excess_kurtosis':float(kurtosis(v,fisher=True,bias=False)),
   'quantiles_01_05_25_50_75_95_99':np.quantile(v,[.01,.05,.25,.5,.75,.95,.99]).tolist(),
   'mass_lower_1pct_prior_width':float(np.mean(v<lo+.01*(hi-lo))),
   'mass_upper_1pct_prior_width':float(np.mean(v>hi-.01*(hi-lo))),
   'matched_gaussian_mass_outside_bounds':float(norm.cdf((lo-v.mean())/sd)+norm.sf((hi-v.mean())/sd))}
 return rows
def dependence(chains):
 a=chains.reshape(-1,3);z=(a-a.mean(0))/a.std(0)
 result={'pearson':np.corrcoef(a.T).tolist(),'spearman':spearmanr(a).statistic.tolist(),'pairs':[]}
 for i,j in [(0,1),(0,2),(1,2)]:
  errors=[[],[]]
  for hold in range(len(chains)):
   test=z[hold*chains.shape[1]:(hold+1)*chains.shape[1]]
   train=np.concatenate([z[:hold*chains.shape[1]],z[(hold+1)*chains.shape[1]:]])
   for order in (1,2):
    design=np.column_stack([train[:,i]**k for k in range(order+1)])
    beta=np.linalg.lstsq(design,train[:,j],rcond=None)[0]
    prediction=np.column_stack([test[:,i]**k for k in range(order+1)])@beta
    errors[order-1].append(float(np.mean((test[:,j]-prediction)**2)))
  linear=np.column_stack([np.ones(len(z)),z[:,i]])
  beta=np.linalg.lstsq(linear,z[:,j],rcond=None)[0];resid=z[:,j]-linear@beta
  edges=np.quantile(z[:,i],np.linspace(0,1,11));bins=[]
  for b in range(10):
   take=(z[:,i]>=edges[b])&((z[:,i]<=edges[b+1]) if b==9 else (z[:,i]<edges[b+1]))
   bins.append({'x_mean':float(a[take,i].mean()),'y_mean':float(a[take,j].mean()),
                'linear_residual_mean_in_residual_sd':float(resid[take].mean()/resid.std()),
                'rows':int(take.sum())})
  result['pairs'].append({'x':('H','eta','rho')[i],'y':('H','eta','rho')[j],
   'held_chain_linear_rmse':float(np.sqrt(np.mean(errors[0]))),
   'held_chain_quadratic_rmse':float(np.sqrt(np.mean(errors[1]))),
   'quadratic_rmse_reduction_fraction':float(1-np.sqrt(np.mean(errors[1])/np.mean(errors[0]))),
   'bins':bins,'scope':'Descriptive fourfold held-chain regression, not a posterior mode test or an acceptance threshold.'})
 return result
def main():
 p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
 args=p.parse_args();start=time.perf_counter();args.out.mkdir(parents=True,exist_ok=False)
 inputs=json.loads(args.inputs.read_text());bounds=np.asarray(inputs['bounds']);data={};records=[]
 for case in range(3):
  entry=inputs['cases'][case]
  for kind in ('reference','neural'):
   f=Path(entry[kind]['path'])
   if sha(f)!=entry[kind]['sha256']:raise ValueError('Saved sample identity mismatch')
   with np.load(f,allow_pickle=False) as archive:a=archive['parameters'].copy()
   expected=(4,8000,3) if kind=='reference' else (10000,3)
   if a.shape!=expected or not np.isfinite(a).all():raise ValueError('Unexpected saved sample axes')
   pooled=a.reshape(-1,3);u=(pooled-bounds[:,0])/(bounds[:,1]-bounds[:,0])
   interior=np.all((u>0)&(u<1),axis=1)
   if kind=='reference' and not interior.all():raise ValueError('Reference has noninterior saved draws')
   transformed=logit(u[interior])
   row={'case':case,'kind':kind,'shape':list(a.shape),'input_sha256':entry[kind]['sha256'],
        'physical':marginal(pooled,bounds),'logit_skewness':skew(transformed,axis=0,bias=False).tolist(),
        'logit_excess_kurtosis':kurtosis(transformed,axis=0,fisher=True,bias=False).tolist(),
        'noninterior_rows_excluded_from_logit':int((~interior).sum())}
   if kind=='reference':
    row['physical_dependence']=dependence(a)
    row['logit_dependence']=dependence(transformed.reshape(a.shape))
    row['chainwise_physical_skewness']=skew(a,axis=1,bias=False).tolist()
   records.append(row);data[(case,kind)]=a
 report={'format':'rough_saved_posterior_geometry_v1','scientific_acceptance':False,'no_new_model_calls':True,
  'inputs':inputs,'analysis_source_sha256':sha(__file__),'records':records,
  'scope':'Geometry of existing approximate deterministic-emulator PyMC targets and existing neural posterior outputs, not exact rough-model truth. Physical and logit coordinates differ. Boundary asymmetry is not proof of curved ridges or multimodality. No normality p-values or mode-count acceptance tests.',
  'analysis_seconds_before_plotting':time.perf_counter()-start}
 (args.out/'geometry.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 # Plotting uses a deterministic stride only; statistics above use every draw.
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from matplotlib.lines import Line2D
 from matplotlib.patches import Ellipse
 plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
 names=['H','eta','rho'];blue='#23618f';red='#bb4a36';ink='#222b33'
 case=2;chain=data[(case,'reference')];a=chain.reshape(-1,3);shown=chain[:,::4,:].reshape(-1,3)
 limits=[]
 for j in range(3):
  q=np.quantile(a[:,j],[.001,.999]);pad=.12*(q[1]-q[0]);lo,hi=q[0]-pad,q[1]+pad
  if j==1:hi=max(hi,3.5+.15*a[:,j].std())
  limits.append((lo,hi))
 fig,axes=plt.subplots(3,3,figsize=(9.4,9.0),layout='constrained')
 for row in range(3):
  for col in range(3):
   ax=axes[row,col]
   if row<col:ax.axis('off');continue
   if row==col:
    ax.hist(a[:,col],bins=55,density=True,color=ink,alpha=.65)
    xx=np.linspace(*limits[col],300);ax.plot(xx,norm.pdf(xx,a[:,col].mean(),a[:,col].std(ddof=1)),color=red,lw=1.8,ls='--')
    if col==1:ax.axvline(3.5,color='#777777',lw=1.4);ax.text(.03,.95,'upper bound 3.5',transform=ax.transAxes,va='top',fontsize=10)
    ax.set_xlim(limits[col]);ax.set_yticks([])
   else:
    ax.hexbin(shown[:,col],shown[:,row],gridsize=42,mincnt=1,cmap='Blues',rasterized=True)
    mean=a[:,[col,row]].mean(0);cov=np.cov(a[:,[col,row]].T);vals,vecs=np.linalg.eigh(cov);order=np.argsort(vals)[::-1];vals=vals[order];vecs=vecs[:,order]
    angle=np.degrees(np.arctan2(vecs[1,0],vecs[0,0]));radius=np.sqrt(chi2.ppf(.95,2))
    ax.add_patch(Ellipse(mean,2*radius*np.sqrt(vals[0]),2*radius*np.sqrt(vals[1]),angle=angle,fill=False,color=red,ls='--',lw=1.2))
    if row==1:ax.axhline(3.5,color='#777777',lw=1.)
    if col==1:ax.axvline(3.5,color='#777777',lw=1.)
    ax.set_xlim(limits[col]);ax.set_ylim(limits[row])
   if row==2:ax.set_xlabel(names[col])
   elif row!=col:ax.tick_params(labelbottom=False)
   if col==0 and row>0:ax.set_ylabel(names[row])
   elif row!=col:ax.tick_params(labelleft=False)
 r=next(r for r in records if r['case']==2 and r['kind']=='reference')
 axes[0,2].text(0.,.9,'Case 2: known curve\n\nPhysical eta skew: '+f"{r['physical']['eta']['skewness']:.2f}"+'\nGaussian eta mass\nabove prior: '+f"{100*r['physical']['eta']['matched_gaussian_mass_outside_bounds']:.1f}%",va='top',fontsize=12,transform=axes[0,2].transAxes)
 fig.suptitle('Rough Bergomi: boundary-shaped posterior geometry',fontsize=16)
 fig.legend(handles=[Line2D([],[],color=ink,lw=6,label='Approximate-emulator PyMC reference'),Line2D([],[],color=red,ls='--',label='Moment-matched untruncated Gaussian')],loc='outside lower center',ncol=1,frameon=False)
 fig.savefig(args.out/'rough-case2-reference-corner.png',dpi=180);plt.close(fig)
 fig,axes=plt.subplots(2,3,figsize=(12.0,6.4),layout='constrained')
 for case in range(3):
  for row in range(2):
   ax=axes[row,case]
   for kind,color,label in [('reference',ink,'Approximate-emulator PyMC'),('neural',blue,'Saved neural posterior')]:
    eta=data[(case,kind)].reshape(-1,3)[:,1];unit=(eta-.5)/3.
    v=eta if row==0 else logit(unit[(unit>0)&(unit<1)])
    ax.hist(v,bins=55,density=True,histtype='step',color=color,lw=1.6,label=label)
    if kind=='reference':
     xx=np.linspace(*np.quantile(v,[.0005,.9995]),300);ax.plot(xx,norm.pdf(xx,v.mean(),v.std(ddof=1)),color=red,ls='--',lw=1.2,label='Matched Gaussian')
   if row==0:
    ax.set_title(f'Case {case}');ax.set_xlabel('eta, physical coordinate')
    if case>0:ax.axvline(3.5,color='#777777',lw=1.)
   else:ax.set_xlabel('logit of eta in its prior interval')
   if case==0:ax.set_ylabel('Density')
 axes[0,0].legend(fontsize=8,frameon=False)
 fig.suptitle('Asymmetry changes with coordinates; the saved neural fit also differs',fontsize=14)
 fig.savefig(args.out/'rough-eta-physical-and-logit.png',dpi=180);plt.close(fig)
 print(json.dumps({'status':'completed','output':str(args.out),'seconds':time.perf_counter()-start,
  'reference_eta':[{k:r[k] for k in ('case','physical','logit_skewness')} for r in records if r['kind']=='reference']}))
if __name__=='__main__':main()
