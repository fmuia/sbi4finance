"""Complete corners from retained draws; no model calls or posterior fitting."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def interior_summary(chains, bounds):
    pooled = chains.reshape(-1, chains.shape[-1])
    unit = (pooled-bounds[:,0])/(bounds[:,1]-bounds[:,0])
    q = np.quantile(unit, [.005,.995], axis=0)
    return {'criterion':'Each central99% marginal interval is strictly inside inner90% of prior range.',
            'unit_prior_quantiles_005_995':q.tolist(),
            'interior_passed':bool(np.all(q[0]>.05) and np.all(q[1]<.95)),
            'minimum_central99_interval_distance_from_prior_boundary_as_fraction_of_prior_width':float(np.min(np.minimum(q[0],1-q[1]))),
            'fraction_of_draws_with_any_parameter_outside_inner90_prior_box':float(np.mean(np.any((unit<.05)|(unit>.95),axis=1)))}


def contours(ax, values, x, y, limits, color):
    h, xe, ye = np.histogram2d(values[:,x], values[:,y], bins=55, range=[limits[x],limits[y]])
    h = gaussian_filter(h, 1.)
    ordered = np.sort(h.ravel())[::-1]
    cumulative = np.cumsum(ordered)/h.sum()
    levels = sorted(set(float(ordered[np.searchsorted(cumulative,p)]) for p in (.9,.5)))
    if len(levels)==2 and levels[0]>0:
        ax.contour((xe[1:]+xe[:-1])/2,(ye[1:]+ye[:-1])/2,h.T,
                   levels=levels,colors=color,linewidths=(1.05,1.4))


def make_plot(chains, neural, names, truth, bounds, title, scope, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator
    d = len(names)
    all_ref = chains.reshape(-1,d)
    shown = chains[:,::4,:].reshape(-1,d)
    limits=[]
    for j in range(d):
        values=all_ref[:,j] if neural is None else np.r_[all_ref[:,j],neural[:,j]]
        q=np.quantile(values,[.001,.999]);padding=.10*(q[1]-q[0])
        limits.append((float(q[0]-padding),float(q[1]+padding)))
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(d,d,figsize=(3.*d,2.7*d),layout='constrained')
    for row in range(d):
        for col in range(d):
            ax=axes[row,col]
            if row<col:ax.axis('off');continue
            if row==col:
                ax.hist(all_ref[:,col],bins=55,range=limits[col],density=True,histtype='step',color='#222b33',lw=1.5)
                if neural is not None:
                    ax.hist(neural[:,col],bins=55,range=limits[col],density=True,histtype='step',color='#22669b',lw=1.35)
                ax.axvline(truth[col],color='#3d8b62',lw=1.1,ls='--')
                ax.set_xlim(limits[col]);ax.set_yticks([]);ax.set_title(names[col],fontsize=11)
            else:
                ax.hexbin(shown[:,col],shown[:,row],gridsize=42,mincnt=1,cmap='Greys',rasterized=True)
                if neural is not None:contours(ax,neural,col,row,limits,'#22669b')
                ax.axvline(truth[col],color='#3d8b62',lw=.8,ls='--')
                ax.axhline(truth[row],color='#3d8b62',lw=.8,ls='--')
                ax.set_xlim(limits[col]);ax.set_ylim(limits[row])
            if row==d-1:ax.set_xlabel(names[col])
            elif row!=col:ax.tick_params(labelbottom=False)
            if col==0 and row>0:ax.set_ylabel(names[row])
            elif row!=col:ax.tick_params(labelleft=False)
            ax.tick_params(labelsize=8)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            if row!=col:ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    legend=[Line2D([],[],color='#222b33',lw=1.6,label='PyMC reference'),
            Line2D([],[],color='#3d8b62',lw=1.2,ls='--',label='Generating parameters')]
    if neural is not None:legend.insert(1,Line2D([],[],color='#22669b',lw=1.5,label='Neural posterior (pair contours: 50% / 90%)'))
    fig.suptitle(title+'\n'+scope,fontsize=14)
    fig.legend(handles=legend,loc='outside lower center',ncol=1,frameon=False,fontsize=10)
    fig.savefig(out,dpi=180)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--neural',type=Path);p.add_argument('--metadata',type=Path,required=True)
    p.add_argument('--truth',required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--model',choices=('heston','rough'),required=True)
    args=p.parse_args();record=json.loads(args.metadata.read_text());task=record['task']
    with np.load(args.reference,allow_pickle=False) as a:chains=a['parameters'].copy()
    neural=None
    if args.neural:
        with np.load(args.neural,allow_pickle=False) as a:neural=a['parameters'].reshape(-1,chains.shape[-1]).copy()
    d=4 if args.model=='heston' else 3
    if chains.shape!=(4,8000,d) or (neural is not None and neural.shape!=(10000,d)):
        raise ValueError('Unexpected declared sample axes')
    truth=np.asarray(json.loads(args.truth));bounds=np.asarray(task['prior_bounds'])
    if truth.shape!=(d,):raise ValueError('Complete generating vector required')
    names=['Initial variance','Long-run variance','Vol-of-vol','Correlation'] if d==4 else ['H','eta','Correlation']
    title=('Heston: all four inferred parameters' if d==4 else 'Rough Bergomi: all three inferred parameters')
    scope=('Interior midpoint development case; independent Fourier / PyMC reference' if d==4 else
           'Interior case 0; known curve supplied; approximate-emulator PyMC reference')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    summary={'format':'complete_interior_corner_v1','model':args.model,'scientific_acceptance':False,
             'reference_shape':list(chains.shape),'neural_shape':None if neural is None else list(neural.shape),
             'parameter_names':task['parameter_names'],'prior_bounds':bounds.tolist(),'truth':truth.tolist(),
             'input_sha256':{str(p):sha(p) for p in [args.reference,args.metadata]+([args.neural] if args.neural else [])},
             'source_sha256':sha(__file__),'reference_interior':interior_summary(chains,bounds),
             'plot_scope':'All inferred parameters and every pair shown. Reference uses all4chains; deterministic stride4 for pair display only; statistics use all32000draws. Axes zoom to posterior, not full prior range. Smoothed neural contours are descriptive.'}
    make_plot(chains,neural,names,truth,bounds,title,scope,args.out)
    summary['figure_sha256']=sha(args.out)
    args.out.with_suffix('.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({'figure':str(args.out),'interior':summary['reference_interior']}))


if __name__=='__main__':main()
