"""Render rough-option and trajectory lesson schematic diagrams; no numerical experiment data."""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
DECK = REPO.parent / 'talk-slides-sbi4finance-overleaf'
font = DECK / 'fonts/Inter-Regular.otf'
if font.exists():
    for font_path in DECK.glob('fonts/Inter-*.otf'):
        font_manager.fontManager.addfont(str(font_path))
    plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font)).get_name()
plt.rcParams.update({'font.size': 13, 'text.color': '#15153B', 'svg.fonttype': 'none',
                     'mathtext.fontset':'dejavusans', 'axes.unicode_minus':False})
INK='#15153B'; PURPLE='#4E2A84'; BLUE='#6969FF'; GREEN='#2E7D43'; PINK='#EC008C'
PALE_PURPLE='#F0ECF6'; PALE_GREEN='#EFF7EF'; GRAY='#63637A'

def setup(w,h):
    fig, ax=plt.subplots(figsize=(w,h))
    fig.patch.set_facecolor('white')
    ax.set(xlim=(0,w), ylim=(0,h)); ax.axis('off')
    fig.subplots_adjust(0,0,1,1)
    return fig,ax

def label(ax,x,y,text,size=13,color=INK,weight='normal',ha='center',va='center'):
    return ax.text(x,y,text,fontsize=size,color=color,fontweight=weight,ha=ha,va=va,linespacing=1.4)

def box(ax,x,y,w,h,text,color=PURPLE,fill=PALE_PURPLE,size=13,weight='normal'):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.015,rounding_size=0.14',
                              linewidth=1.5,edgecolor=color,facecolor=fill))
    label(ax,x+w/2,y+h/2,text,size=size,color=color,weight=weight)

def arrow(ax,a,b,color=PURPLE,style='-|>',rad=0):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle=style,mutation_scale=15,linewidth=1.8,
                                color=color,connectionstyle=f'arc3,rad={rad}',shrinkA=2,shrinkB=2))

def save(fig,name):
    fig.savefig(HERE/(name+'.png'),dpi=145,facecolor='white')
    fig.savefig(HERE/(name+'.svg'),facecolor='white')
    plt.close(fig)

fig,ax=setup(14.4,6.6)
label(ax,.4,6.2,'Rough Bergomi: many latent paths produce one option surface',size=20,weight='bold',ha='left')
label(ax,.4,5.73,'The forward-variance curve defines the simulator; its role in inference depends on the task.',size=13,ha='left',color=GRAY)
box(ax,.4,4.4,6.6,.95,'Known curve: infer '+r'$H,\eta,\rho$'+'\nSupply seven '+r'$\xi_0(t_j)$'+' levels together with the observed quotes',size=13)
box(ax,7.4,4.4,6.6,.95,'Joint curve: infer '+r'$H,\eta,\rho,S,L,k$'+'\n'+r'$\xi_0(t)=L+(S-L)e^{-kt}$'+'; generating curve is not an input',color=GREEN,fill=PALE_GREEN,size=13)
xs=[.4,3.2,6.0,8.8,11.6]
texts=[
 'Prior draw\nparameters + curve',
 'Many latent asset\nand variance paths',
 'Average discounted\ncontract payoffs\n'+r'$\widehat{\mathbf{P}}_{\rm MC}$'+': 63 numerical prices',
 'Add Gaussian errors\n'+r'$y_c=\widehat P_{{\rm MC},c}+\sigma_c\epsilon_c$',
 'Posterior network\n'+r'$\mathbf{y}$'+' (+curve only if known)\n'+r'$\rightarrow$'+' parameter draws',
]
for i,(x,text) in enumerate(zip(xs,texts)):
 box(ax,x,2.45,2.4,1.25,text,color=GREEN if i>=3 else PURPLE,
     fill=PALE_GREEN if i>=3 else PALE_PURPLE,size=12 if i in (2,3,4) else 13)
 if i:arrow(ax,(xs[i-1]+2.44,3.075),(x-.04,3.075),color=GREEN if i>=4 else PURPLE)
label(ax,7.2,1.82,'Finite-path Monte Carlo error belongs to the pricing calculation.',size=12.5,color=PURPLE)
label(ax,7.2,1.43,r'It is distinct from the prescribed observation noise: $\epsilon_c\overset{\rm iid}{\sim}\mathcal{N}(0,1)$, fixed $\sigma_c$.',size=12.5,color=GREEN)
label(ax,.4,.82,'Offline training uses the generating parameters as targets. At a new observation, neither those labels nor latent paths are supplied.',size=11.5,ha='left',color=GRAY)
label(ax,.4,.39,'Schematic only • Assess numerical price error separately; a trained network does not by itself establish posterior accuracy.',size=10.8,ha='left',color=GRAY)
save(fig,'rough-data-and-inference')

fig,ax=setup(14.4,5.9)
label(ax,.4,5.5,'Trajectory inference: one history, many posterior snapshots',size=20,weight='bold',ha='left')
label(ax,.4,5.02,'Observe ordered daily log returns from a physical-measure process. Its variance remains hidden.',size=13,ha='left',color=GRAY)
xs=[.35,2.70,5.05,7.40,9.75,12.10]
texts=[
 'Prior draw\n'+r'$\mathbf{\Theta}$'+' includes drift '+r'$\mu$',
 'One stochastic\nasset/variance path\nunder '+r'$\mathbb{P}$',
 'Daily log returns\n'+r'$r_t=\log(S_t/S_{t-1})$',
 'Reveal '+r'$r_{1:T}$'+'\n+ time design\n+ padding mask',
 'GRU summary\nordered observations',
 'Conditional flow\n'+r'$q_\phi(\mathbf{\Theta}\mid r_{1:T},T)$'+'\nparameter draws',
]
for i,(x,text) in enumerate(zip(xs,texts)):
 box(ax,x,3.0,1.95,1.35,text,color=GREEN if i>=3 else PURPLE,
     fill=PALE_GREEN if i>=3 else PALE_PURPLE,size=11.5 if i in (0,2,5) else 12)
 if i:arrow(ax,(xs[i-1]+1.99,3.675),(x-.04,3.675),color=GREEN if i>=4 else PURPLE)
label(ax,3.675,2.48,'Process randomness generates the returns; no extra quote noise is added.',size=11.5,color=PURPLE)
label(ax,10.725,2.48,'The mask excludes unrevealed days and padding.',size=11.5,color=GREEN)
box(ax,.4,1.25,6.6,.7,'Offline: random prefixes and known parameter targets\ntrain the GRU and conditional flow jointly.',size=12)
box(ax,7.4,1.25,6.6,.7,'New history: use the fixed estimator at any revealed prefix.\nGenerating parameters and latent states are not inputs.',color=GREEN,fill=PALE_GREEN,size=12)
label(ax,.4,.69,'Each snapshot uses the original prior and the entire revealed history. Do not multiply posteriors from overlapping prefixes.',size=11.7,ha='left',color=INK)
label(ax,.4,.26,'Schematic only • Reusing an estimator across prefixes is not a guarantee of calibration, filtering or forecasting accuracy.',size=10.8,ha='left',color=GRAY)
save(fig,'trajectory-prefix-workflow')


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
provenance={
 'format':'teaching_illustration_provenance_v1','scientific_acceptance':False,
 'scope':'Original conceptual schematics of rough-option simulation/inference and physical-path prefix inference; no empirical results or model-accuracy claims.',
 'source':'lab/evidence/illustrations/application-workflows-20261101/render.py','source_sha256':sha(Path(__file__)),
 'visual_inspiration':{
  'repository':'talk-slides-sbi4finance-overleaf','commit':'75dce200d423ecfea988571809cee8b46450b581',
  'author_credit':'Stefan T. Radev, Introduction to Simulation-Based Inference',
  'sources':[
   {'path':'bayesops.sty','sha256':'f49358f40b39da05ed4999270c128cba48b2572efe99c6cbb75a3fb7012db6c0'},
   {'path':'macros/colors.tex','sha256':'72efcd91163a5d230e8e3f38d2b9426c204c832209683ab7420fc764c894ce5c'},
   {'path':'figures/amortized-3.pdf','sha256':'7a8b3eccb0abcc9469e5cb9cd46e085826ad97e71f7f7b72275e9d74e0653342'},
   {'path':'figures/basic-workflow.pdf','sha256':'12aa4e20e80e072eb7e4500ae3bb991fc53c950177690c3649f2f4f85cd19f1d'}],
  'use':'Original arrangement and labels, using the Heston lesson palette and training/inference separation. No source artwork copied.'},
 'style_source':{'path':'lab/evidence/illustrations/heston-workflow-20261101/render.py',
                 'sha256':sha(REPO/'lab/evidence/illustrations/heston-workflow-20261101/render.py')},
 'palette':{'text':INK,'training_purple':PURPLE,'inference_green':GREEN,'theme_accent':BLUE},
 'notation':{
  'rough':'Known curve targets H/eta/rho with seven supplied xi_0 levels; joint targets also S/L/k and excludes its generating curve from conditions.',
  'rough_price_error':'Finite-path MC error is numerical pricing error, distinct from prescribed Gaussian observation noise.',
  'trajectory':'Theta is the full static parameter vector, including physical drift mu. r_1:T is the complete revealed daily return prefix. Variance and generating labels are excluded from new-case conditions.',
  'prefix_posterior':'Each prefix posterior uses the original prior and the whole revealed history; overlapping-prefix posteriors are not multiplied.'},
 'outputs':{p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(HERE.glob('*')) if p.suffix in ('.png','.svg')},
 'publication':'Publish PNG and provenance JSON only through lab/studies/accept.py --run; editable SVG and rendering source remain in lab evidence.'}
(HERE/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
