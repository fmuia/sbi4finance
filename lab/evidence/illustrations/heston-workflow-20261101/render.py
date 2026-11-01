"""Render the Heston lesson's schematic diagrams; no numerical experiment data."""
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

fig,ax=setup(14.4,4.6)
label(ax,.45,4.2,'One simulated Heston calibration example',size=21,weight='bold',ha='left')
label(ax,.45,3.7,'A clean option price is an expected payoff. Quote error is a separate observation assumption.',size=13,ha='left',color=GRAY)
box(ax,.45,1.85,2.3,1.15,'Draw parameters\n'+r'$\vartheta\sim p(\vartheta)$',size=15)
box(ax,3.35,1.85,3.0,1.15,'Heston Fourier pricer\n'+r'$P_c(\vartheta)=\mathbb{E}^{\mathbb{Q}}_\vartheta[\mathrm{payoff}_c]$',size=14)
box(ax,6.95,1.85,2.4,1.15,'63 clean prices\n'+r'$\mathbf{P}(\vartheta)$',size=15)
box(ax,10.0,1.85,3.95,1.15,'63 observed quotes\n'+r'$y_c=P_c(\vartheta)+\sigma_c\epsilon_c$',color=GREEN,fill=PALE_GREEN,size=15)
for a,b in [((2.8,2.425),(3.3,2.425)),((6.4,2.425),(6.9,2.425)),((9.4,2.425),(9.95,2.425))]:arrow(ax,a,b)
label(ax,11.97,1.15,r'$\epsilon_c\overset{\mathrm{iid}}{\sim}\mathcal{N}(0,1)$; fixed quote SDs $\sigma_c$',size=13,color=GREEN)
arrow(ax,(11.97,1.48),(11.97,1.8),color=GREEN)
label(ax,.45,.65,'Store the training pair '+r'$(\vartheta,\mathbf{y})$'+': one hypothetical option market, not one asset path.',size=14,ha='left')
label(ax,.45,.23,'Schematic only • Heston path randomness is integrated into the expected prices.',size=10.5,ha='left',color=GRAY)
save(fig,'simulation-and-observation')

fig,ax=setup(14.4,8.0)
label(ax,.4,7.6,'What is learned once — and what happens for each observation?',size=20,weight='bold',ha='left')
# Two deliberately parallel panels, with separate training and inference phases.
for x,w,color,fill in [(.4,6.6,PURPLE,PALE_PURPLE),(7.4,6.6,GREEN,PALE_GREEN)]:
    ax.add_patch(FancyBboxPatch((x,.85),w,6.25,boxstyle='round,pad=0.015,rounding_size=.16',linewidth=0,facecolor=fill))
label(ax,3.7,6.75,'Learn a forward pricer',size=19,color=PURPLE,weight='bold')
label(ax,10.7,6.75,'Learn a posterior distribution',size=19,color=GREEN,weight='bold')
label(ax,.7,6.15,'SIMULATION + TRAINING',size=10.5,ha='left',color=PURPLE,weight='bold')
label(ax,7.7,6.15,'SIMULATION + TRAINING',size=10.5,ha='left',color=GREEN,weight='bold')
box(ax,.75,5.18,2.65,.7,r'Pairs $(\vartheta,\mathbf{P}(\vartheta))$',size=14,fill='white')
box(ax,4.0,5.18,2.65,.7,r'Fixed pricer $\widehat{\mathbf{P}}_\phi(\vartheta)$',size=14,fill='white')
arrow(ax,(3.45,5.53),(3.95,5.53))
box(ax,7.75,5.18,2.65,.7,r'Pairs $(\vartheta,\mathbf{y})$',color=GREEN,fill='white',size=14)
box(ax,11.,5.18,2.65,.7,r'Fixed $q_\phi(\vartheta\mid\mathbf{y})$',color=GREEN,fill='white',size=14)
arrow(ax,(10.45,5.53),(10.95,5.53),color=GREEN)
for x in [.7,7.7]: ax.plot([x,x+6.0],[4.85,4.85],color='#D0CDDD',lw=1)
label(ax,.7,4.5,'INFERENCE FOR ONE NEW QUOTE VECTOR',size=10.5,ha='left',color=PURPLE,weight='bold')
label(ax,7.7,4.5,'INFERENCE FOR ONE NEW QUOTE VECTOR',size=10.5,ha='left',color=GREEN,weight='bold')
box(ax,1.15,3.42,5.1,.75,r'Observed $\mathbf{y}_{\rm obs}$ + chosen prior + quote likelihood',size=12.5,fill='white')
box(ax,1.15,2.22,5.1,.75,'MCMC using the learned prices in the likelihood',size=12.5,fill='white')
box(ax,1.8,1.15,3.8,.65,'Approximate posterior draws',size=13,fill='white')
arrow(ax,(3.7,3.37),(3.7,3.02));arrow(ax,(3.7,2.17),(3.7,1.85))
box(ax,8.15,3.42,5.1,.75,r'Observed $\mathbf{y}_{\rm obs}$ enters the fixed estimator',color=GREEN,fill='white',size=12.5)
box(ax,8.15,2.22,5.1,.75,r'Sample from $q_\phi(\vartheta\mid\mathbf{y}_{\rm obs})$',color=GREEN,fill='white',size=14)
box(ax,8.8,1.15,3.8,.65,'Approximate posterior draws',color=GREEN,fill='white',size=13)
arrow(ax,(10.7,3.37),(10.7,3.02),color=GREEN);arrow(ax,(10.7,2.17),(10.7,1.85),color=GREEN)
label(ax,.45,.52,'Pricer route: prior and quote likelihood are chosen downstream. Posterior route: those assumptions enter training.',size=11.5,ha='left',color=GRAY)
label(ax,.45,.19,'Both need validation. Neural sampling may use repeated evaluations; schematic arrows do not imply one forward pass.',size=10.7,ha='left',color=GRAY)
save(fig,'two-amortization-routes')

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
provenance={
 'format':'teaching_illustration_provenance_v1', 'scientific_acceptance':False,
 'scope':'Original conceptual schematics; no empirical results or new model-accuracy claims.',
 'source':'lab/evidence/illustrations/heston-workflow-20261101/render.py', 'source_sha256':sha(Path(__file__)),
 'visual_inspiration':{'repository':'talk-slides-sbi4finance-overleaf', 'commit':'75dce200d423ecfea988571809cee8b46450b581',
   'author_credit':'Stefan T. Radev, Introduction to Simulation-Based Inference',
   'sources':[{'path': 'bayesops.sty', 'sha256': 'f49358f40b39da05ed4999270c128cba48b2572efe99c6cbb75a3fb7012db6c0'}, {'path': 'macros/colors.tex', 'sha256': '72efcd91163a5d230e8e3f38d2b9426c204c832209683ab7420fc764c894ce5c'}, {'path': 'figures/amortized-3.pdf', 'sha256': '7a8b3eccb0abcc9469e5cb9cd46e085826ad97e71f7f7b72275e9d74e0653342'}, {'path': 'figures/basic-workflow.pdf', 'sha256': '12aa4e20e80e072eb7e4500ae3bb991fc53c950177690c3649f2f4f85cd19f1d'}],
   'use':'Original arrangement and labels, inspired by the deck palette and separation of training/inference. No source artwork copied.'},
 'palette':{'text':INK,'training_purple':PURPLE,'inference_green':GREEN,'theme_accent':BLUE},
 'notation':{'vartheta':'four unknown Heston parameters (v0, long variance, vol-of-vol, correlation); kappa fixed',
             'P':'vector of clean expected option payoffs', 'y':'vector of 63 prices with independent prescribed Gaussian quote errors'},
 'outputs':{p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(HERE.glob('*')) if p.suffix in ('.png','.svg')},
 'publication':'Publish PNG and provenance JSON only through lab/studies/accept.py --run; editable SVG and rendering source remain in lab evidence.'}
(HERE/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
