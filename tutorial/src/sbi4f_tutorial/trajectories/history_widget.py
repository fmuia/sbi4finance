"""Reveal a verified synthetic return/latent-variance pair, with no inference."""

import json
from collections.abc import Mapping

import numpy as np


def build_payload(history, *, end, label="Selected synthetic history"):
    """Keep original returns and paired latent boundary values in separate fields."""
    if isinstance(end, bool) or not isinstance(end, (int, np.integer)) or end < 1:
        raise ValueError("history display needs a positive integer day limit")
    arrays = {}
    for name, size in (("returns", end), ("variance", end + 1)):
        raw = history[name]
        if np.iscomplexobj(raw) or np.ma.isMaskedArray(raw):
            raise ValueError("history display needs real unmasked synthetic values")
        values = np.asarray(raw)
        if values.ndim != 1 or len(values) < size or values.dtype.kind not in "fiu" or not np.isfinite(values[:size]).all():
            raise ValueError("history display needs finite returns and their paired boundary variances")
        arrays[name] = np.asarray(values[:size], dtype=float)
    if np.any(arrays["variance"] < 0):
        raise ValueError("effective variance must be nonnegative; signed numerical states are a different quantity")
    references = {}
    for name in ("mean_variance", "median_variance"):
        raw = history.get(name)
        if raw is None:
            references[name] = None
            continue
        values = np.asarray(raw)
        if (np.iscomplexobj(raw) or np.ma.isMaskedArray(raw) or values.ndim != 1
                or len(values) < end + 1 or not np.isfinite(values[:end+1]).all()
                or np.any(values[:end+1] < 0)):
            raise ValueError("variance references must be finite nonnegative paired boundary values")
        references[name] = values[:end+1].astype(float).tolist()
    return {"format": "synthetic_history_player_v1", "end": int(end), "label": str(label),
            "returns": (100 * arrays["returns"]).tolist(), "variance": arrays["variance"].tolist(),
            **references,
            "scope": "paired synthetic truth only; variance is never supplied to the posterior network"}


def render_html(history, *, end, label="Selected synthetic history", show_header=True,
                show_footer=True, show_references=True, labels=None):
    """Display the same data with notebook-owned titles and optional annotations.

    ``labels`` accepts strings for returns, variance, returnhist, variancehist
    and description. Log views append their mathematical transform to the
    supplied variance titles. Omitting a header/footer removes its DOM nodes.
    """
    options = {"show_header": show_header, "show_footer": show_footer,
               "show_references": show_references}
    if any(not isinstance(value, (bool, np.bool_)) for value in options.values()):
        raise ValueError("history presentation flags must be booleans")
    titles = {"returns": "Observed daily log return (%)",
              "variance": "Actual hidden variance V (annualized variance)",
              "returnhist": "Revealed returns (%)", "variancehist": "Revealed boundary V",
              "description": "Play reveals the same stored history. The orange trace is its actual synthetic variance, "
                             "available here because we know how the example was generated."}
    if labels is not None:
        if (not isinstance(labels, Mapping) or set(labels) - set(titles)
                or any(not isinstance(value, str) for value in labels.values())):
            raise ValueError("history labels must be strings for the declared panel titles or description")
        titles.update(labels)
    payload = build_payload(history, end=end, label=label)
    payload["presentation"] = {**{key: bool(value) for key, value in options.items()}, "labels": titles}
    encoded = json.dumps(payload, allow_nan=False).replace("</", "<\\/")
    document = _DOCUMENT
    for name in ("header", "footer"):
        before, region = document.split("<!--" + name + "-->", 1)
        contents, after = region.split("<!--/" + name + "-->", 1)
        document = before + (contents if options["show_" + name] else "") + after
    return document.replace("__PAYLOAD__", encoded)


_DOCUMENT = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>
*{box-sizing:border-box}body{margin:0;padding:10px;color:#17233a;font:14px/1.45 system-ui}.box{border:1px solid #d4dfef;border-radius:12px;padding:14px;background:#fbfcff}h3{font-size:17px;margin:0 0 5px}.small{font-size:12px;color:#526177}.controls{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin:12px 0}button,select{font:inherit;border:1px solid #bcc9db;border-radius:6px;background:white;color:#17233a;padding:5px 9px}button{cursor:pointer}#play{background:#2563eb;color:white;border-color:#2563eb}label{font-size:12px}input{width:100%;accent-color:#2563eb}.scrub{display:grid;grid-template-columns:1fr auto;gap:10px;align-items:center}#position{font-size:13px;font-weight:650}.charts{display:grid;grid-template-columns:minmax(0,2.4fr) minmax(0,1fr);gap:8px;margin-top:10px}.charts svg{display:block;width:100%;height:auto;background:#fff;border:1px solid #e2e8f2;border-radius:8px}.note{font-size:12px;padding:9px;background:#fff6e9;border-left:3px solid #d98725;margin:10px 0 0}.counts{font-size:11px;color:#526177;margin-top:7px}@media(max-width:490px){.charts{grid-template-columns:1fr}.controls{gap:5px}body{padding:3px}.box{padding:9px}}
</style></head><body data-widget="synthetic-history"><section class="box" aria-label="Observed synthetic history player">
<!--header--><h3 id="title"></h3><p id="description" class="small"></p><!--/header-->
<div class="controls"><button id="play">Replay</button><button id="advance">Advance</button><button id="reset">Reset</button>
<label>Reveal step <select id="step" aria-label="History reveal step"><option value="1">1 day</option><option value="5">5 trading days</option><option value="21" selected>21 trading days</option></select></label>
<label>Axis units <select id="units" aria-label="History axis units"><option value="1">Trading days</option><option value="5">Trading weeks</option><option value="21" selected>Trading months</option></select></label>
<label>Variance view <select id="variance-view" aria-label="Variance view"><option value="linear">Variance V</option><option value="log">Log₁₀ variance</option></select></label>
<label>Fixed axes <select id="scale" aria-label="History fixed axes"><option value="standard">Standard</option><option value="wide">Wide</option><option value="verywide">Very wide</option></select></label></div>
<div class="scrub"><input id="scrub" type="range" min="0" value="0" aria-label="Revealed history day"><output id="position" aria-live="polite"></output></div>
<div class="charts"><svg id="returns" viewBox="0 0 680 210" role="img" aria-label="Revealed daily log returns"></svg><svg id="returnhist" viewBox="0 0 280 210" role="img" aria-label="Marginal distribution of revealed daily log returns"></svg>
<svg id="variance" viewBox="0 0 680 210" role="img" aria-label="Actual hidden variance at revealed day boundaries"></svg><svg id="variancehist" viewBox="0 0 280 210" role="img" aria-label="Marginal distribution of revealed synthetic boundary variance"></svg></div>
<!--footer--><div id="variance-legend" class="small"></div><div id="variance-readout" class="note"></div>
<div id="counts" class="counts"></div><p class="note"><b>Simulation truth, not a filtered state.</b> Variance is shown only to explain this synthetic history. The posterior network receives returns, their mask and known times; it never receives this variance path or its histogram.</p>
<p class="small">Right-hand panels are empirical distributions across the revealed time points, not posterior distributions of a hidden state. The variance view shows V in annualized variance units, or its explicitly selected log₁₀ transform; neither is volatility √V. These marginals do not replace the ordered return sequence used for inference. This local animation does not change the prefix selected for posterior calculations elsewhere.</p><!--/footer-->
</section><script>
const data=__PAYLOAD__,ns='http://www.w3.org/2000/svg';let day=data.end,timer=null;const el=id=>document.getElementById(id),scales={standard:{returns:[-8,8],variance:[0,.25],logvariance:[-12,0]},wide:{returns:[-20,20],variance:[0,1],logvariance:[-20,2]},verywide:{returns:[-50,50],variance:[0,4],logvariance:[-40,4]}};
function add(svg,tag,a={},text=null){const n=document.createElementNS(ns,tag);for(const[k,v]of Object.entries(a))n.setAttribute(k,String(v));if(text!==null)n.textContent=text;svg.append(n);return n;}function fmt(x){return Number(x).toPrecision(3).replace(/\.0+(?=e|$)/,'');}
function setText(id,text){const target=el(id);if(target)target.textContent=text;}
function stop(){clearInterval(timer);timer=null;el('play').textContent=day===data.end?'Replay':'Play';}function advance(){day=Math.min(data.end,day+Number(el('step').value));draw();if(day===data.end)stop();}function play(){if(timer){stop();return;}if(day===data.end)day=0;el('play').textContent='Pause';timer=setInterval(advance,200);draw();}
function trace(id,values,start,title,color,limits,references=[]){const svg=el(id);svg.replaceChildren();const W=680,H=210,L=54,R=16,T=28,B=36,[lo,hi]=limits,x=t=>L+t/data.end*(W-L-R),y=v=>T+(hi-v)/(hi-lo)*(H-T-B);add(svg,'rect',{x:x(day),y:T,width:Math.max(0,x(data.end)-x(day)),height:H-T-B,fill:'#f2f5f9'});
 for(let i=0;i<5;i++){const v=lo+(hi-lo)*i/4;add(svg,'line',{x1:L,x2:W-R,y1:y(v),y2:y(v),stroke:'#e4eaf3'});add(svg,'text',{x:L-6,y:y(v)+3,'text-anchor':'end','font-size':10,fill:'#526177'},fmt(v));}
 const unit=Number(el('units').value),ticks=[...new Set(Array.from({length:5},(_,i)=>unit===1?Math.round(data.end*i/4):data.end*i/4))];for(const t of ticks)add(svg,'text',{x:x(t),y:H-B+17,'text-anchor':'middle','font-size':10,fill:'#526177'},unit===1?String(t):fmt(t/unit));
 function line(rows,stroke,dash=null,markers=false){let points=[],outside=0;function flush(){if(points.length>1)add(svg,'polyline',{points:points.join(' '),fill:'none',stroke,'stroke-width':markers?1.4:1.2,...(dash?{'stroke-dasharray':dash}:{})});else if(points.length){const [cx,cy]=points[0].split(',');add(svg,'circle',{cx,cy,r:2,fill:stroke});}points=[];}
 for(let i=0;i<rows.length;i++){const v=rows[i],d=i+start;if(v===null){flush();continue;}if(v<lo||v>hi){outside++;if(markers){const mark=add(svg,'circle',{cx:x(d),cy:y(Math.max(lo,Math.min(hi,v))),r:2.4,fill:'#b45309'});add(mark,'title',{},'Day '+d+': '+fmt(v));}}points.push(x(d)+','+y(Math.max(lo,Math.min(hi,v))));}flush();return outside;}
 for(const ref of references)line(ref.values,ref.color,'5 3');const outside=line(values,color,null,true);add(svg,'line',{x1:x(day),x2:x(day),y1:T,y2:H-B,stroke:'#17233a','stroke-width':1});
 add(svg,'text',{x:L,y:17,'font-size':12,fill:'#17233a'},title);add(svg,'text',{x:(W+L-R)/2,y:H-3,'text-anchor':'middle','font-size':10,fill:'#526177'},unit===1?'Trading day':unit===5?'Trading week (5 days)':'Trading month (21 days)');return outside;}
function histogram(id,values,title,color,limits){const svg=el(id);svg.replaceChildren();const W=280,H=210,L=35,R=12,T=28,B=36,[lo,hi]=limits,bins=18,counts=Array(bins).fill(0);for(const v of values)if(v!==null&&v>=lo&&v<=hi)counts[Math.min(bins-1,Math.floor((v-lo)/(hi-lo)*bins))]++;
 const x=v=>L+(v-lo)/(hi-lo)*(W-L-R),y=n=>H-B-n*(H-T-B);for(let i=0;i<bins;i++)add(svg,'rect',{x:x(lo+i*(hi-lo)/bins)+.5,y:y(counts[i]/Math.max(1,values.length)),width:(W-L-R)/bins-1,height:H-B-y(counts[i]/Math.max(1,values.length)),fill:color,opacity:.65});add(svg,'text',{x:L,y:17,'font-size':11,fill:'#17233a'},title);add(svg,'text',{x:L,y:T+11,'font-size':9,fill:'#526177'},'Fraction / bin · '+values.length+' values');for(const v of[lo,(lo+hi)/2,hi])add(svg,'text',{x:x(v),y:H-B+17,'text-anchor':'middle','font-size':9,fill:'#526177'},fmt(v));add(svg,'line',{x1:L,x2:W-R,y1:H-B,y2:H-B,stroke:'#b7c4d5'});if(!values.length)add(svg,'text',{x:W/2,y:H/2,'text-anchor':'middle','font-size':11,fill:'#7a879a'},'No returns revealed');}
function median(rows){const a=[...rows].sort((a,b)=>a-b),m=Math.floor(a.length/2);return a.length%2?a[m]:(a[m-1]+a[m])/2;}
function draw(){if(!timer)el('play').textContent=day===data.end?'Replay':'Play';document.body.dataset.revealedDay=day;el('scrub').value=day;el('position').textContent=day+' / '+data.end+' trading days';const r=data.returns.slice(0,day),v=data.variance.slice(0,day+1),scale=scales[el('scale').value],log=el('variance-view').value==='log',convert=a=>log?a.map(x=>x>0?Math.log10(x):null):a,shown=convert(v),limits=log?scale.logvariance:scale.variance;
 const references=[];if(data.presentation.show_references)for(const[key,color]of[['mean_variance','#0f8b78'],['median_variance','#8b5cf6']])if(data[key])references.push({values:convert(data[key].slice(0,day+1)),color});
 const titles=data.presentation.labels,logLabel=log?' · log₁₀ V':'';
 const ro=trace('returns',r,1,titles.returns,'#2563eb',scale.returns),vo=trace('variance',shown,0,titles.variance+logLabel,'#d58024',limits,references);histogram('returnhist',r,titles.returnhist,'#2563eb',scale.returns);histogram('variancehist',shown,titles.variancehist+logLabel,'#d58024',limits);
 const zero=v.filter(x=>x===0).length;let text='Actual V at day '+day+': '+fmt(v.at(-1))+'. Minimum revealed V: '+fmt(Math.min(...v))+'. Exact zeros: '+zero+'/'+v.length+'.';if(day>=2){const w=Math.min(252,Math.floor(day/2)),first=median(v.slice(1,w+1)),last=median(v.slice(-w));text+=' In this path, median V over the first '+w+' days = '+fmt(first)+'; last '+w+' days = '+fmt(last)+'.'+(first>0?' Last/first ratio = '+fmt(last/first)+'.':'');}setText('variance-readout',text);
 setText('variance-legend',references.length?'Orange: actual path · dashed teal: mean ξ(t) across possible realizations · dashed purple: median at each time across possible realizations. These references are not time averages or state estimates.':'');
 setText('counts','Outside the fixed views: '+ro+'/'+r.length+' returns and '+vo+'/'+v.length+' variance values.'+(log?' Log view leaves '+zero+' exact zero values undefined; none is replaced by a positive floor. A log value of −6 means V = 10⁻⁶.':'')+' All values remain unchanged. Histogram denominators include all revealed values, including those outside the view.');el('advance').disabled=day===data.end;}
setText('title',data.label);setText('description',data.presentation.labels.description);el('scrub').max=data.end;el('play').onclick=play;el('advance').onclick=()=>{stop();advance();};el('reset').onclick=()=>{stop();day=0;draw();};el('scrub').oninput=()=>{stop();day=Number(el('scrub').value);draw();};el('units').onchange=draw;el('scale').onchange=draw;el('variance-view').onchange=draw;window.addEventListener('pagehide',stop);draw();
let resizePending=false;new ResizeObserver(()=>{if(resizePending)return;resizePending=true;requestAnimationFrame(()=>{resizePending=false;try{const f=window.frameElement,h=Math.ceil(document.body.scrollHeight);if(f&&Math.abs(f.getBoundingClientRect().height-h)>1)f.style.height=h+'px';}catch{}});}).observe(document.body);
</script></body></html>"""
