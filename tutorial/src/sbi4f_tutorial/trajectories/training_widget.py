"""Illustrate the actual return-token/mask route without inventing activations."""

import json

import numpy as np

from .inference import conditioning_data, training_data
from .simulation import validate_task


def build_payload(task, observations, *, parameters=None):
    """Actual prefix inputs and one optional, separate joint training label.

    No encoder, density estimator, optimizer or sampler is evaluated here.
    The physical label and its exact float32 training transform stay fixed
    when the observed prefix changes.
    """
    validate_task(task)
    values = np.asarray(observations)
    if values.ndim == 1:
        values = values[:, None]
    if (values.shape != (task["n_observations"], 1) or values.dtype.kind not in "fiu"
            or not np.isfinite(values).all() or np.ma.isMaskedArray(observations)):
        raise ValueError("training animation needs one complete finite return history")
    lengths = sorted({min(day, len(values)) for day in (1, 21, 63, 126, 252)})
    examples = []
    for length in lengths:
        inputs = conditioning_data(values[None, :length], task)
        tokens = inputs["observations"][0]
        examples.append({"length": length, "tokens": tokens.tolist(),
                         "mask": inputs["summary_mask"][0].tolist(), "time_context": inputs["time_context"][0].tolist(),
                         "first": {"day": 1, "raw_return": float(values[0, 0]), "token": tokens[0].tolist(), "mask": True},
                         "last": {"day": length, "raw_return": float(values[length-1, 0]), "token": tokens[length-1].tolist(), "mask": True},
                         "first_padding": None if length == len(values) else
                             {"day": length+1, "raw_return": None, "token": tokens[length].tolist(), "mask": False}})
    target = None
    if parameters is not None:
        if np.iscomplexobj(parameters) or np.ma.isMaskedArray(parameters):
            raise ValueError("provide one real unmasked generating parameter vector")
        physical = np.asarray(parameters)
        if physical.shape != (len(task["parameter_names"]),) or physical.dtype.kind not in "fiu":
            raise ValueError("the generating label must be one complete joint parameter vector")
        transformed = training_data({"observations": values[None], "parameters": physical[None]}, task)["parameters"][0]
        target = {"physical": physical.astype(float).tolist(), "transformed": transformed.tolist(),
                  "shape": [1, len(physical)], "route": "training_label_only"}
    largest = max(float(np.max(np.abs(100 * values[:, 0]))) * 1.12, .01)
    step = 10 ** np.floor(np.log10(largest)) / 2
    return {"maximum": len(values), "examples": examples, "summary_dim": 16,
            "targets": list(task["parameter_names"]), "target": target,
            "raw_returns_percent": (100 * values[:, 0]).tolist(),
            "return_limit_percent": float(np.ceil(largest / step) * step),
            "return_reference_scale": float(np.sqrt(.04 * task["dt_years"])),
            "default_prefix": min(63, len(values))}


def render_html(task, observations, *, parameters=None):
    encoded = json.dumps(build_payload(task, observations, parameters=parameters), allow_nan=False).replace("</", "<\\/")
    return _DOCUMENT.replace("__PAYLOAD__", encoded)


def encoder_comparison_html():
    """A code-faithful comparison: a recurrent encoder plus other explicit changes."""
    return """<style>
    .encoder-compare{border:1px solid #d8e0ed;border-radius:10px;padding:14px;margin:10px 0;font:14px/1.4 system-ui;color:#17233a}
    .encoder-compare .route{display:grid;grid-template-columns:1.1fr 1fr 1.15fr;gap:12px;align-items:stretch;margin:9px 0 16px}
    .encoder-compare .node{border:1px solid #cbd5e1;border-radius:8px;padding:10px;background:#f8fafc;min-width:0}
    .encoder-compare .new{border-color:#93b6f5;background:#eff5ff}.encoder-compare .density{border-color:#e7bd86;background:#fff8ef}
    .encoder-compare small{display:block;color:#526177;margin-top:5px}.encoder-compare code{font-size:12px;overflow-wrap:anywhere}
    @media(max-width:580px){.encoder-compare .route{grid-template-columns:1fr}.encoder-compare .node{padding:8px}}
    </style><section class="encoder-compare" aria-label="Option and trajectory observation encoders">
    <strong>What changes between option prices and return histories?</strong>
    <div class="route"><div class="node"><b>Option inputs</b><small>A fixed grid of quotes; rough options also supply the curve when known.</small></div>
    <div class="node"><b>Direct condition vector →</b><small>The displayed option workflows route this vector through the adapter.</small><code>inference_conditions</code></div>
    <div class="node density"><b>FlowMatching → parameters</b><small>A conditional density estimator learns the joint parameter posterior.</small></div></div>
    <div class="route"><div class="node new"><b>Ordered return inputs</b><small>Return/time tokens + observed-prefix mask. Padding is excluded.</small><code>summary_variables + summary_mask</code></div>
    <div class="node new"><b>Recurrent encoder →</b><small>A causal GRU reads the sequence and emits a learned summary.</small><code>RecurrentNetwork</code></div>
    <div class="node density"><b>CouplingFlow → parameters</b><small>Conditions on the summary and known time design.</small></div></div>
    <small><b>The GRU is a recurrent observation encoder.</b> It is not an autoregressive posterior decoder or a forecast of the next return. The density family, units, target transforms, masking and training-prefix design also differ here. <code>BasicWorkflow</code> still organizes simulation, fitting and conditional sampling.</small>
    </section>"""


_DOCUMENT = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>
*{box-sizing:border-box}body{margin:0;padding:10px;font:14px/1.45 system-ui;color:#17233a}.lesson{border:1px solid #d6dfec;border-radius:12px;padding:15px;background:#fbfcff}h3{font-size:18px;margin:0 0 7px}.intro{margin:0 0 7px}.small{font-size:12px;color:#526177}.try{font-size:12px;border-left:3px solid #0f8b72;padding:6px 9px;background:#eef9f5;margin:9px 0}.controls{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin:12px 0}label{font-size:12px}button,select{font:inherit;padding:6px 10px;border:1px solid #b8c6d9;border-radius:6px;background:white;color:#17233a}button{cursor:pointer}button:disabled{opacity:.45;cursor:default}#play{background:#2563eb;color:white;border-color:#2563eb}.stages{display:flex;flex-wrap:wrap;gap:6px;padding:0;margin:10px 0;list-style:none;font-size:11px}.stages li{padding:4px 7px;border:1px solid #dbe3ef;border-radius:5px;color:#64748b;background:#fff}.stages li.current{border-color:#2563eb;background:#eaf2ff;color:#1649a4;font-weight:650}.facts{display:flex;gap:9px;flex-wrap:wrap;font-size:12px;font-variant-numeric:tabular-nums;margin-bottom:8px}.facts span{border-radius:5px;background:#edf2f9;padding:4px 7px}#trace{display:block;width:100%;height:auto;border:1px solid #e1e7f0;border-radius:8px;background:white}.route{display:grid;grid-template-columns:1fr 22px 1fr 22px 1fr;gap:7px;align-items:stretch;margin-top:12px}.node{border:1px solid #cbd5e1;border-radius:8px;background:#fff;padding:10px;min-width:0}.node strong{font-size:13px;display:block;margin-bottom:4px}.node p{font-size:12px;color:#526177;margin:0}.node.active{border-color:#3b82f6;background:#edf4ff}.arrow{align-self:center;text-align:center;color:#74849b;font-size:21px}.conditions{font-size:11px;color:#526177;text-align:right;margin:4px 2px 7px}.training-row{display:grid;grid-template-columns:1fr 34px 1fr;gap:8px;align-items:stretch}.training-row .node{background:#faf5ff;border-color:#c4b5fd}.training-row .node.active{background:#f1e8ff;border-color:#9066dd}.training-row strong{color:#6433a4}.training-row .arrow{color:#9066dd}.math{font-size:15px;color:#6433a4;margin:3px 0}.label-values{font-size:12px;font-variant-numeric:tabular-nums;overflow-wrap:anywhere}.query-result{border-color:#93cbbd;background:#f0faf6}.query-result strong{color:#08745f}.stage-card{border:1px solid #bfd2f3;border-radius:8px;padding:11px;background:#f0f5ff;margin-top:12px;min-height:109px}.stage-card h4{font-size:14px;margin:0 0 5px}.stage-card p{margin:3px 0;font-size:12px}.takeaway{color:#214c85}.schematic{font-size:11px;color:#64748b;margin:7px 0 0}.details{font-size:12px;margin-top:12px}.details summary{cursor:pointer;color:#456084}.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;margin-top:8px;font-size:11px;font-variant-numeric:tabular-nums}th,td{text-align:left;padding:6px;border-bottom:1px solid #e0e7f1;white-space:nowrap}th{background:#f0f4fa}code{font-size:11px;overflow-wrap:anywhere}.completion{border-left:3px solid #0f8b72;padding:8px 10px;background:#eef9f5;font-size:12px;margin-top:9px}.footer{font-size:11px;color:#526177;margin-bottom:0}[hidden]{display:none!important}@media(max-width:540px){body{padding:3px}.lesson{padding:10px}.route,.training-row{grid-template-columns:1fr}.arrow{transform:rotate(90deg)}.conditions{text-align:left}.stage-card{min-height:0}}
</style></head><body data-widget="training-route"><section class="lesson" aria-label="Guided prefix training and inference comparison">
<h3>What changes when the network sees 21 days instead of 252?</h3>
<p class="intro">Follow one of the live tutorial’s 128 training pairs. Keep its history and generating parameters fixed; reveal a prefix, mask its future, then trace the information reaching the posterior estimator.</p>
<div class="try"><b>Try this:</b> compare 21 and 252 days. Which inputs change? Then switch to <b>Inference</b>: which two training-only boxes disappear?</div>
<div class="controls"><label>Observed prefix <select id="prefix" aria-label="Training observed prefix"></select></label><label>Mode <select id="mode" aria-label="Animation mode"><option value="training">Training</option><option value="inference">Inference</option></select></label><button id="play">Play</button><button id="next">Next stage</button><button id="reset">Reset</button></div>
<ol id="stages" class="stages" aria-label="Named computation stages"></ol>
<div id="facts" class="facts"></div>
<svg id="trace" viewBox="0 0 900 190" role="img" aria-label="Signed observed log returns and blank masked future"></svg>
<div class="route" aria-label="Schematic conditional posterior estimator"><section id="inputs" class="node"><strong>Return tokens + mask</strong><p id="input-description"></p></section><div class="arrow">→</div><section id="encoder" class="node"><strong>Causal GRU encoder</strong><p>Reads observed tokens in order.<br>Emits a 16-value learned summary s<sub>ψ</sub>.</p></section><div class="arrow">→</div><section id="flow" class="node"><strong>Conditional coupling flow</strong><p>Receives summary s<sub>ψ</sub> and known time context c.<br>Models one joint parameter vector.</p></section></div>
<div id="condition-link" class="conditions">↓ Summary and known context condition the density score</div>
<div id="training-route" class="training-row"><section id="label-node" class="node"><strong>Fixed generating label Θ<sub>true</sub></strong><div id="label-values" class="label-values"></div><p>One joint label for every prefix; transformed to z<sub>true</sub> for scoring.</p></section><div class="arrow">→</div><section id="objective-node" class="node"><strong>Joint-density training objective</strong><div class="math">−log q<sub>φ</sub>(z<sub>true</sub> | s<sub>ψ</sub>, c)</div><p>Score the known transformed label. The batch loss averages these negative log densities; it is not squared parameter error.</p></section></div>
<section id="query-route" class="node query-result" hidden><strong>Sample the unknown parameters</strong><p>With fitted weights fixed, <code>workflow.sample</code> draws joint z vectors. <code>inverse_parameters</code> maps each complete draw back to physical Θ. This illustration generates no posterior samples.</p></section>
<div id="stage-card" class="stage-card" aria-live="polite"><h4 id="stage-title"></h4><p id="stage-fact"></p><p id="stage-takeaway" class="takeaway"></p></div>
<div id="completion" class="completion" hidden></div>
<p class="schematic">Network boxes and highlights are schematic, not measured neural activations. Input, mask, time-context and transformed-label numbers use the actual preprocessing code. No summary values, losses, optimizer updates or samples are invented.</p>
<details class="details"><summary>Inspect actual first/last tokens and the label transform</summary><p id="context-definition"></p><div class="scroll"><table><thead><tr><th>Position</th><th>Raw log return</th><th>Scaled return</th><th>Day / maximum</th><th>Mask</th></tr></thead><tbody id="token-rows"></tbody></table></div><div id="target-details"></div></details>
<p class="footer">The prefix choices illustrate the mechanism. Actual training samples lengths across all supported integers, using its stated random-prefix mixture. Latent variance and process shocks are never encoder inputs.</p>
</section><script>
const data=__PAYLOAD__,ns='http://www.w3.org/2000/svg',el=id=>document.getElementById(id);let stage=0,timer=null;
const trainStages=['One fixed pair','Reveal & mask','Encode the prefix','Condition the flow','Score the known label','Update both networks'];
const queryStages=['Observed history','Reveal & mask','Encode the prefix','Condition the flow','Sample parameters'];
const symbols={hurst:'H',eta:'η',rho:'ρ',drift:'μ',kappa:'κ',long_variance:'L',short_variance:'S',decay:'k',vol_of_vol:'σ'};
function fmt(v,n=4){return Number(v).toPrecision(n).replace(/\.0+(?=e|$)/,'');}
function mode(){return el('mode').value;}function names(){return mode()==='training'?trainStages:queryStages;}
function stop(){clearInterval(timer);timer=null;el('play').textContent=stage===names().length-1?'Replay':'Play';}
function next(){if(stage<names().length-1)stage++;if(stage===names().length-1)stop();draw();}
function play(){if(timer){stop();return;}if(stage===names().length-1)stage=0;el('play').textContent='Pause';timer=setInterval(next,1800);draw();}
function add(svg,tag,a={},text=null){const n=document.createElementNS(ns,tag);for(const[k,v]of Object.entries(a))n.setAttribute(k,String(v));if(text!==null)n.textContent=text;svg.append(n);return n;}
function drawTrace(example){const svg=el('trace');svg.replaceChildren();const W=900,H=190,L=57,R=20,T=23,B=53,limit=data.return_limit_percent,x=d=>L+d/data.maximum*(W-L-R),y=r=>T+(limit-r)/(2*limit)*(H-T-B),n=example.length;
add(svg,'rect',{x:x(n),y:T,width:x(data.maximum)-x(n),height:H-T-B,fill:'#eef1f6'});for(const r of[-limit,0,limit]){add(svg,'line',{x1:L,x2:W-R,y1:y(r),y2:y(r),stroke:r===0?'#9aabc2':'#e4eaf2'});add(svg,'text',{x:L-7,y:y(r)+3,'text-anchor':'end','font-size':10,fill:'#526177'},fmt(r,3));}
const points=data.raw_returns_percent.slice(0,n).map((r,i)=>x(i+1)+','+y(r));if(n===1)add(svg,'circle',{cx:x(1),cy:y(data.raw_returns_percent[0]),r:3,fill:'#2563eb'});else add(svg,'polyline',{points:points.join(' '),fill:'none',stroke:'#2563eb','stroke-width':1.35,'data-observed-points':n});
add(svg,'line',{x1:x(n),x2:x(n),y1:T,y2:H-B,stroke:'#566a87','stroke-dasharray':'3 3'});for(const d of [...new Set(Array.from({length:5},(_,i)=>Math.round(data.maximum*i/4)))])add(svg,'text',{x:x(d),y:H-B+15,'text-anchor':'middle','font-size':10,fill:'#526177'},d);
add(svg,'text',{x:L,y:14,'font-size':11,fill:'#17233a'},'Signed daily log return (%) · same fixed axis for every prefix');if(data.maximum-n>data.maximum*.18)add(svg,'text',{x:(x(n)+x(data.maximum))/2,y:T+18,'text-anchor':'middle','font-size':11,fill:'#77859a'},'Blank padding — no future returns displayed');
for(let j=0;j<data.maximum;j++)add(svg,'rect',{x:x(j),y:163,width:Math.max(.7,x(j+1)-x(j)-.15),height:7,fill:example.mask[j]?'#0d9c81':'#d6deea'});add(svg,'text',{x:L,y:183,'font-size':10,fill:'#526177'},'Mask: '+n+' true / '+(data.maximum-n)+' false · genuine observed zero returns still have a true mask');}
function describe(example){const n=example.length,padded=data.maximum-n,c=example.time_context.map(v=>fmt(v,3)).join(', '),joint=data.targets.length;
if(stage===0)return mode()==='training'?['The simulator supplied one '+joint+'-parameter vector and this fixed '+data.maximum+'-day history.','The generating label stays the same when the observed prefix changes; it is not a time-varying state path.']:['Supply returns to a previously trained estimator. The generating label is absent from this query.','Inference does not require the unknown true parameter values.'];
if(stage===1)return[n+' observed days produce '+n+' true mask entries and '+padded+' padded rows. Both token fields in every padded row are zero.','Changing the prefix changes evidence, mask and known context; it does not regenerate the history or change its generating label.'];
if(stage===2)return['The GRU reads '+n+' observed return/time tokens in order. Its learned output has 16 values. Padded positions are excluded by the mask.','The recurrent summary represents information about parameters, not a filtered estimate of hidden variance. No summary activations are evaluated here.'];
if(stage===3)return['The density conditions are the 16-value summary plus two known time features c = ['+c+']. The target has '+joint+' joint coordinates.','A summary dimension is not a parameter count. The known time features remain separate from the learned summary.'];
if(mode()==='inference')return['Draw complete z vectors from the conditional density, then inverse-transform each vector to physical parameters.','The label, training loss and weight updates are absent. The actual inference action elsewhere in the notebook computes samples; this animation does not.'];
if(stage===4)return['The same transformed joint label z_true is scored under qφ for the selected prefix. The objective is its negative log density; no numeric loss is computed here.','Training rewards density at the generating label. It is not a squared-error fit of a posterior mean. More observations need not lower every individual example’s loss.'];
return['This pair’s density score contributes to the batch loss, whose gradients update both the encoder and the coupling flow. This stage illustrates the update; it performs none.','Training repeats across many prior-drawn parameter vectors, histories and prefixes. The separate Train button generates 128 pairs and fits weights.'];}
function drawDetails(example){el('context-definition').textContent='Return scaling uses √(0.04 Δt) = '+fmt(data.return_reference_scale,6)+'. Context c = [observed fraction of this task’s maximum, step in daily reference units] = ['+example.time_context.map(v=>fmt(v,6)).join(', ')+'].';
const rows=[['First observed',example.first],['Last observed',example.last],...(example.first_padding?[['First padded',example.first_padding]]:[])];el('token-rows').replaceChildren();for(const[label,row]of rows){const tr=document.createElement('tr');for(const text of[label+' (day '+row.day+')',row.raw_return===null?'Not supplied':fmt(row.raw_return,7),fmt(row.token[0],7),fmt(row.token[1],7),String(row.mask)]){const td=document.createElement('td');td.textContent=text;tr.append(td);}el('token-rows').append(tr);}
el('target-details').replaceChildren();if(mode()==='training'){const p=document.createElement('p');p.textContent=data.target?'One fixed joint label, shape ['+data.target.shape.join(', ')+']: Θ_true = ['+data.target.physical.map(v=>fmt(v,7)).join(', ')+']; z_true = ['+data.target.transformed.map(v=>fmt(v,7)).join(', ')+'].':'No generating-label values were supplied to this illustration.';el('target-details').append(p);const def=document.createElement('p');def.textContent='Actual training coordinates: z = logit(F_prior(Θ)) / (π/√3), stored as float32. The label is transformed once and is never concatenated to return tokens or density conditions.';el('target-details').append(def);}else{const p=document.createElement('p');p.textContent='No generating label or training objective is part of the inference input.';el('target-details').append(p);}}
function draw(){const example=data.examples.find(e=>e.length===Number(el('prefix').value)),training=mode()==='training',list=names(),last=stage===list.length-1;document.body.dataset.phase=stage;document.body.dataset.mode=mode();document.body.dataset.observedLength=example.length;el('next').disabled=last;if(!timer)el('play').textContent=last?'Replay':'Play';
el('stages').replaceChildren();for(let i=0;i<list.length;i++){const li=document.createElement('li');li.textContent=(i+1)+'. '+list[i];li.className=i===stage?'current':'';if(i===stage)li.setAttribute('aria-current','step');el('stages').append(li);}
el('facts').replaceChildren();for(const text of[example.length+' observed days',(data.maximum-example.length)+' padded rows','Context c = ['+example.time_context.map(v=>fmt(v,3)).join(', ')+']']){const span=document.createElement('span');span.textContent=text;el('facts').append(span);}
el('input-description').textContent='['+data.maximum+', 2] padded token matrix; '+example.length+' observed rows. Mask shape ['+data.maximum+']. No parameter labels enter here.';
for(const[id,focus]of[['inputs',1],['encoder',2],['flow',3],['label-node',0],['objective-node',4]])el(id).classList.toggle('active',stage===focus||(training&&stage===5&&(id==='encoder'||id==='flow'||id==='objective-node')));
el('training-route').hidden=!training;el('condition-link').hidden=!training;el('query-route').hidden=training;
el('label-values').textContent=data.target?data.targets.map((name,i)=>(symbols[name]||name)+' = '+fmt(data.target.physical[i])).join(' · '):'Generating values were not supplied.';
const[fact,takeaway]=describe(example);el('stage-title').textContent=(stage+1)+'. '+list[stage];el('stage-fact').textContent=fact;el('stage-takeaway').textContent='Takeaway: '+takeaway;
el('completion').hidden=!last;el('completion').textContent=training?'What to learn: within one pair, the prefix changes inputs, not the generating label. Repeat across many different pairs to train both networks; then query without labels. The actual Train button below performs that fitting.':'What to learn: querying uses returns, mask and known times with fitted weights. It neither needs Θ_true nor returns a latent-state trajectory.';drawTrace(example);drawDetails(example);}
for(const example of data.examples){const o=document.createElement('option');o.value=example.length;o.textContent=example.length+' days';el('prefix').append(o);}el('prefix').value=data.default_prefix;
el('play').onclick=play;el('next').onclick=()=>{stop();next();};el('reset').onclick=()=>{stop();stage=0;draw();};for(const id of['prefix','mode'])el(id).onchange=()=>{stop();stage=0;draw();};window.addEventListener('pagehide',stop);
let resizePending=false;new ResizeObserver(()=>{if(resizePending)return;resizePending=true;requestAnimationFrame(()=>{resizePending=false;try{const f=window.frameElement,h=Math.ceil(document.body.scrollHeight);if(f&&Math.abs(f.getBoundingClientRect().height-h)>1)f.style.height=h+'px';}catch{}});}).observe(document.body);draw();
</script></body></html>"""
