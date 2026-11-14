"""Illustrate the actual return-token/mask route without inventing activations."""

import json

import numpy as np

from .inference import conditioning_data
from .simulation import validate_task


def build_payload(task, observations):
    """Use canonical preprocessing at each displayed prefix of one fixed pair."""
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
        examples.append({"length": length, "tokens": inputs["observations"][0].tolist(),
                         "mask": inputs["summary_mask"][0].tolist(), "time_context": inputs["time_context"][0].tolist()})
    return {"maximum": len(values), "examples": examples, "summary_dim": 16,
            "targets": list(task["parameter_names"])}


def render_html(task, observations):
    encoded = json.dumps(build_payload(task, observations), allow_nan=False).replace("</", "<\\/")
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
*{box-sizing:border-box}body{margin:0;padding:10px;font:14px/1.45 system-ui;color:#17233a}.box{border:1px solid #d6dfec;border-radius:10px;padding:14px;background:#fbfcff}
h3{margin:0 0 7px;font-size:17px}.controls{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin:10px 0}button,select{font:inherit;padding:5px 9px;border:1px solid #bcc9db;border-radius:6px;background:#fff;color:#17233a}button{cursor:pointer}#play{background:#2563eb;color:#fff;border-color:#2563eb}
label{font-size:12px}.small{font-size:12px;color:#526177}svg{display:block;width:100%;height:auto;background:white;border:1px solid #e3e9f2;border-radius:8px}.caption{padding:10px;border-left:3px solid #2563eb;background:#eff5ff;min-height:65px;margin-top:10px}
</style></head><body data-widget="training-route"><section class="box" aria-label="Training prefix and network animation">
<h3>Follow one training pair through the networks</h3><p class="small">The return tokens, padding mask and time context below use the real preprocessing code. Moving highlights explain the computation; they are not measured neural activations and do not run an optimizer.</p>
<div class="controls"><button id="play">Play</button><button id="next">Advance</button><button id="reset">Reset</button><label>Observed prefix <select id="prefix" aria-label="Training observed prefix"></select></label><span id="phase" class="small"></span></div>
<svg id="diagram" viewBox="0 0 920 365" role="img" aria-label="Observed prefix to mask to GRU to conditional flow"></svg><div id="caption" class="caption"></div>
<p class="small">Latent variance and process shocks are absent from every encoder input. The fixed generating parameters are training labels routed to the loss. At inference, their values are unknown: only returns, mask and known times are supplied.</p></section>
<script>
const data=__PAYLOAD__,ns='http://www.w3.org/2000/svg';let phase=0,timer=null;const el=id=>document.getElementById(id);
function add(tag,a={},text=null){const n=document.createElementNS(ns,tag);for(const[k,v]of Object.entries(a))n.setAttribute(k,String(v));if(text!==null)n.textContent=text;el('diagram').append(n);return n;}
function stop(){clearInterval(timer);timer=null;el('play').textContent='Play';}function next(){phase=(phase+1)%7;draw();}function play(){if(timer){stop();return;}el('play').textContent='Pause';timer=setInterval(next,1300);}
const captions=[
'1. One simulator draw gives one parameter vector and its complete return history. Parameters are labels; the hidden variance path and process shocks are not network inputs.',
'2. Training chooses a random prefix length. The observations remain unchanged; later positions receive a false mask and padded zeros. A genuine observed zero return retains a true mask.',
'3. The adapter routes scaled return/time tokens to summary_variables, the mask to summary_mask, and known duration/step features to inference_conditions. Transformed parameter labels take a separate route.',
'4. The causal GRU reads observed tokens in order. Each recurrent state depends on the preceding state and the current token. Masked padding does not become additional observations.',
'5. The GRU produces 16 learned summary values for the tutorial model. These summarize evidence about parameters; they are not estimates of latent instantaneous variance.',
'6. The spline coupling flow conditions on the summary plus two known time features. During training it scores the transformed generating parameter vector to obtain the density loss.',
'7. The loss updates both the GRU and the density estimator. After training, conditional sampling returns paired parameter draws for a supplied history; no generating labels are supplied at that query.'];
function draw(){const example=data.examples.find(e=>e.length===Number(el('prefix').value));el('diagram').replaceChildren();document.body.dataset.phase=phase;document.body.dataset.observedLength=example.length;el('phase').textContent='Step '+(phase+1)+' of 7';el('caption').textContent=captions[phase];
 const W=920,L=34,R=28,top=35,width=W-L-R;add('text',{x:L,y:20,'font-size':13,fill:'#17233a'},'One fixed '+data.maximum+'-day history · '+example.length+' observed tokens · '+(data.maximum-example.length)+' padded positions');
 const stride=width/data.maximum;for(let i=0;i<data.maximum;i++){const active=example.mask[i],v=Math.min(2.8,Math.abs(example.tokens[i][0]));add('rect',{x:L+i*stride,y:top+45-(active?v*13:2),width:Math.max(.8,stride-.3),height:active?v*13:4,fill:active?'#3b82f6':'#e2e8f0'});add('rect',{x:L+i*stride,y:94,width:Math.max(.8,stride-.3),height:9,fill:active?'#0f9d80':'#cbd5e1'});}
 add('text',{x:L,y:122,'font-size':11,fill:'#526177'},'True mask: observed return/time tokens');add('text',{x:W-R,y:122,'text-anchor':'end','font-size':11,fill:'#526177'},'False mask: padding, never future data');
 const nodes=[['Adapter',185],['Causal GRU',360],['16-value summary',535],['Coupling flow',710]];for(let j=0;j<nodes.length;j++){const[label,x]=nodes[j],lit=phase>=j+2;add('rect',{x:x-68,y:165,width:136,height:64,rx:8,fill:lit?'#e8f1ff':'#f8fafc',stroke:lit?'#2563eb':'#cbd5e1','stroke-width':lit?2:1});add('text',{x,y:190,'text-anchor':'middle','font-size':13,fill:'#17233a'},label);add('text',{x,y:211,'text-anchor':'middle','font-size':10,fill:'#526177'},j===0?'tokens / mask / time':j===1?'hₜ = GRU(xₜ, hₜ₋₁)':j===2?'learned evidence':'conditional joint density');if(j<3)add('text',{x:x+87,y:202,'text-anchor':'middle','font-size':23,fill:lit?'#2563eb':'#cbd5e1'},'→');}
 add('text',{x:360,y:149,'text-anchor':'middle','font-size':10,fill:'#526177'},phase===3?'Observed tokens are read in order →':'');
 add('rect',{x:113,y:279,width:270,height:42,rx:7,fill:'#fff5e8',stroke:'#dda457'});add('text',{x:248,y:305,'text-anchor':'middle','font-size':12,fill:'#80501b'},'Generating parameters → transformed labels');add('line',{x1:384,x2:710,y1:300,y2:300,stroke:'#dda457','stroke-width':2});add('text',{x:710,y:284,'text-anchor':'middle','font-size':11,fill:'#80501b'},'Density loss');add('line',{x1:710,x2:710,y1:232,y2:272,stroke:'#dda457','stroke-width':2});
 const first=example.tokens[0].map(x=>Number(x).toPrecision(3)).join(', ');add('text',{x:34,y:350,'font-size':10,fill:'#526177'},'Actual first token ['+first+'] · known time context ['+example.time_context.map(x=>Number(x).toPrecision(3)).join(', ')+']');if(phase===6){add('path',{d:'M710 325 L710 335 L360 335 L360 234',fill:'none',stroke:'#7c3aed','stroke-width':2,'stroke-dasharray':'5 3'});add('text',{x:530,y:330,'text-anchor':'middle','font-size':10,fill:'#7c3aed'},'gradients update both networks');}}
for(const e of data.examples){const option=document.createElement('option');option.value=e.length;option.textContent=e.length+' days';el('prefix').append(option);}el('prefix').value=data.examples.find(e=>e.length===63)?.length??data.examples.at(-1).length;
el('play').onclick=play;el('next').onclick=()=>{stop();next();};el('reset').onclick=()=>{stop();phase=0;draw();};el('prefix').onchange=()=>{stop();phase=0;draw();};window.addEventListener('pagehide',stop);draw();
</script></body></html>"""
