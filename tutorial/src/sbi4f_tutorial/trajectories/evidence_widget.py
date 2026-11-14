"""An offline HTML evidence player: reveal one fixed history and actual posteriors.

The browser only selects retained frames. It never interpolates distributions,
simulates a process, queries a model, or treats parameter draws as hidden states.
"""

import json

import numpy as np

from .simulation import validate_task


def build_payload(task, observations, frames, *, end, labels=None, history_label="Selected history", truth=None):
    """Prepare fixed-axis histograms from complete, paired physical draw rows."""
    validate_task(task)
    if isinstance(end, bool) or not isinstance(end, (int, np.integer)) or not 1 <= end <= task["n_observations"]:
        raise ValueError("evidence end must be a supported integer prefix")
    if np.iscomplexobj(observations) or np.ma.isMaskedArray(observations):
        raise ValueError("evidence observations must be real unmasked returns")
    values = np.asarray(observations)
    if values.ndim == 2 and values.shape[1] == 1:
        values = values[:, 0]
    if values.ndim != 1 or len(values) < end or values.dtype.kind not in "fiu" or not np.isfinite(values[:end]).all():
        raise ValueError("provide one finite observed prefix; later values are not read")
    names, bounds = task["parameter_names"], np.asarray(task["prior_bounds"])
    if truth is not None:
        if np.iscomplexobj(truth) or np.ma.isMaskedArray(truth):
            raise ValueError("generating values must be one real complete parameter row")
        truth = np.asarray(truth)
        if (truth.shape != (len(names),) or truth.dtype.kind not in "fiu" or not np.isfinite(truth).all()
                or np.any(truth < bounds[:, 0]) or np.any(truth > bounds[:, 1])):
            raise ValueError("generating values must be one complete row within prior support")
    labels = labels or {}
    bins = [np.linspace(low, high, 25) for low, high in bounds]
    coordinates = []
    for index, (name, kind, edges) in enumerate(zip(names, task["prior_types"], bins, strict=True)):
        low, high = edges[0], edges[-1]
        cdf = (np.log(edges / low) / np.log(high / low) if kind == "log_uniform"
               else (edges - low) / (high - low))
        probabilities = np.array([.05, .5, .95])
        quantiles = (np.exp(np.log(low) + probabilities * np.log(high / low)) if kind == "log_uniform"
                     else low + probabilities * (high - low))
        coordinates.append({"name": name, "label": labels.get(name, name), "edges": edges.tolist(),
                            "prior": (np.diff(cdf) / np.diff(edges)).tolist(), "prior_quantiles": quantiles.tolist(),
                            "truth": None if truth is None else float(truth[index])})
    actual, seen = [], set()
    for frame in sorted(frames, key=lambda item: item["length"]):
        length = frame["length"]
        if isinstance(length, bool) or not isinstance(length, (int, np.integer)) or not 1 <= length <= end or length in seen:
            raise ValueError("posterior frames need unique exact integer prefix lengths within the playback end")
        if np.iscomplexobj(frame["draws"]) or np.ma.isMaskedArray(frame["draws"]):
            raise ValueError("posterior frames require real unmasked paired rows")
        draws = np.asarray(frame["draws"])
        if (draws.ndim != 2 or len(draws) < 2 or draws.shape[1] != len(names) or draws.dtype.kind not in "fiu"
                or not np.isfinite(draws).all() or np.any(draws < bounds[:, 0]) or np.any(draws > bounds[:, 1])):
            raise ValueError("evidence frames require finite complete physical posterior rows within prior support")
        source = {"saved": "Saved neural posterior", "native": "Fresh exact-prefix neural posterior"}.get(frame["source"], str(frame["source"]))
        actual.append({"length": int(length), "source": source, "frame_id": str(frame["frame_id"]),
                       "count": len(draws), "densities": [np.histogram(draws[:, i], edges, density=True)[0].tolist()
                                                          for i, edges in enumerate(bins)],
                       "quantiles": np.quantile(draws, [.05, .5, .95], axis=0).T.tolist()})
        seen.add(length)
    # The day-zero display is the analytic declared prior, not a neural query.
    prior = {"length": 0, "source": "Declared prior · no observations", "frame_id": "prior", "count": 0,
             "densities": [coordinate["prior"] for coordinate in coordinates],
             "quantiles": [coordinate["prior_quantiles"] for coordinate in coordinates]}
    all_frames = [prior, *actual]
    for i, coordinate in enumerate(coordinates):
        coordinate["density_max"] = 1.12 * max(max(frame["densities"][i]) for frame in all_frames)
    return {"format": "trajectory_evidence_widget_v1", "end": int(end), "history_label": str(history_label),
            "returns": (100 * values[:end]).tolist(), "coordinates": coordinates,
            "frames": all_frames, "return_limits": [-8., 8.], "week_days": 5, "month_days": 21}


def render_html(task, observations, frames, *, end, labels=None, history_label="Selected history", truth=None):
    """Build a dependency-free document for the documented ``mo.iframe`` API."""
    payload = build_payload(task, observations, frames, end=end, labels=labels, history_label=history_label, truth=truth)
    encoded = json.dumps(payload, allow_nan=False, separators=(",", ":")).replace("</", "<\\/")
    return _DOCUMENT.replace("__PAYLOAD__", encoded)


_DOCUMENT = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<style>
*{box-sizing:border-box}body{margin:0;padding:12px;color:#162138;background:#fff;font:14px/1.45 system-ui,sans-serif}
.player{border:1px solid #d6dfec;border-radius:12px;padding:14px;background:#fbfcff}h3{font-size:17px;margin:0 0 6px}
.muted{font-size:12px;color:#526177}.controls{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:12px 0}
button,select{font:inherit;border:1px solid #bbc8d9;border-radius:6px;padding:6px 10px;background:white;color:#162138}
button{cursor:pointer}button:disabled{opacity:.42;cursor:default}button.primary{background:#2563eb;color:white;border-color:#2563eb}
label{font-size:12px;display:flex;gap:5px;align-items:center}.scrub{display:grid;grid-template-columns:1fr auto;gap:10px;align-items:center;margin-bottom:5px}
input[type=range]{width:100%;accent-color:#2563eb}#position{font-variant-numeric:tabular-nums;font-size:13px;font-weight:650}
svg{display:block;width:100%;height:auto}.path{background:#fff;border:1px solid #e4e9f1;border-radius:8px;margin-top:10px}
.parameters{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px;margin-top:12px}.card{background:#fff;border:1px solid #e0e7f0;border-radius:8px;padding:10px}
.card h4{font-size:13px;margin:0 0 2px}.readout{font-size:11px;color:#43516a;font-variant-numeric:tabular-nums}
.legend{font-size:11px;display:flex;gap:14px;flex-wrap:wrap;color:#526177;margin:8px 0}.swatch{display:inline-block;width:20px;border-top:3px solid #2563eb;vertical-align:middle;margin-right:4px}
.status{font-size:12px;border-left:3px solid #2563eb;padding:5px 8px;margin-top:9px;background:#eef4ff;overflow-wrap:anywhere}
@media(max-width:470px){.parameters{grid-template-columns:1fr}.controls{gap:6px}body{padding:3px}.player{padding:10px}}
</style></head><body>
<section class="player" aria-label="Evidence reveal player">
 <h3>Reveal one fixed history</h3>
 <div class="muted" id="context"></div>
 <div class="controls"><button id="play" class="primary">Play</button><button id="advance">Advance</button><button id="reset">Reset</button>
 <label>Axis units <select id="units" aria-label="Axis units"><option value="1">Trading days</option><option value="5">Trading weeks</option><option value="21" selected>Trading months</option></select></label>
 <label>Early-time zoom <select id="zoom" aria-label="Early-time zoom"></select></label>
 <label>Playback <select id="speed" aria-label="Playback speed"><option value="1100">Slow</option><option value="500">Normal</option><option value="200" selected>Fast</option></select></label></div>
 <div class="scrub"><input id="scrub" type="range" min="0" value="0" aria-label="Actual posterior frame"><output id="position" aria-live="polite"></output></div>
 <div class="muted" id="schedule"></div>
 <div class="path"><svg id="path" viewBox="0 0 900 190" role="img" aria-label="Revealed observed daily log-return path"></svg></div>
 <div class="legend"><span><i class="swatch"></i>Current empirical posterior</span><span><i class="swatch" style="border-color:#94a3b8;border-top-style:dashed"></i>Declared prior</span><span><i class="swatch" style="border-color:#111827;border-top-style:dotted"></i>Generating value (synthetic)</span><span>Shading: central 90% marginal interval</span><span>Y axes: marginal density, fixed across frames</span></div>
 <div id="parameters" class="parameters"></div>
 <div id="status" class="status"></div>
 <p class="muted" style="margin-bottom:0">One parameter vector governs the whole history. Each posterior conditions on the original prior and its entire revealed prefix. Playback reveals observations; it does not propagate hidden variance, restart a process, or recursively multiply overlapping posteriors. Histogram scales stay fixed throughout these prepared frames.</p>
</section>
<script>
const data=__PAYLOAD__;
const byId=id=>document.getElementById(id), ns='http://www.w3.org/2000/svg';
let index=0,timer=null,available=data.frames;
function svgEl(tag,attributes={},text=null){const el=document.createElementNS(ns,tag);for(const[k,v]of Object.entries(attributes))el.setAttribute(k,String(v));if(text!==null)el.textContent=text;return el;}
function add(svg,tag,attributes={},text=null){const el=svgEl(tag,attributes,text);svg.append(el);return el;}
function number(x){return Number(x).toPrecision(3).replace(/\.0+(?=e|$)/,'');}
function stop(){if(timer!==null)clearInterval(timer);timer=null;byId('play').textContent='Play';byId('play').setAttribute('aria-pressed','false');}
function advance(){if(index<available.length-1){index++;draw();}if(index>=available.length-1)stop();}
function play(){if(timer!==null){stop();return;}if(index>=available.length-1)index=0;byId('play').textContent='Pause';byId('play').setAttribute('aria-pressed','true');timer=setInterval(advance,Number(byId('speed').value));draw();}
function drawPath(frame){const svg=byId('path');svg.replaceChildren();const end=Number(byId('zoom').value),unit=Number(byId('units').value),W=900,H=190,L=55,R=20,T=20,B=38;
 const x=d=>L+d/end*(W-L-R),y=r=>T+(8-r)/16*(H-T-B);add(svg,'rect',{x:x(frame.length),y:T,width:Math.max(0,x(end)-x(frame.length)),height:H-T-B,fill:'#f1f4f8'});
 for(const r of[-8,-4,0,4,8]){add(svg,'line',{x1:L,x2:W-R,y1:y(r),y2:y(r),stroke:r===0?'#b7c4d5':'#e7ecf3'});add(svg,'text',{x:L-8,y:y(r)+3,'text-anchor':'end','font-size':10,fill:'#526177'},r);}
 const ticks=[...new Set(Array.from({length:5},(_,t)=>unit===1?Math.round(end*t/4):end*t/4))];for(const d of ticks)add(svg,'text',{x:x(d),y:H-B+17,'text-anchor':'middle','font-size':10,fill:'#526177'},unit===1?String(d):number(d/unit));
 let points=[],outside=0;for(let d=1;d<=frame.length;d++){const r=data.returns[d-1];if(r<-8||r>8)outside++;points.push(x(d)+','+y(Math.max(-8,Math.min(8,r))));if(r<-8||r>8)add(svg,'circle',{cx:x(d),cy:y(Math.max(-8,Math.min(8,r))),r:2.5,fill:'#c2410c'});}
 if(points.length>1)add(svg,'polyline',{points:points.join(' '),fill:'none',stroke:'#2563eb','stroke-width':1.2});else if(points.length)add(svg,'circle',{cx:x(1),cy:y(Math.max(-8,Math.min(8,data.returns[0]))),r:2.8,fill:'#2563eb'});
 add(svg,'line',{x1:x(frame.length),x2:x(frame.length),y1:T,y2:H-B,stroke:'#111827','stroke-width':1.1});
 add(svg,'text',{x:L,y:12,'font-size':11,fill:'#162138'},'Observed daily log returns (%) · '+outside+'/'+frame.length+' points outside the fixed ±8% view');
 add(svg,'text',{x:(W+L-R)/2,y:H-2,'text-anchor':'middle','font-size':10,fill:'#526177'},unit===1?'Trading day':unit===5?'Trading week (5 days)':'Trading month (21 days)');
 if(end-frame.length>end*.16)add(svg,'text',{x:(x(end)+x(frame.length))/2,y:T+18,'text-anchor':'middle','font-size':11,fill:'#7a879a'},'Not revealed');
}
function drawCards(frame){const holder=byId('parameters');holder.replaceChildren();for(let i=0;i<data.coordinates.length;i++){const c=data.coordinates[i],card=document.createElement('section');card.className='card';const heading=document.createElement('h4');heading.textContent=c.label;card.append(heading);const svg=svgEl('svg',{viewBox:'0 0 390 145',role:'img','aria-label':c.label+' marginal at '+frame.length+' observed days'});card.append(svg);
 const L=35,R=12,T=8,B=27,W=390,H=145,low=c.edges[0],high=c.edges[c.edges.length-1],x=v=>L+(v-low)/(high-low)*(W-L-R),y=v=>H-B-v/c.density_max*(H-T-B),q=frame.quantiles[i];
 add(svg,'rect',{x:x(q[0]),y:T,width:x(q[2])-x(q[0]),height:H-T-B,fill:'#edf3ff'});
 for(let j=0;j<frame.densities[i].length;j++)add(svg,'rect',{x:x(c.edges[j])+.3,y:y(frame.densities[i][j]),width:Math.max(.5,x(c.edges[j+1])-x(c.edges[j])-.6),height:H-B-y(frame.densities[i][j]),fill:frame.length?'#7ba6f6':'#c6d1e1',opacity:.8});
 let prior=[];for(let j=0;j<c.prior.length;j++){prior.push(x(c.edges[j])+','+y(c.prior[j]));prior.push(x(c.edges[j+1])+','+y(c.prior[j]));}add(svg,'polyline',{points:prior.join(' '),fill:'none',stroke:'#64748b','stroke-width':1.5,'stroke-dasharray':'4 3'});
 add(svg,'line',{x1:x(q[1]),x2:x(q[1]),y1:T,y2:H-B,stroke:'#1d4ed8','stroke-width':1.4});add(svg,'line',{x1:L,x2:W-R,y1:H-B,y2:H-B,stroke:'#b7c4d5'});
 if(c.truth!==null)add(svg,'line',{x1:x(c.truth),x2:x(c.truth),y1:T,y2:H-B,stroke:'#111827','stroke-width':1.5,'stroke-dasharray':'2 3','data-generating-value':c.truth});
 for(let j=0;j<3;j++){const v=low+(high-low)*j/2;add(svg,'text',{x:x(v),y:H-B+16,'text-anchor':'middle','font-size':11,fill:'#526177'},number(v));}
 add(svg,'text',{x:L-3,y:15,'text-anchor':'end','font-size':9,fill:'#7a879a'},number(c.density_max));add(svg,'text',{x:L-3,y:H-B,'text-anchor':'end','font-size':9,fill:'#7a879a'},'0');
 const readout=document.createElement('div');readout.className='readout';readout.textContent='Median '+number(q[1])+' · central 90% ['+number(q[0])+', '+number(q[2])+']';card.append(readout);holder.append(card);
}}
function draw(){const frame=available[index];byId('scrub').max=available.length-1;byId('scrub').value=index;byId('position').textContent=frame.length===0?'Day 0 · prior':'Day '+frame.length+' · frame '+index+'/'+(available.length-1);document.body.dataset.currentLength=frame.length;document.body.dataset.frameId=frame.frame_id;
 byId('schedule').textContent=(available.length-1)+' actual posterior frames in this view. Slider positions select their exact days; no distribution is interpolated.';byId('advance').disabled=index>=available.length-1;byId('play').disabled=available.length<2;
 byId('status').textContent=frame.length===0?'Declared prior before revealing observations; no network has been queried for this display.':frame.source+' · conditioned on exactly '+frame.length+' observed days · '+frame.count+' paired draws. These are approximate posteriors, with the calibration limitations described in the lesson.';
 drawPath(frame);drawCards(frame);
}
byId('context').textContent=data.history_label+' · local playback ends at or before the '+data.end+' days selected above. This player does not change the other notebook views.';
for(const end of [...new Set([Math.min(21,data.end),Math.min(63,data.end),data.end])]){const option=document.createElement('option');option.value=end;option.textContent=end===data.end?'All '+end+' selected days':'First '+end+' days';byId('zoom').append(option);}byId('zoom').value=data.end;available=data.frames.filter(frame=>frame.length<=Number(byId('zoom').value));
byId('play').onclick=play;byId('advance').onclick=()=>{stop();advance();};byId('reset').onclick=()=>{stop();index=0;draw();};byId('scrub').oninput=()=>{stop();index=Number(byId('scrub').value);draw();};byId('units').onchange=draw;byId('zoom').onchange=()=>{stop();const previous=available[index].length;available=data.frames.filter(frame=>frame.length<=Number(byId('zoom').value));index=Math.max(0,available.findLastIndex(frame=>frame.length<=previous));draw();};byId('speed').onchange=()=>{if(timer!==null){stop();play();}};
window.addEventListener('pagehide',stop);draw();
</script></body></html>"""
