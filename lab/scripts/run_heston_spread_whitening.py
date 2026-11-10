"""Freeze and supervise a bounded diagonal-versus-ZCA Heston pilot."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_heston_spread_scale_capacity as shared

MODES = ('diagonal', 'zca')
LIMITS = {'worker_wall_seconds': 900, 'whole_training_wall_seconds': 1000,
          'max_concurrent_workers': 2, 'automatic_retry': False, 'automatic_extension': False}


def identity(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': shared.digest(path)}


def prepare(previous_path, evidence, payload, notebook):
    from scripts import train_heston_spread_whitening as trainer
    from sbi4f_tutorial.shared import whitening
    from scripts import evaluate_heston_spread_whitening as evaluator
    previous_path, evidence, payload, notebook = map(Path, (previous_path, evidence, payload, notebook))
    if evidence.exists() or payload.exists():
        raise FileExistsError('whitening pilot identities must be new')
    old = shared.read(previous_path)
    if old['format'] != 'heston_spread_scale_precision_recovery_study_v1':
        raise ValueError('frozen precision-safe task source required')
    parent = shared.read(Path(old['parent_checkpoint']['path'])/'checkpoint.json')
    if shared.digest(Path(old['parent_checkpoint']['path'])/'checkpoint.json') != old['parent_checkpoint']['record']['sha256']:
        raise ValueError('historical task source changed')
    task, bank = parent['task'], old['bank']
    arms = [{'mode': mode, 'contract': trainer.protocol_contract(bank['path'], task, mode=mode),
             'output': str((payload/mode).resolve()), 'log': str((payload/(mode+'.log')).resolve())}
            for mode in MODES]
    record = {'format': 'heston_spread_whitening_study_v1', 'predeclared_before_fit': True,
        'declared_utc': datetime.now(timezone.utc).isoformat(), 'task': task,
        'task_path': str((evidence/'task.json').resolve()), 'joint_task': task['target'],
        'evaluation': evaluator.evaluation_contract(task['target']), 'bank': bank, 'arms': arms,
        'historical_study': identity(previous_path), 'campaign': LIMITS,
        'sources': {'trainer': identity(trainer.__file__), 'controller': identity(__file__),
                    'whitening': identity(whitening.__file__), 'evaluator': identity(evaluator.__file__)},
        'upstream_notebook': {'pr': 'https://github.com/fmuia/sbi4finance/pull/87',
            'commit': 'ddf3619faf48b8f4b0ee56ee841ef24124a508d1',
            'repository_path': 'lab/notebooks/heston_6_parameter_bayesflow.ipynb',
            'sha256': shared.digest(notebook), 'local_copy': str((payload/'upstream-notebook.ipynb').resolve())},
        'baseline_evaluation': old['baseline_evaluation'],
        'original_study': old['original_study'], 'reference_recovery': old['reference_recovery'],
        'matched_mcmc_report': identity(evidence.parent/'mcmc-recovery/report-manifest.json'),
        'comparison': 'Two fresh paired spline networks differ only in fixed diagonal versus full ZCA input transform; both retain bounded targets and zero dropout. Both fit transforms on the same noisy training rows only.',
        'scope': 'Bounded preprocessing pilot; historical warm-start/fresh-noise results are contextual, not matched training controls.',
        'reserved_arrays_opened': False, 'new_pricing_labels': False, 'new_reference_sampling': False,
        'scientific_acceptance': False, 'automatic_extension': False}
    evidence.mkdir(parents=True); payload.mkdir(parents=True)
    shutil.copyfile(notebook, payload/'upstream-notebook.ipynb')
    shared.write(evidence/'task.json', task)
    record['task_file'] = identity(evidence/'task.json')
    shared.write(evidence/'protocol.json', record)
    return evidence/'protocol.json'


def validate(protocol):
    if (protocol.get('format') != 'heston_spread_whitening_study_v1'
            or [a.get('mode') for a in protocol.get('arms', [])] != list(MODES)
            or protocol.get('campaign') != LIMITS):
        raise ValueError('declared bounded whitening pilot required')
    for name, source in protocol['sources'].items():
        if shared.digest(source['path']) != source['sha256']:
            raise ValueError('declared source changed: '+name)
    if protocol['sources']['controller']['sha256'] != shared.digest(__file__):
        raise ValueError('controller source differs')
    if (shared.digest(protocol['task_file']['path']) != protocol['task_file']['sha256']
            or shared.read(protocol['task_path']) != protocol['task']):
        raise ValueError('statistical task changed')
    outputs = [str(Path(a[k]).resolve()) for a in protocol['arms'] for k in ('output','log')]
    if len(outputs) != len(set(outputs)):
        raise ValueError('arm identities must be distinct')


def run(protocol_path, out):
    protocol_path, out = Path(protocol_path).resolve(), Path(out).resolve()
    started = time.monotonic(); protocol = shared.read(protocol_path); validate(protocol)
    if out.exists() or out.with_suffix('.claim.json').exists() or any(Path(a[k]).exists() for a in protocol['arms'] for k in ('output','log')):
        raise FileExistsError('pilot outputs must be new; no retries')
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.with_suffix('.claim.json').open('x') as stream:
        json.dump(identity(protocol_path), stream)
    result = {'format': 'heston_spread_whitening_campaign_v1', 'status': 'running',
        'protocol': identity(protocol_path), 'started_utc': datetime.now(timezone.utc).isoformat(),
        'git_revision': subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'arms': {}, 'automatic_retry': False, 'automatic_extension': False, 'scientific_acceptance': False}
    workers = {}; shared.write(out,result)
    try:
        for arm in protocol['arms']:
            log = Path(arm['log']).open('x')
            command = [sys.executable, protocol['sources']['trainer']['path'], '--bank', protocol['bank']['path'],
                '--task', protocol['task_path'], '--mode', arm['mode'], '--protocol', str(protocol_path), '--out', arm['output'], '--worker']
            env = dict(os.environ); env.update(KERAS_BACKEND='jax', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                VECLIB_MAXIMUM_THREADS='1', MPLCONFIGDIR='/private/tmp/sbi4f-matplotlib')
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
            workers[arm['mode']] = {'process': process, 'log': log, 'started': time.monotonic(), 'arm': arm}
            result['arms'][arm['mode']] = {'status':'running','pid':process.pid,'command':command}
            shared.write(out,result)
        while any(w['process'].poll() is None for w in workers.values()):
            now = time.monotonic()
            for mode,w in workers.items():
                if w['process'].poll() is None and (now-w['started']>=900 or now-started>=1000):
                    shared.stop(w['process']);result['arms'][mode]['status']='wall_cap'
                if w['process'].poll() is not None and 'finished' not in w:w['finished']=time.monotonic()
            time.sleep(1)
        for mode,w in workers.items():
            entry=result['arms'][mode];train=Path(w['arm']['output'])/'training.json'
            if entry['status']!='wall_cap':entry['status']='complete' if w['process'].returncode==0 and train.exists() else 'failed'
            entry.update(returncode=w['process'].returncode,seconds=w.get('finished',time.monotonic())-w['started'])
            if train.exists():entry['training']=identity(train)
            w['log'].close()
        validate(protocol)
        if shared.digest(protocol_path)!=result['protocol']['sha256']:raise ValueError('protocol changed')
        result.update(status='complete' if all(a['status']=='complete' for a in result['arms'].values()) else 'incomplete',seconds=time.monotonic()-started)
        shared.write(out,result);return result
    except BaseException as error:
        for w in workers.values():shared.stop(w['process']);w['log'].close()
        result.update(status='failed',error_type=type(error).__name__,error=str(error));shared.write(out,result);raise


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    prep=sub.add_parser('prepare')
    for key in ('previous','evidence','payload','notebook'):prep.add_argument('--'+key,type=Path,required=True)
    launch=sub.add_parser('run')
    for key in ('protocol','out'):launch.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args()
    if args.command=='prepare':print(prepare(args.previous,args.evidence,args.payload,args.notebook))
    else:
        result=run(args.protocol,args.out);print(json.dumps(result),flush=True)
        if result['status']!='complete':raise SystemExit(1)
