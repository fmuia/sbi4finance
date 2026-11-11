"""Pure saved-artifact fixture; no models, simulator pricing or posterior queries."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile

from scripts import render_heston_pr87_grid as per_run
from scripts import render_heston_pr87_grid_campaign_recovery as aggregate
from test_heston_pr87_grid_evaluation import fixture

root = Path(tempfile.mkdtemp(prefix='pr87-grid-campaign-renderer-', dir='/private/tmp'))
write, read, binding = per_run.evaluation.fixed.write_json, per_run.read, per_run.binding
runs = {}
for name, count, best in [('200k-300', 200000, 287), ('300k-300', 300000, 281)]:
    folder = root/name
    task, run, control, protocol, primary, secondary, record = fixture(folder, rows=count, best_epoch=best)
    runs[name] = {'task': task, 'run': run, 'control': control, 'protocol': protocol, 'primary': primary, 'secondary': secondary, 'record': record, 'count': count}
first = runs['200k-300']
old54 = read(first['protocol'])['old54_control']
common_bank = first['record']['common_bank']
common_bank_dir = Path(common_bank['path']).parent
sources = read(first['protocol'])['sources']
sources['producer'] = binding(Path(aggregate.__file__).with_name('reproduce_heston_pr87_grid.py'))
metadata = {}
for name in ['bank-protocol', 'reference-protocol', 'conditional-panel', 'preserved-artifacts']:
    path = root/(name+'.json'); write(path, {'format': 'pure_saved_fixture', 'kind': name}); metadata[name] = binding(path)
write(root/'bank-protocol.json', {'format': 'heston_pr87_grid_bank_study_v1', 'contract': {'unit_fixture': True}, 'output': str(common_bank_dir)})
metadata['bank-protocol'] = binding(root/'bank-protocol.json')
task_path = root/'task.json'; write(task_path, first['task'])
declaration = {'format': 'heston_pr87_grid_campaign_protocol_v1', 'predeclared_before_bank': True,
    'scientific_acceptance': False, 'sources': sources, 'old54_control': old54,
    'bank_protocol': metadata['bank-protocol'], 'reference_protocol': metadata['reference-protocol'],
    'preserved_artifacts': metadata['preserved-artifacts'], 'runs': {}}
campaign = {'format': 'heston_pr87_grid_campaign_v1', 'status': 'running', 'scientific_acceptance': False, 'runs': {}}
for name, item in runs.items():
    study = read(item['protocol']); run = item['run']
    assert item['task'] == first['task']
    run.update(bank_path=str(common_bank_dir), bank_record_sha256=common_bank['sha256'],
        data_payload=str(common_bank_dir/'data.npz'), data_payload_sha256=per_run.evaluation.fixed.digest(common_bank_dir/'data.npz'))
    run_path = Path(item['record']['training_run']['path']); write(run_path, run)
    training_path = run_path.with_name('training.json'); training_path.write_bytes(run_path.read_bytes())
    study.update(format='heston_pr87_grid_study_v1', output=str(run_path.parent), contract=run['protocol'], sources=sources,
        old54_control=old54, bank_protocol=metadata['bank-protocol'], reference_protocol=metadata['reference-protocol'],
        conditional_panel=metadata['conditional-panel'], task_file=binding(task_path))
    write(item['protocol'], study)
    for path in [item['primary'], item['secondary']]:
        record = read(path)
        record.update(protocol=binding(item['protocol']), training_run=binding(run_path), common_bank=common_bank,
            old54_control=old54, reference_protocol=metadata['reference-protocol'], conditional_panel=metadata['conditional-panel'])
        write(path, record)
    result = per_run.build(item['protocol'], item['primary'], Path(item['protocol']).parent/'report.html', secondary_path=item['secondary'])
    report_path = Path(result['report']['path']); document = report_path.read_text()
    extra = '<svg id="fixture-plot" aria-labelledby="fixture-title" viewBox="0 0 1 1"><title id="fixture-title">Shared fixture identifiers</title><defs><clipPath id="fixture-clip"><rect width="1" height="1"/></clipPath></defs><g clip-path="url(#fixture-clip)"><circle id="fixture-point" cx=".5" cy=".5" r=".1"/></g><use href="#fixture-point"/></svg>'
    document = document.replace('<main>', '<main>'+extra)
    report_path.write_text(document)
    summary_path = report_path.with_name('report-summary.json'); summary = read(summary_path)
    summary.update(report=binding(report_path), report_bytes=len(document.encode()), inline_svg_count=document.count('<svg'))
    write(summary_path, summary)
    declaration['runs'][name] = {'train_rows': item['count'], 'protocol': binding(item['protocol'])}
    campaign['runs'][name] = {'training': binding(training_path), 'best_validation_epoch': run['best_validation_archive']['epoch']}
protocol_path = root/'campaign-protocol.json'; write(protocol_path, declaration)
campaign.update(protocol=binding(protocol_path))
campaign_path = root/'campaign.json'; write(campaign_path, campaign)
try:
    aggregate.build(campaign_path, root/'incomplete'/'report.html')
except ValueError: pass
else: raise AssertionError('running campaign accepted')
assert not (root/'incomplete'/'report.html').exists()
campaign['status'] = 'complete'; write(campaign_path, campaign)
result = aggregate.build(campaign_path, root/'aggregate'/'report.html')
document = Path(result['report']['path']).read_text()
assert result['complete_run_bodies_embedded'] == 2 and len(result['endpoints']) == 4
assert result['inline_svg_count'] == 20 and result['report_bytes'] < 1024*1024
assert not result['duplicate_ids'] and not result['scientific_acceptance'] and not result['promotion']
for run in runs:
    assert f'run-{run}-fixture-clip' in document and f'url(#run-{run}-fixture-clip)' in document
    assert f'href="#run-{run}-fixture-point"' in document
assert all(phrase in document for phrase in ['Old54 control', 'different observation tasks', 'Black control variate', 'No official acceptance or promotion'])
# Exact canonical producer aliases pass; same-content unrelated files do not.
case = '200k-300'
training_alias = Path(campaign['runs'][case]['training']['path'])
training_bytes, campaign_bytes = training_alias.read_bytes(), campaign_path.read_bytes()
unrelated = training_alias.with_name('unrelated.json'); unrelated.write_bytes(training_bytes)
try:
    campaign['runs'][case]['training'] = binding(unrelated); write(campaign_path, campaign)
    try: aggregate.build(campaign_path, root/'unrelated-alias'/'report.html')
    except ValueError as error: assert 'canonical' in str(error), str(error)
    else: raise AssertionError('same-content unrelated training record accepted')
finally:
    campaign_path.write_bytes(campaign_bytes); campaign = read(campaign_path)
assert not (root/'unrelated-alias'/'report.html').exists()
# Even a newly bound alias with identical decoded JSON must be byte-identical.
try:
    training_alias.write_bytes(training_bytes+b' ')
    campaign['runs'][case]['training'] = binding(training_alias); write(campaign_path, campaign)
    try: aggregate.build(campaign_path, root/'changed-alias'/'report.html')
    except ValueError as error: assert 'identical bytes' in str(error), str(error)
    else: raise AssertionError('tampered training alias accepted')
finally:
    training_alias.write_bytes(training_bytes); campaign_path.write_bytes(campaign_bytes); campaign = read(campaign_path)
assert not (root/'changed-alias'/'report.html').exists()

# A changed report cannot be silently paired with its saved old digest.
report_path = root/'200k-300'/'report.html'; summary_path = report_path.with_name('report-summary.json')
original_report, original_summary = report_path.read_bytes(), summary_path.read_bytes()
try:
    report_path.write_bytes(original_report+b' ')
    try: aggregate.build(campaign_path, root/'tampered'/'report.html')
    except ValueError: pass
    else: raise AssertionError('tampered report accepted')
finally: report_path.write_bytes(original_report)
# Both bodies are mandatory: oversize fails, without dropping the earlier run.
try:
    document2 = original_report.decode()
    padding = aggregate.LIMIT_BYTES-len(original_report)-1000
    document2 = document2.replace('</main>', '<p>'+'x'*padding+'</p></main>')
    report_path.write_text(document2)
    summary = read(summary_path); summary.update(report=binding(report_path), report_bytes=len(document2.encode()))
    assert summary['report_bytes'] < aggregate.LIMIT_BYTES
    write(summary_path, summary)
    try: aggregate.build(campaign_path, root/'oversize'/'report.html')
    except ValueError as error: assert 'both complete' in str(error), str(error)
    else: raise AssertionError('oversize reports accepted or truncated')
finally:
    report_path.write_bytes(original_report); summary_path.write_bytes(original_summary)
assert not (root/'oversize'/'report.html').exists()
# Recovery formatting retains its original aborted attempt and new resource scope.
abort_bank = root/'aborted-bank'; abort_bank.mkdir()
write(abort_bank/'generation-progress.json', {'completed_pricing_calls': 290000, 'total_pricing_calls': 300612})
old_bank_path = root/'old-bank-protocol.json'
write(old_bank_path, {'contract': {'unit_fixture': True}, 'output': str(abort_bank)})
old_declaration = deepcopy(declaration); old_declaration['bank_protocol'] = binding(old_bank_path)
old_declaration_path = root/'aborted-protocol.json'; write(old_declaration_path, old_declaration)
old_campaign = root/'aborted-campaign.json'
write(old_campaign, {'format': 'heston_pr87_grid_campaign_v1', 'status': 'failed', 'protocol': binding(old_declaration_path),
    'error_type': 'RuntimeError', 'error': 'grid worker process/output failed: reference_synthetic-market-like',
    'jobs': [{'id': 'bank', 'kind': 'bank', 'status': 'stopped_after_peer_failure', 'wall_seconds_limit': 3600},
             {'id': 'reference_synthetic-market-like', 'kind': 'reference', 'status': 'failed', 'wall_seconds_limit': 950, 'returncode': 1}]})
recovery_declaration = deepcopy(declaration)
recovery_declaration.update(format='heston_pr87_grid_recovery_campaign_protocol_v1', recovery_of=binding(old_campaign),
    operational_recovery={'new_reference_budget': '2400s inner /2450s outer', 'failure_isolation': 'References cannot kill bank/fits'})
recovery_protocol = root/'recovery-protocol.json'; write(recovery_protocol, recovery_declaration)
recovery_campaign = root/'recovery-campaign.json'; recovered = deepcopy(campaign)
recovered.update(protocol=binding(recovery_protocol), recovery_of=binding(old_campaign)); write(recovery_campaign, recovered)
recovery_result = aggregate.build(recovery_campaign, root/'recovery-aggregate'/'report.html')
recovery_html = Path(recovery_result['report']['path']).read_text()
assert recovery_result['preserved_aborted_attempt']['campaign'] == binding(old_campaign)
assert all(text in recovery_html for text in ['Initial aborted attempt', '290,000', 'stopped_after_peer_failure', '2400s inner /2450s outer'])
print(json.dumps({'status': 'passed', 'fixture_root': str(root), 'report_bytes': result['report_bytes'],
    'inline_svg_count': result['inline_svg_count'], 'full_bodies': 2, 'endpoints': 4,
    'canonical_training_aliases_pass': True, 'unrelated_same_content_alias_rejected': True, 'changed_alias_even_with_updated_binding_rejected': True, 'identifier_prefix_and_references_valid': True, 'incomplete_campaign_rejected': True,
    'changed_report_rejected': True, 'oversize_rejected_without_truncation': True,
    'recovery_aborted_attempt_and_resource_scope_visible': True, 'scientific_queries_or_pricing': 0, 'aggregate': result['report']}, indent=2))
