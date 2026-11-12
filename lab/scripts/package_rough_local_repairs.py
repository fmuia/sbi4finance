"""Package and verify a separately keyed local-repair overlay; never prices options."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import shutil
import sqlite3
import tarfile
import numpy as np

FORMAT = 'rough_local_repair_upload_package_v1'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def safe(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('unsafe archive-relative path')
    return root / relative


def equal(left, right):
    return np.array_equal(left, right, equal_nan=left.dtype.kind in 'fc')


def audit_check(audit, plan_sha, row_hashes):
    if (audit.get('format') != 'rough_actual63_independent_local_repair_payload_check_v1'
            or audit.get('complete') is not True or audit.get('complete_count') != 87
            or audit.get('requested_indices') != list(range(87))
            or audit.get('missing_indices') != [] or audit.get('failed_indices') != []
            or audit.get('prices_generated') != 0
            or audit.get('original_transport_hashes_unchanged') is not True
            or audit.get('repair_plan', {}).get('sha256') != plan_sha):
        raise ValueError('successful independent all87 audit is required')
    audit_rows = audit['rows']
    if len(audit_rows) != 87 or sorted(row['index'] for row in audit_rows) != list(range(87)):
        raise ValueError('independent audit row indices must be unique and complete')
    checked = {row['index']: row['payload_sha256'] for row in audit_rows}
    if checked != row_hashes:
        raise ValueError('independent audit does not bind these exact87 payloads')


def validate_tree(root):
    manifest = read(root / 'package-manifest.json')
    if (manifest.get('format') != FORMAT or manifest['row_count'] != 87
            or manifest['originals_modified'] or manifest['scientific_acceptance']):
        raise ValueError('separate87-row candidate-overlay identity required')
    actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
    if actual != set(manifest['files']) | {'package-manifest.json'}:
        raise ValueError('archive inventory differs; extra or missing files')
    for relative, identity in manifest['files'].items():
        path = safe(root, relative)
        if digest(path) != identity['sha256'] or path.stat().st_size != identity['bytes']:
            raise ValueError('archived file content changed: ' + relative)
    relocation = read(root / 'relocation.json')

    def relocated(binding):
        mapped = relocation['direct_files'][binding['path']]
        relative = mapped['archive_path']
        if (mapped['sha256'] != binding['sha256'] or relative not in manifest['files']
                or manifest['files'][relative]['sha256'] != binding['sha256']):
            raise ValueError('relocation binding differs from archived inventory')
        path = safe(root, relative)
        if digest(path) != binding['sha256']:
            raise ValueError('immutable absolute-path binding differs after relocation')
        return path

    plan_path = root / 'repair/plan.json'
    plan = read(plan_path)
    plan_sha = digest(plan_path)
    if len(manifest['rows']) != 87 or len(plan['rows']) != 87:
        raise ValueError('exactly87 declared and enumerated rows required')
    for path, expected in plan['transport_files'].items():
        relocated({'path': path, 'sha256': expected})
    transport_path = relocated(plan['transport_manifest'])
    transport = read(transport_path)
    source_bundle = safe(transport_path.parent, transport['bundle']['path'])
    original = sqlite3.connect('file:' + str(source_bundle) + '?mode=ro', uri=True)
    if original.execute('PRAGMA integrity_check').fetchone() != ('ok',):
        raise ValueError('input bundle integrity differs')
    combined = np.load(root / 'combined.npz', allow_pickle=False)
    keys = set()
    row_hashes = {}
    reused = recomputed = 0
    try:
        for index, row in enumerate(manifest['rows']):
            if row['index'] != index:
                raise ValueError('stable original repair-row ordering required')
            item = plan['rows'][index]
            key = (item['plan_sha256'], item['bank_lane'], item['split'], item['id'])
            if key in keys or row['join_key'] != list(key):
                raise ValueError('duplicate or changed original physical-row key')
            keys.add(key)
            row_manifest = read(safe(root, row['manifest']))
            identity = row_manifest['identity']
            if (row_manifest['status'] not in ('complete', 'complete_candidate')
                    or identity['repair_plan_sha256'] != plan_sha
                    or identity['row_index'] != index
                    or identity['original_identity'] != item['identity']
                    or identity['original_row_id'] != item['id']
                    or row_manifest['SBI_training_readiness']
                    or row_manifest['scientific_acceptance']):
                raise ValueError('row provenance or candidate scope differs')
            payload_path = relocated(row_manifest['payload'])
            journal_path = relocated(row_manifest['journal'])
            row_hashes[index] = digest(payload_path)
            data = np.load(payload_path, allow_pickle=False)
            try:
                for name in data.files:
                    if not equal(data[name], combined[name][index]):
                        raise ValueError('combined array differs from exact row payload: ' + name)
                if (data['parameters'].shape != (7,) or data['physical_parameters'].shape != (6,)
                        or data['reference_prices_forward'].shape != (63,)
                        or data['observations_forward'].shape != (63,)):
                    raise ValueError('seven-target63-quote payload required')
                for name in ['parameters', 'physical_parameters', 'reference_prices_forward',
                             'observations_forward', 'pricing_covariance_forward']:
                    if not np.isfinite(data[name]).all():
                        raise ValueError('nonfinite repaired payload: ' + name)
                spec = identity['original_failure']['spec']
                if (not np.array_equal(data['physical_parameters'], spec['parameters'])
                        or float(data['noise_scale']) != spec['noise_scale']):
                    raise ValueError('original stored physical truths or c changed')
                producer = transport['producers'][item['producer_kind']]
                old_plan = read(safe(transport_path.parent, producer['plan']['path']))
                h = np.asarray(old_plan['tasks'][item['bank_lane']]['design']['half_spread_forward'])
                if not np.array_equal(data['observations_forward'], data['reference_prices_forward']
                                      + float(data['noise_scale']) * h * data['quote_standard_normals']):
                    raise ValueError('original c-times-spread observation law changed')
                if data['refinement_evaluated'] or data['continuous_bias_certified']:
                    raise ValueError('no unevaluated continuous-grid certification permitted')
            finally:
                data.close()
            journal = sqlite3.connect('file:' + str(journal_path) + '?mode=ro', uri=True)
            try:
                if journal.execute('PRAGMA integrity_check').fetchone() != ('ok',):
                    raise ValueError('repaired journal integrity differs')
                stored_identity = journal.execute('SELECT value FROM identity').fetchall()
                if len(stored_identity) != 1 or json.loads(stored_identity[0][0]) != identity:
                    raise ValueError('row journal identity differs')
                for block, rung, status, summary, blob, origin in journal.execute(
                        'SELECT block,rung,status,summary,payload,origin FROM attempts'):
                    if origin == 'oscar_reused':
                        found = original.execute('SELECT status,summary,payload FROM attempts WHERE source_key=? AND id=? AND block=? AND rung=?',
                                                 (item['source_key'], item['id'], block, rung)).fetchone()
                        if found != (status, summary, blob):
                            raise ValueError('reused original attempt was changed')
                        reused += 1
                    elif origin == 'local_recomputed':
                        value = json.loads(summary)
                        expected_entropy = [*spec['identity_entropy'], 1, 1, block, rung]
                        ladder = old_plan['numerical_policy']['paths_ladder']
                        if (item['bank_lane'] != 'joint_unknown' or not 0 <= rung < len(ladder)
                                or value['entropy'] != expected_entropy or value['paths'] != ladder[rung]):
                            raise ValueError('local recomputed attempt seed or path rung changed')
                        recomputed += 1
                    else:
                        raise ValueError('unknown block execution origin')
            finally:
                journal.close()
            for name, value in [('original_plan_sha256', key[0]), ('bank_lane', key[1]),
                                ('split', key[2]), ('original_row_id', key[3])]:
                if combined[name][index].item() != value:
                    raise ValueError('combined original join-key columns differ')
        audit = read(root / 'audit/independent-all87-payload-check.json')
        audit_check(audit, plan_sha, row_hashes)
        if digest(root / 'audit/independent-check.py') != audit['checker_sha256']:
            raise ValueError('archived independent checker source differs')
        for relative, expected in plan['source_bindings'].items():
            if digest(safe(root / 'sources', relative)) != expected:
                raise ValueError('source closure differs')
    finally:
        original.close()
        combined.close()
    return {'rows_verified': 87, 'unique_original_join_keys': len(keys),
            'reused_attempts_byte_verified': reused, 'local_recomputed_attempts': recomputed,
            'all_row_and_combined_arrays_match': True, 'independent_audit_bound': True,
            'source_files_verified': len(plan['source_bindings']),
            'originals_modified': False, 'scientific_acceptance': False, 'simulation_calls': 0}


def package(repo, run, out, audit_path, audit_checker):
    if out.exists():
        raise FileExistsError('new immutable package destination required')
    plan_path = run / 'plan.json'
    plan = read(plan_path)
    batch = read(run / 'output/batch-status.json')
    if (batch['status'] != 'finished' or batch['requested'] != 87
            or batch['pending'] or batch['active']):
        raise ValueError('all local workers must have finished before packaging')
    plan_sha = digest(plan_path)
    audit = read(audit_path)
    row_hashes = {}
    payloads = []
    row_records = []
    for index, item in enumerate(plan['rows']):
        directory = run / 'output/rows' / ('row-%03d' % index)
        record = read(directory / 'manifest.json')
        if record['status'] not in ('complete', 'complete_candidate'):
            raise ValueError('all87 complete repair outcomes required for this package')
        row_hashes[index] = digest(directory / 'data.npz')
        with np.load(directory / 'data.npz', allow_pickle=False) as data:
            payloads.append({name: data[name].copy() for name in data.files})
        row_records.append({'index': index, 'join_key': [item['plan_sha256'], item['bank_lane'], item['split'], item['id']],
                            'manifest': 'repair/output/rows/row-%03d/manifest.json' % index})
    audit_check(audit, plan_sha, row_hashes)
    if digest(audit_checker) != audit['checker_sha256']:
        raise ValueError('independent checker source differs')
    if len({tuple(row['join_key']) for row in row_records}) != 87:
        raise ValueError('duplicate original bank-row key')
    out.mkdir(parents=True, mode=0o700)
    direct = {}

    def copy(source, relative, expected=None):
        source = Path(source).resolve()
        if expected is not None and digest(source) != expected:
            raise ValueError('source file changed before packaging: ' + str(source))
        target = safe(out, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if digest(target) != digest(source):
                raise ValueError('conflicting archive path')
        else:
            shutil.copyfile(source, target)
        direct[str(source)] = {'archive_path': relative, 'sha256': digest(source)}

    for name in ['plan.json', 'launch.json', 'run_batch.py', 'supervisor.log', 'proof-verification.json']:
        if (run / name).is_file():
            copy(run / name, 'repair/' + name)
    copy(run / 'output/batch-status.json', 'repair/output/batch-status.json')
    for index in range(87):
        directory = run / 'output/rows' / ('row-%03d' % index)
        for name in ['manifest.json', 'data.npz', 'journal.sqlite']:
            copy(directory / name, 'repair/output/rows/row-%03d/%s' % (index, name))
        log = run / 'output' / ('worker-%03d.log' % index)
        if log.is_file():
            copy(log, 'repair/output/' + log.name)
    transport_path = Path(plan['transport_manifest']['path'])
    for source in sorted(transport_path.parent.rglob('*')):
        if source.is_file():
            copy(source, 'inputs/bundle/' + str(source.relative_to(transport_path.parent)))
    copy(plan['transport_archive']['path'], 'inputs/bundle.tar.gz', plan['transport_archive']['sha256'])
    for path, expected in plan['transport_files'].items():
        if digest(path) != expected or str(Path(path).resolve()) not in direct:
            raise ValueError('original transport binding missing or changed')
    for relative, expected in plan['source_bindings'].items():
        copy(repo / relative, 'sources/' + relative, expected)
    copy(audit_path, 'audit/independent-all87-payload-check.json')
    copy(audit_checker, 'audit/independent-check.py', audit['checker_sha256'])
    copy(Path(__file__), 'tools/package_rough_local_repairs.py')
    names = set(payloads[0])
    if any(set(data) != names for data in payloads):
        raise ValueError('all repaired rows must preserve the same payload schema')
    combined = {name: np.stack([data[name] for data in payloads]) for name in names}
    combined.update(original_plan_sha256=np.asarray([r['join_key'][0] for r in row_records]),
                    bank_lane=np.asarray([r['join_key'][1] for r in row_records]),
                    split=np.asarray([r['join_key'][2] for r in row_records]),
                    original_row_id=np.asarray([r['join_key'][3] for r in row_records], dtype=np.int64),
                    repair_row_index=np.arange(87, dtype=np.int64),
                    producer_kind=np.asarray([r['producer_kind'] for r in plan['rows']]),
                    source_key=np.asarray([r['source_key'] for r in plan['rows']]))
    np.savez_compressed(out / 'combined.npz', **combined)
    write(out / 'relocation.json', {
        'format': 'rough_local_repair_relocation_map_v1', 'direct_files': direct,
        'original_manifests_rewritten': False,
        'original_oscar_journal_references': [
            {'source_journal': row['source_journal'], 'source_key': row['source_key'],
             'selected_tables_archive_path': 'inputs/bundle/bundle.sqlite',
             'scope': 'selected failed-row tables only; not a complete original shard snapshot'}
            for row in plan['rows']],
    })
    inventory = {str(path.relative_to(out)): {'sha256': digest(path), 'bytes': path.stat().st_size}
                 for path in sorted(out.rglob('*')) if path.is_file()}
    write(out / 'package-manifest.json', {
        'format': FORMAT, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'row_count': 87, 'repair_plan_sha256': plan_sha, 'rows': row_records,
        'files': inventory, 'combined_payload': 'combined.npz',
        'join_rule': plan['join_rule'], 'merge_action_performed': False,
        'originals_modified': False, 'SBI_training_readiness': False,
        'scientific_acceptance': False, 'cross_platform_scope': plan['cross_platform_scope'],
    })
    result = validate_tree(out)
    archive = out.parent / (out.name + '.tar.gz')
    if archive.exists():
        raise FileExistsError('immutable archive identity required')
    with tarfile.open(archive, 'w:gz') as tar:
        for path in sorted(out.rglob('*')):
            if path.is_file():
                tar.add(path, arcname=str(path.relative_to(out)), recursive=False)
    receipt = {**result, 'archive': str(archive), 'archive_sha256': digest(archive),
               'archive_bytes': archive.stat().st_size, 'manifest_sha256': digest(out / 'package-manifest.json')}
    write(out.parent / (out.name + '-packaging-receipt.json'), receipt)
    return receipt


def validate_upload(archive, expected_sha, destination):
    if digest(archive) != expected_sha:
        raise ValueError('transport archive SHA differs; nothing extracted')
    if destination.exists():
        raise FileExistsError('new immutable private upload directory required')
    destination.mkdir(parents=True, mode=0o700)
    with tarfile.open(archive) as tar:
        members = tar.getmembers()
        names = [str(Path(item.name)) for item in members]
        if len(names) != len(set(names)):
            raise ValueError('duplicate archive member paths')
        if any(not item.isfile() or item.name.startswith('/') or '..' in Path(item.name).parts for item in members):
            raise ValueError('unsafe upload archive')
        tar.extractall(destination, filter='data')
    result = validate_tree(destination)
    if digest(destination / 'tools/package_rough_local_repairs.py') != digest(__file__):
        raise ValueError('executed validator differs from packaged validator')
    receipt = {'format': 'rough_local_repair_oscar_upload_receipt_v1',
               'created_utc': datetime.now(timezone.utc).isoformat(), **result,
               'archive_sha256': expected_sha, 'archive_bytes': archive.stat().st_size,
               'manifest_sha256': digest(destination / 'package-manifest.json'),
               'validator_sha256': digest(__file__), 'private_destination': str(destination),
               'merge_action_performed': False, 'original_production_payloads_modified': False}
    write(destination.parent / (destination.name + '-upload-receipt.json'), receipt)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='role', required=True)
    pack = sub.add_parser('package')
    for name in ['repo', 'run', 'out', 'audit', 'audit-checker']:
        pack.add_argument('--' + name, type=Path, required=True)
    verify = sub.add_parser('validate-upload')
    verify.add_argument('--archive', type=Path, required=True)
    verify.add_argument('--archive-sha', required=True)
    verify.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    result = package(args.repo, args.run, args.out, args.audit, args.audit_checker) if args.role == 'package' else validate_upload(args.archive, args.archive_sha, args.destination)
    print(json.dumps(result), flush=True)
