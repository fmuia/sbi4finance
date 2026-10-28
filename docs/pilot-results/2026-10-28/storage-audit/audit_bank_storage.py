"""Read-only NPZ ZIP metadata audit; never loads/decompresses array payloads."""
import ast, json, os, re, statistics, struct, subprocess, sys, time, zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOTS = {
 'known': '/oscar/scratch/afengler/sbi4finance-rough-known-backbone-20261027/collection',
 'joint': '/oscar/scratch/afengler/sbi4finance-native-bank-20261027/collection',
 'heston': '/oscar/scratch/afengler/sbi4finance-full-core-20261026-wy2_j3o4/runs/heston-development/collection',
}
TRAINING = {'observations','parameters','reference_prices','example_id','context_values'}
def header(stream):
    magic = stream.read(8)
    if magic[:6] != b'\x93NUMPY': raise ValueError('Not an NPY member')
    width = 2 if magic[6] == 1 else 4
    length = struct.unpack('<H' if width == 2 else '<I', stream.read(width))[0]
    if length > 4096: raise ValueError('Unexpected long header')
    value = ast.literal_eval(stream.read(length).decode('latin1'))
    return {**value,'header_bytes':8+width+length}

started=time.monotonic(); result={'format':'sbi4finance_storage_audit_v1','utc':datetime.now(timezone.utc).isoformat(),
 'scope':'Read-only completed-shard stat and ZIP/NPY headers; no full arrays, simulation paths or test outcomes read.',
 'banks':{}}
for name, base in ROOTS.items():
    root=Path(base);parts=[];counts={'train':0,'validation':0,'test':0};zipbytes=metadata_bytes=physical=0
    for entry in os.scandir(root/'parts'):
        match=re.fullmatch(r'(train|validation|test)-(\d+)-(\d+)',entry.name)
        if not match or not entry.is_dir(follow_symlinks=False):continue
        path=Path(entry.path);manifest=path/'manifest.json';arrays=path/'arrays.npz'
        if not manifest.is_file() or not arrays.is_file():continue
        stat=arrays.stat();start,stop=map(int,match.group(2,3));n=stop-start
        counts[match[1]]+=n;zipbytes+=stat.st_size;physical+=getattr(stat,'st_blocks',0)*512
        metadata_bytes+=manifest.stat().st_size;parts.append((entry.name,str(arrays),n,stat.st_size))
    parts.sort();size=min(16,len(parts));indices=sorted(set(round(i*(len(parts)-1)/max(1,size-1)) for i in range(size)))
    rows=[];roles={}
    for index in indices:
        label,path,n,compressed=parts[index];members={}
        with zipfile.ZipFile(path) as archive:
            for item in archive.infolist():
                key=item.filename.removesuffix('.npy')
                with archive.open(item) as stream: info=header(stream)
                values={'compressed_bytes':item.compress_size,'uncompressed_npy_bytes':item.file_size,
                        'raw_array_bytes':item.file_size-info['header_bytes'],'shape':list(info['shape']),
                        'dtype':info['descr'],'compressed_bytes_per_example':item.compress_size/n}
                members[key]=values;roles.setdefault(key,[]).append(values)
        rows.append({'part':label,'examples':n,'payload_bytes':compressed,'members':members})
    summary={}
    for role,items in roles.items():
        vals=[x['compressed_bytes_per_example'] for x in items]
        summary[role]={'mean_compressed_bytes_per_example':statistics.mean(vals),
            'median_compressed_bytes_per_example':statistics.median(vals),'min':min(vals),'max':max(vals),
            'raw_bytes_per_example':statistics.mean(x['raw_array_bytes']/rows[i]['examples'] for i,x in enumerate(items)),
            'training_role':role in TRAINING}
    rowcount=sum(counts.values());perexample=zipbytes/rowcount if rowcount else 0
    trainingmean=sum(v['mean_compressed_bytes_per_example'] for v in summary.values() if v['training_role'])
    auxiliarymean=sum(v['mean_compressed_bytes_per_example'] for v in summary.values() if not v['training_role'])
    future=110000 if name=='heston' else 112000
    result['banks'][name]={'root':base,'completed_parts':len(parts),'completed_examples':counts,
       'payload_bytes':zipbytes,'payload_physical_allocated_bytes':physical,'part_manifest_bytes':metadata_bytes,
       'observed_payload_bytes_per_example':perexample,'sampled_parts':rows,'role_summary':summary,
       'sample_training_compressed_bytes_per_example':trainingmean,
       'sample_auxiliary_compressed_bytes_per_example':auxiliarymean,
       'sample_auxiliary_fraction':auxiliarymean/(trainingmean+auxiliarymean),
       'extrapolation':{'training':100000,'fixed_holdouts':future-100000,
          'total_payload_bytes_at_observed_rate':perexample*future,
          'training_roles_only_bytes_at_sample_rate':trainingmean*future,
          'scope':'Extrapolation, not forecast; current numerical target and sample compression rate.'}}
    sys.stderr.write(json.dumps({'bank':name,'examples':rowcount,'payload_bytes':zipbytes,
        'payload_bytes_per_example':perexample,'estimated100k_plus_holdouts_bytes':perexample*future,
        'auxiliary_fraction':auxiliarymean/(trainingmean+auxiliarymean)})+'\n');sys.stderr.flush()
result['seconds']=time.monotonic()-started
print(json.dumps(result,indent=2))
