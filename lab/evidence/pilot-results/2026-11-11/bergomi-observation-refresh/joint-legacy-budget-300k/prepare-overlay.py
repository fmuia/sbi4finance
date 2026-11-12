"""Create a content-addressed overlay of the reviewed new producer and plan."""
from pathlib import Path
import hashlib
import io
import json
import sys
import tarfile

root = Path(__file__).resolve().parents[6]
evidence = Path(__file__).resolve().parent
plan = Path(sys.argv[1]).resolve()
review = evidence.parent / 'authorized-pilot/intentional-stop.json'
producer = root / 'lab/src/sbi4finance/calibration/rough_actual_layout_overnight_bank.py'
cli = root / 'lab/scripts/rough_actual_layout_overnight_bank.py'
members = {
    'repo/lab/src/sbi4finance/calibration/rough_actual_layout_overnight_bank.py': producer,
    'repo/lab/scripts/rough_actual_layout_overnight_bank.py': cli,
    'preparation/plan.json': plan,
    'preparation/pilot-readout.json': review,
}
target = Path('/private/tmp/rough-main-full-20261111.tar')
with tarfile.open(target, 'w') as archive:
    for name, source in members.items():
        content = source.read_bytes()
        header = tarfile.TarInfo(name)
        header.size = len(content)
        header.mode = 0o600
        header.mtime = 0
        archive.addfile(header, io.BytesIO(content))
digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
record = {
    'path': str(target), 'sha256': digest(target), 'bytes': target.stat().st_size,
    'plan': {'path': str(plan), 'sha256': digest(plan)},
    'producer': {'path': str(producer), 'sha256': digest(producer)},
    'CLI': {'path': str(cli), 'sha256': digest(cli)},
    'review': {'path': str(review), 'sha256': digest(review)},
    'members': list(members),
}
with (evidence / 'overlay-identity.json').open('x') as stream:
    json.dump(record, stream, indent=2)
    stream.write('\n')
print(json.dumps(record))
