"""Internal, evidence-gated F4 cleanup; --check is strictly read-only."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse,hashlib,json,os,datetime

def main():
    p=argparse.ArgumentParser(description=__doc__)
    m=p.add_mutually_exclusive_group(required=True);m.add_argument('--check',action='store_true');m.add_argument('--apply',action='store_true')
    p.add_argument('--source',type=Path,required=True);p.add_argument('--sidecar',type=Path,required=True)
    p.add_argument('--invocation',required=True);p.add_argument('--plate-id',required=True)
    p.add_argument('--expected-sha256');p.add_argument('--expected-revision',type=int);p.add_argument('--backup',type=Path)
    a=p.parse_args()
    from auto_annotation_tool.ranking.mobile_human_review import MobileReviewSession
    from auto_annotation_tool.ranking import mobile_reference_reconciliation as repair
    s=MobileReviewSession.open(a.source,sidecar_path=a.sidecar)
    target=repair.prepare(s,a.invocation,a.plate_id)
    if a.check:print(json.dumps(target,ensure_ascii=False));return
    if not a.expected_sha256 or a.expected_revision is None or not a.backup:p.error('--apply requires --expected-sha256, --expected-revision and --backup')
    expected=dict(review_revision=a.expected_revision,sidecar_sha256=a.expected_sha256)
    current={k:target[k] for k in expected}
    if expected!=current and not (target['status']=='NOOP_ALREADY_RECONCILED' and target.get('original_precondition')==expected):
        raise ValueError('CAS_PRECONDITION_FAILED')
    if target['status']=='READY':
        raw=a.sidecar.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=a.expected_sha256:raise ValueError('CAS_PRECONDITION_FAILED')
        if a.backup.exists():
            if a.backup.read_bytes()!=raw:raise ValueError('BACKUP_ALREADY_EXISTS_WITH_DIFFERENT_DATA')
        else:
            a.backup.parent.mkdir(parents=True,exist_ok=True)
            with a.backup.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
        metadata=a.backup.with_name(a.backup.name+'.manifest.json')
        if not metadata.exists():
            info=dict(target,schema='alpr.mobile.reconciliation_backup.v1',created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                sidecar_path=str(a.sidecar.resolve()),backup_path=str(a.backup.resolve()),source_path=str(a.source.resolve()),
                size=len(raw),mtime_ns=a.sidecar.stat().st_mtime_ns)
            with metadata.open('x',encoding='utf-8') as f:json.dump(info,f,ensure_ascii=False,indent=2);f.flush();os.fsync(f.fileno())
    result=repair.apply(s,target,backup_path=a.backup)
    print(json.dumps(dict(result=result,target=target),ensure_ascii=False))

if __name__=='__main__':main()
