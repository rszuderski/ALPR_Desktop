"""Check a frozen MZ comparison protocol; --train is an explicit later step."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from auto_annotation_tool.training.mz_experiment_runtime import (
    build_mz_training_request, check_mz_experiment_protocol, file_sha256,
)


def check_protocol(protocol, *, repo_root=None):
    return check_mz_experiment_protocol(protocol, repo_root=repo_root)


def main(argv=None):
    parser=argparse.ArgumentParser(description="Read-only E-MZ-NS-01 preflight; --train requires separate approval of the prepared report.")
    parser.add_argument("--protocol",required=True)
    parser.add_argument("--variant",choices=["MZ-n","MZ-s"],default="MZ-n")
    parser.add_argument("--train",action="store_true",help="Start the selected full trial only after report acceptance")
    args=parser.parse_args(argv)
    protocol=json.loads(Path(args.protocol).read_text(encoding="utf-8"))
    result=check_protocol(protocol)
    print(json.dumps(result),flush=True)
    if not args.train:
        print("Read-only protocol check passed. Training was not started.")
        return 0
    from auto_annotation_tool.campaign_manager import CAMPAIGN
    from auto_annotation_tool.training.training_history import TrainingHistory
    from auto_annotation_tool.training.trainer import YOLOPoseTrainer
    state=json.loads(CAMPAIGN.state_file.read_text(encoding="utf-8-sig"))
    if state.get("active_project")!=protocol["project"] or state["projects"][protocol["project"]]["current_iteration"]!=protocol["iteration"]:
        raise ValueError("Active project/iteration differs from the experiment protocol")
    CAMPAIGN.state=state
    history=TrainingHistory(Path(protocol["history_dir"]))
    trainer=YOLOPoseTrainer(history)
    trainer.on_progress=lambda percent,text:print(f"{percent:.1f}% {text}",flush=True)
    run_id=trainer.start_training(**build_mz_training_request(protocol, args.variant))
    if not run_id:
        raise RuntimeError("The native trainer rejected experiment preflight")
    print("Native trial run_id: "+str(run_id),flush=True)
    while trainer.is_training:
        time.sleep(.25)
    run=trainer.history.get_run(run_id)
    if not run or run.status!="completed" or not run.training_protocol_snapshot.get("comparable"):
        raise RuntimeError("Trial failed or is not experimentally comparable; do not compare it to the other scale")
    print(json.dumps({"run_id":run_id,"status":run.status,"protocol":run.training_protocol_snapshot}),flush=True)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
