"""Run the authorized main experiment and preserve its pre-run inputs."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys

ROOT = Path(r"C:\Users\Денис\Desktop\Decision_experiment")
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
from src.experiment.config import load_config
from src.experiment.storage import generate_run_id

cfg = load_config(ROOT / "src/config.yaml")
run_id = generate_run_id((cfg.results.smoke, cfg.results.pilot, cfg.results.main))
out = cfg.results.main / run_id
files = sorted(set([
    ROOT / "src/config.yaml",
    *ROOT.glob("src/**/*.py"),
    *ROOT.glob("prompts/**/*.txt"),
    *ROOT.glob("data/dataset/*.json"),
]))
snapshot = {
    "captured_utc": datetime.now(timezone.utc).isoformat(),
    "capture_stage": "Before the first API call",
    "expected_run_id": run_id,
    "protocol": "25 bouts x 2 models x 2 orientations x 3 collegia x 3 repeats = 900 planned votes",
    "files": {
        p.relative_to(ROOT).as_posix(): {
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            **({"text": p.read_text(encoding="utf-8")} if p.suffix in (".yaml", ".txt", ".py") else {}),
        }
        for p in files
    },
}
local = Path(__file__).with_name(f"main_{run_id}_pre_run_snapshot.json")
local.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Pre-run snapshot saved; expected run_id={run_id}", flush=True)
code = subprocess.call([
    sys.executable, "-u", "-m", "src.run_experiment",
    "--model", "all", "--dataset", "main", "--confirm-main",
])
if out.is_dir():
    (out / "evaluation_snapshot.json").write_text(local.read_text(encoding="utf-8"), encoding="utf-8")
    changed = [name for name, data in snapshot["files"].items()
               if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != data["sha256"]]
    (out / "snapshot_verification.json").write_text(json.dumps({
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "changed_files": changed,
        "all_inputs_unchanged": not changed,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    if changed:
        print("Inputs changed during the run: " + ", ".join(changed), flush=True)
        raise SystemExit(3)
if code:
    raise SystemExit(code)
raise SystemExit(subprocess.call([
    sys.executable, "-u", "-m", "src.calculate_metrics",
    "--dataset", "main", "--run-id", run_id,
]))
