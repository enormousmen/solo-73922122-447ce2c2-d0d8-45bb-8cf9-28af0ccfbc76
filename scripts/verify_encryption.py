#!/usr/bin/env python3
"""End-to-end verification of AES-256-GCM checkpoint encryption.

Spins up real uvicorn server processes (same harness style as
``verify_repro.py``) and checks:

  I1  service starts with a Base64 32-byte key; /config reports encryption on
  I2  checkpoints on disk and their temp files contain ONLY ciphertext
      (RPCKENC1 frame; no torch ZIP / plaintext state on disk at any time)
  I3  every successful save uses a fresh nonce (frames differ, same point)
  I4  pause -> resume under encryption reproduces an uninterrupted encrypted
      baseline bit-exactly (losses + final parameters)
  I5  hard crash + restart (same key) recovers and still reproduces baseline
  I6  restart WITHOUT key on an encrypted point: start/resume refused (409),
      parameters refused (409), file not overwritten
  I7  restart with the WRONG key: same refusals, GCM auth error surfaced
  I8  restart with the CORRECT key after I6/I7: full recovery works again
  I9  encryption enabled but a PLAINTEXT checkpoint exists: load and start
      refused (409, even with force=true), file not overwritten
  I10 encryption disabled on a plaintext directory: legacy point still works
  I11 encrypted checkpoint download stays ciphertext; parameters endpoint
      decrypts in memory and returns a loadable plaintext state_dict
  I12 malformed key env (bad base64 / wrong length) stops the process at
      startup (fail fast, no silent plaintext fallback)
  I13 save failure under encryption keeps the previous encrypted point

Exit code 0 iff every scenario passes.  Run from the repo root:

    python3 scripts/verify_encryption.py
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable

CFG = {"epochs": 4, "batch_size": 32, "batch_sleep_seconds": 0.02,
       "checkpoint_every": 2, "checkpoint_sleep_seconds": 0.0}
MAGIC = b"RPCKENC1"

RESULTS: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, note: str = "") -> None:
    RESULTS.append((name, ok, note))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" - {note}" if note else ""), flush=True)


def api(port: int, method: str, path: str, body: dict | None = None,
        timeout: float = 120.0) -> tuple[int, dict]:
    url = f"http://127.0.0.1:{port}{path}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            return resp.status, (json.loads(raw) if raw else {})
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        try:
            return exc.code, json.loads(raw)
        except json.JSONDecodeError:
            return exc.code, {"detail": raw.decode(errors="replace")}


def api_bytes(port: int, path: str) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=60) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Server:
    def __init__(self, runs_dir: Path, log_dir: Path, tag: str,
                 key_b64: str | None = None, expect_fail: bool = False,
                 extra_env: dict | None = None):
        self.port = _free_port()
        self.tag = tag
        log = open(log_dir / f"server-{tag}.log", "wb")
        env = dict(os.environ)
        env["RUNS_DIR"] = str(runs_dir)
        env["CONFIG_PATH"] = str(ROOT / "config" / "default.json")
        env["ENABLE_TEST_CRASH_ENDPOINT"] = "1"
        if key_b64 is None:
            env.pop("CHECKPOINT_ENCRYPTION_KEY", None)
        else:
            env["CHECKPOINT_ENCRYPTION_KEY"] = key_b64
        if extra_env:
            env.update(extra_env)
        self.proc = subprocess.Popen(
            [PY, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1",
             "--port", str(self.port)],
            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        self._log_path = log_dir / f"server-{tag}.log"
        if expect_fail:
            self.proc.wait(timeout=30)
            self.exit_code = self.proc.returncode
        else:
            self._wait_ready()
            self.exit_code = None

    def _wait_ready(self) -> None:
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(
                    f"server {self.tag} exited early; log:\n"
                    + self._log_path.read_text(errors="replace")[-3000:])
            try:
                code, _ = api(self.port, "GET", "/train/status", timeout=2)
                if code == 200:
                    return
            except (urllib.error.URLError, ConnectionError, TimeoutError):
                pass
            time.sleep(0.3)
        raise RuntimeError(f"server {self.tag} did not become ready")

    def kill(self) -> None:
        try:
            urllib.request.urlopen(
                f"http://127.0.0.1:{self.port}/test/crash", data=b"", timeout=5)
        except Exception:
            pass
        self.proc.wait(timeout=10)

    def stop(self) -> None:
        if self.proc.poll() is None:
            self.kill()

    def log_tail(self) -> str:
        return self._log_path.read_text(errors="replace")[-1500:]


def wait_state(port: int, want: str, timeout: float = 180.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        code, st = api(port, "GET", "/train/status")
        if code == 200 and st["state"] == want:
            return st
        time.sleep(0.2)
    raise RuntimeError(f"state did not become {want!r} within {timeout}s")


def wait_until(port: int, pred, timeout: float = 60.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        code, st = api(port, "GET", "/train/status")
        if code == 200 and pred(st):
            return st
        time.sleep(0.05)
    raise RuntimeError("condition not met within timeout")


def start_run(port: int, cfg: dict) -> None:
    code, st = api(port, "POST", "/train/start", {"force": True, "config": cfg})
    assert code == 200, f"start failed: {st}"


def run_to_finish(port: int, cfg: dict) -> tuple[list, bytes]:
    start_run(port, cfg)
    wait_state(port, "finished")
    _, losses = api(port, "GET", "/train/losses")
    code, params = api_bytes(port, "/train/parameters")
    assert code == 200
    return losses["losses"], params


def compare(name: str, ref_losses, ref_params, got_losses, got_params) -> bool:
    ok = got_losses == ref_losses
    if not ok:
        print(f"    losses MISMATCH: {len(got_losses)} vs {len(ref_losses)}")
    ref_sd = torch.load(io.BytesIO(ref_params), weights_only=True)
    got_sd = torch.load(io.BytesIO(got_params), weights_only=True)
    params_ok = all(torch.equal(got_sd[k], ref_sd[k]) for k in ref_sd)
    record(name, ok and params_ok,
           f"losses exact={ok}, params bit-identical={params_ok}")
    return ok and params_ok


def main() -> int:
    work = Path(tempfile.mkdtemp(prefix="verify-enc-"))
    runs_dir = work / "runs"
    log_dir = work / "logs"
    log_dir.mkdir()
    print(f"workdir: {work}")
    key = os.urandom(32)
    key_b64 = base64.b64encode(key).decode()
    wrong_key_b64 = base64.b64encode(os.urandom(32)).decode()
    ckpt = runs_dir / "checkpoint.pt"

    # ------------------------------------------------------- I12 malformed key
    print("\n== I12: malformed key fails fast at startup ==")
    for bad, label in [("!!!not-base64!!!", "bad base64"),
                       (base64.b64encode(b"only-16-bytes!!").decode(), "16 bytes")]:
        s = Server(runs_dir, log_dir, f"I12-{label}", key_b64=bad, expect_fail=True)
        log = s.log_tail()
        ok = s.exit_code != 0 and "CHECKPOINT_ENCRYPTION_KEY" in log
        record(f"I12 startup refuses {label}", ok, f"exit={s.exit_code}")

    # ------------------------------------------------- I1/I2/I3 encrypted run
    print("\n== I1-I3: encrypted service basics and on-disk guarantees ==")
    srv = Server(runs_dir, log_dir, "enc-main", key_b64=key_b64)
    code, cfg = api(srv.port, "GET", "/config")
    enc_info = cfg.get("checkpoint_encryption", {})
    record("I1 /config reports AES-256-GCM enabled",
           code == 200 and enc_info.get("enabled") is True
           and enc_info.get("algorithm") == "AES-256-GCM"
           and enc_info.get("key_env_var") == "CHECKPOINT_ENCRYPTION_KEY",
           json.dumps(enc_info))

    start_run(srv.port, CFG)
    wait_until(srv.port, lambda s: s["last_checkpoint_at"] is not None)
    st = wait_until(srv.port, lambda s: s["checkpoint_encrypted"] is True)
    record("I1 status shows checkpoint_encrypted=true",
           st["encryption_enabled"] is True and st["checkpoint_encrypted"] is True)

    # No plaintext on disk: final file and any tmp file must carry the frame.
    disk_blob = ckpt.read_bytes()
    record("I2 checkpoint.pt is an RPCKENC1 ciphertext frame",
           disk_blob[:8] == MAGIC and not disk_blob[:2] == b"PK")
    tmp_files = list(runs_dir.glob(".checkpoint.pt.tmp.*"))
    record("I2 no leftover plaintext tmp files",
           all(p.read_bytes()[:8] == MAGIC for p in tmp_files),
           f"tmp files: {[p.name for p in tmp_files]}")
    # Plaintext torch serialization always embeds tensor dtype keys like
    # 'torch' / 'storage' in the ZIP; assert none of those strings leak.
    leak = b"torch" in disk_blob or b"state_dict" in disk_blob
    record("I2 no plaintext torch markers inside ciphertext", not leak)

    # Fresh nonce per save: snapshot two successive encrypted frames.
    frame1 = bytes(disk_blob)
    wait_until(srv.port, lambda s: s["completed_batches"] >= 4)
    time.sleep(0.3)
    frame2 = ckpt.read_bytes()
    nonces_diff = frame1[8:20] != frame2[8:20] and frame1 != frame2
    record("I3 each save uses a fresh random nonce", nonces_diff)

    # ------------------------------------------------- I4 pause/resume + I11
    print("\n== I4/I11: pause+resume bit-exact; downloads ==")
    code, st = api(srv.port, "POST", "/train/pause")
    record("I4 pause persists encrypted checkpoint",
           code == 200 and st["state"] == "paused" and not st["checkpoint_dirty"])
    frame_paused = ckpt.read_bytes()
    code, _ = api(srv.port, "POST", "/train/resume")
    record("I4 resume accepted", code == 200)
    wait_state(srv.port, "finished")
    _, enc_losses = api(srv.port, "GET", "/train/losses")
    code, enc_params = api_bytes(srv.port, "/train/parameters")
    record("I11 parameters endpoint decrypts in memory to a .pt state_dict",
           code == 200 and enc_params[:2] == b"PK"
           and set(torch.load(io.BytesIO(enc_params), weights_only=True).keys()) != set())
    code, dl = api_bytes(srv.port, "/train/checkpoint")
    record("I11 checkpoint download stays ciphertext on the wire",
           code == 200 and dl[:8] == MAGIC and dl == ckpt.read_bytes())

    # Uninterrupted encrypted baseline (separate directory, same key/config).
    base_runs = work / "baseline"
    base_srv = Server(base_runs, log_dir, "enc-baseline", key_b64=key_b64)
    ref_losses, ref_params = run_to_finish(base_srv.port, CFG)
    base_srv.stop()
    record("I4 baseline encrypted run finished", True,
           f"{len(ref_losses)} batches, frame={base_runs.joinpath('checkpoint.pt').read_bytes()[:8]!r}")
    compare("I4 encrypted pause+resume == uninterrupted baseline",
            ref_losses, ref_params, enc_losses["losses"], enc_params)

    # --------------------------------------------- I5 crash + same-key restart
    print("\n== I5: hard crash + restart with same key ==")
    crash_runs = work / "crash"
    csrv = Server(crash_runs, log_dir, "I5-a", key_b64=key_b64)
    start_run(csrv.port, CFG)
    wait_until(csrv.port, lambda s: s["completed_batches"] >= 8
               and s["last_checkpoint_at"] is not None)
    csrv.kill()
    csrv2 = Server(crash_runs, log_dir, "I5-b", key_b64=key_b64)
    code, st = api(csrv2.port, "GET", "/train/status")
    record("I5 same-key restart recovers encrypted point",
           code == 200 and st["state"] == "paused" and st["recovered_after_restart"]
           and st["checkpoint_encrypted"] is True and st["recovery_error"] is None,
           f"recovered at batch {st['completed_batches']}/{st['total_batches']}")
    code, _ = api(csrv2.port, "POST", "/train/resume")
    record("I5 resume after restart", code == 200)
    wait_state(csrv2.port, "finished")
    _, losses = api(csrv2.port, "GET", "/train/losses")
    _, params = api_bytes(csrv2.port, "/train/parameters")
    compare("I5 crash+restart under encryption == baseline",
            ref_losses, ref_params, losses["losses"], params)

    # ----------------------------- I6/I7/I8 missing key, wrong key, right key
    print("\n== I6/I7/I8: key mismatch refusals then correct-key recovery ==")
    guarded = crash_runs / "checkpoint.pt"
    good_hash = hashlib.sha256(guarded.read_bytes()).hexdigest()

    srv_nokey = Server(crash_runs, log_dir, "I6-nokey")  # no CHECKPOINT_ENCRYPTION_KEY
    code, st = api(srv_nokey.port, "GET", "/train/status")
    record("I6 no-key restart flags recovery_error and failed state",
           code == 200 and st["state"] == "failed"
           and st["recovery_error"] is not None
           and "no key is set" in st["recovery_error"])
    for path, method in [("/train/start", "POST"), ("/train/resume", "POST")]:
        code, body = api(srv_nokey.port, method, path,
                         {"force": True} if path.endswith("start") else None)
        record(f"I6 {method} {path} refused without key",
               code == 409 and "no key is set" in str(body.get("detail", "")))
    code, body = api_bytes(srv_nokey.port, "/train/parameters")
    record("I6 parameters refused without key", code == 409)
    assert hashlib.sha256(guarded.read_bytes()).hexdigest() == good_hash
    record("I6 encrypted file never overwritten", True)
    srv_nokey.stop()

    srv_wrong = Server(crash_runs, log_dir, "I7-wrong", key_b64=wrong_key_b64)
    code, st = api(srv_wrong.port, "GET", "/train/status")
    record("I7 wrong-key restart surfaces GCM auth failure",
           code == 200 and st["state"] == "failed"
           and "authenticate" in (st["recovery_error"] or ""))
    code, body = api(srv_wrong.port, "POST", "/train/start", {"force": True})
    record("I7 force-start refused with wrong key (no overwrite)",
           code == 409 and "authenticate" in str(body.get("detail", "")))
    code, _ = api(srv_wrong.port, "POST", "/train/resume")
    record("I7 resume refused with wrong key", code == 409)
    assert hashlib.sha256(guarded.read_bytes()).hexdigest() == good_hash
    record("I7 encrypted file never overwritten", True)
    srv_wrong.stop()

    srv_right = Server(crash_runs, log_dir, "I8-right", key_b64=key_b64)
    code, st = api(srv_right.port, "GET", "/train/status")
    # The I5 run was resumed to completion before this check, so the point is
    # a finished one; what matters is that the correct key clears the error.
    record("I8 correct key restores recoverability",
           code == 200 and st["state"] in ("paused", "finished")
           and st["recovery_error"] is None and st["recovered_after_restart"])
    srv_right.stop()

    # ------------------------------- I9 plaintext point + encryption enabled
    print("\n== I9: plaintext checkpoint under enabled encryption ==")
    plain_runs = work / "plain"
    psrv = Server(plain_runs, log_dir, "I9-plain")          # no key -> plaintext
    run_to_finish(psrv.port, CFG)
    psrv.stop()
    plain_blob = (plain_runs / "checkpoint.pt").read_bytes()
    assert plain_blob[:2] == b"PK"
    plain_hash = hashlib.sha256(plain_blob).hexdigest()
    esrv = Server(plain_runs, log_dir, "I9-enc", key_b64=key_b64)
    code, st = api(esrv.port, "GET", "/train/status")
    record("I9 encryption-on sees plaintext point -> failed + recovery_error",
           code == 200 and st["state"] == "failed"
           and "plaintext" in (st["recovery_error"] or ""))
    code, body = api(esrv.port, "POST", "/train/start", {"force": True})
    record("I9 force-start refused; plaintext point not overwritten",
           code == 409 and "plaintext" in str(body.get("detail", ""))
           and hashlib.sha256((plain_runs / "checkpoint.pt").read_bytes()).hexdigest()
           == plain_hash)
    code, _ = api(esrv.port, "POST", "/train/resume")
    record("I9 resume refused on plaintext point", code == 409)
    esrv.stop()

    # Same guarantee for an *unfinished* (paused) plaintext point.
    plain_runs2 = work / "plain-paused"
    p2 = Server(plain_runs2, log_dir, "I9-plain2")
    start_run(p2.port, CFG)
    wait_until(p2.port, lambda s: s["completed_batches"] >= 6
               and s["last_checkpoint_at"] is not None)
    code, st = api(p2.port, "POST", "/train/pause")
    assert code == 200 and st["state"] == "paused"
    p2.stop()
    h2 = hashlib.sha256((plain_runs2 / "checkpoint.pt").read_bytes()).hexdigest()
    e2 = Server(plain_runs2, log_dir, "I9-enc2", key_b64=key_b64)
    code, st = api(e2.port, "GET", "/train/status")
    record("I9 paused plaintext point refused under encryption (no overwrite)",
           code == 200 and st["state"] == "failed"
           and "plaintext" in (st["recovery_error"] or "")
           and hashlib.sha256((plain_runs2 / "checkpoint.pt").read_bytes()).hexdigest() == h2)
    e2.stop()

    # ---------------------------- I10 no encryption on a plaintext directory
    print("\n== I10: legacy plaintext directory still works without key ==")
    lsrv = Server(plain_runs, log_dir, "I10-legacy")
    code, st = api(lsrv.port, "GET", "/train/status")
    record("I10 plaintext point recovered with encryption disabled",
           code == 200 and st["state"] in ("paused", "finished")
           and st["checkpoint_encrypted"] is False and st["recovery_error"] is None)
    code, params = api_bytes(lsrv.port, "/train/parameters")
    record("I10 parameters downloadable from plaintext point", code == 200)
    lsrv.stop()

    # --------------------------- I13 save failure keeps old encrypted point
    print("\n== I13: encrypted save failure preserves previous point ==")
    fail_runs = work / "savefail"
    fsrv = Server(fail_runs, log_dir, "I13", key_b64=key_b64)
    start_run(fsrv.port, CFG)
    wait_until(fsrv.port, lambda s: s["last_checkpoint_at"] is not None)
    flag = fail_runs / "INJECT_SAVE_FAILURES"
    flag.write_text("1")
    time.sleep(0.3)
    before = (fail_runs / "checkpoint.pt").read_bytes()
    h_before = hashlib.sha256(before).hexdigest()
    assert before[:8] == MAGIC
    code, body = api(fsrv.port, "POST", "/train/pause")
    record("I13 pause with failing encrypted save -> 500", code == 500,
           str(body.get("detail", ""))[:80])
    after = (fail_runs / "checkpoint.pt").read_bytes()
    record("I13 previous encrypted point intact (same hash, still ciphertext)",
           hashlib.sha256(after).hexdigest() == h_before and after[:8] == MAGIC)
    flag.unlink()
    code, st = api(fsrv.port, "POST", "/train/pause")
    record("I13 retry succeeds and stays encrypted",
           code == 200 and not st["checkpoint_dirty"]
           and (fail_runs / "checkpoint.pt").read_bytes()[:8] == MAGIC)
    fsrv.stop()

    srv.stop()
    print("\n================ SUMMARY ================")
    failed = 0
    for name, ok, note in RESULTS:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        failed += 0 if ok else 1
    print(f"  {len(RESULTS) - failed}/{len(RESULTS)} checks passed")
    if failed:
        print(f"  server logs under {log_dir}")
    else:
        shutil.rmtree(work, ignore_errors=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
