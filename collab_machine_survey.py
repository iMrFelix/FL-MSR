#!/usr/bin/env python3
"""Machine survey, v2.1 (second review round) — step 1 of the roadmap.

Read-only survey of the experiment machine. Runs in ~1 minute, changes
nothing, needs no sudo. Python 3.8+ standard library only.

    python3 collab_machine_survey.py [--expected-gpus 4]
        [--expected-gpu-model A100] [--min-free-gib 200]
        [--data-path PATH] [--out machine_survey.json]

Writes machine_survey.json (send it back to us). Exit 0 means the report
was WRITTEN; individual checks may be pass/warn/fail/unknown/info inside
it. Guarantees, per review: a failed query is UNKNOWN with its error
preserved, never a silent pass; every section runs inside a guard so no
probe failure can prevent the report; every external command has a
timeout and runs under LC_ALL=C; the JSON is written atomically; no
containers are launched (functional GPU/shaping verification belongs to
the step-2 smoke test).
"""
import argparse
import csv
import io
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import tempfile
import urllib.request
from datetime import datetime, timezone

SCHEMA = "2.1"
CHECKS: dict = {}
RAW: dict = {}


def run(cmd, timeout=15):
    """(rc, stdout, stderr) with timeout; rc=None if unrunnable/timed out."""
    env = dict(os.environ, LC_ALL="C", LANG="C")
    try:
        p = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=timeout, env=env)
        return p.returncode, p.stdout.strip(), p.stderr.strip()
    except FileNotFoundError:
        return None, "", f"{cmd[0]}: not found"
    except subprocess.TimeoutExpired:
        return None, "", f"{cmd[0]}: timed out after {timeout}s"
    except Exception as e:  # noqa: BLE001 — a probe must never crash the survey
        return None, "", f"{cmd[0]}: {e}"


def record(name, status, value=None, error=None, note=None):
    CHECKS[name] = {"status": status, "value": value}
    if error:
        CHECKS[name]["error"] = str(error)[:500]
    if note:
        CHECKS[name]["note"] = note
    tag = {"pass": "[PASS]", "warn": "[WARN]", "fail": "[FAIL]",
           "unknown": "[UNKNOWN]", "info": "[info]"}[status]
    val = value if isinstance(value, str) else json.dumps(value)
    if val and len(val) > 60:
        val = val[:57] + "..."
    print(f"  {name:32s} {tag:9s} {val or ''}" + (f"  ({note})" if note else ""))


def disk(name, path, threshold_gib=None):
    try:
        st = os.statvfs(path)
        free_bytes = st.f_frsize * st.f_bavail
    except OSError as e:
        record(name, "unknown", value={"path": path}, error=e)
        return
    rc, fs, fserr = run(["findmnt", "-n", "-o", "FSTYPE", "-T", path])
    val = {"path": path, "free_gib": round(free_bytes / 2**30, 1),
           "fstype": fs if rc == 0 and fs else None}
    err = None if rc == 0 else (fserr or "findmnt unavailable")
    if threshold_gib is not None:
        ok = free_bytes >= threshold_gib * 2**30   # compare bytes, not rounded
        record(name, "pass" if ok else "warn", val, error=err,
               note=None if ok else f"campaigns need ~{threshold_gib} GiB")
    else:
        record(name, "info", val, error=err)


def sec_host(args):
    try:
        osr = open("/etc/os-release").read()
        RAW["os_release"] = osr
        pretty = [l for l in osr.splitlines() if l.startswith("PRETTY_NAME=")]
        record("os", "info", pretty[0].split("=", 1)[1].strip('"') if pretty
               else platform.platform())
    except OSError:
        record("os", "info", platform.platform())
    record("kernel", "info", platform.release())
    rc, out, err = run(["lscpu"])
    if rc == 0:
        RAW["lscpu"] = out
        summary = {k.strip(): v.strip() for k, v in
                   (l.split(":", 1) for l in out.splitlines() if ":" in l)}
        record("cpu", "info", {k: summary.get(k) for k in
               ("Model name", "Socket(s)", "Core(s) per socket",
                "Thread(s) per core", "CPU(s)", "NUMA node(s)")})
    else:
        record("cpu", "unknown", error=err, note="lscpu unavailable")
    try:
        aff = sorted(os.sched_getaffinity(0))
        record("cpu_affinity", "info",
               {"count": len(aff), "cpus": aff if len(aff) <= 64 else
                f"{aff[0]}-{aff[-1]} ({len(aff)} cpus)"})
    except AttributeError:
        record("cpu_affinity", "unknown", note="not on this platform")
    try:
        mems = [l.split(":")[1].strip() for l in open("/proc/self/status")
                if l.startswith("Mems_allowed_list")]
        record("mem_nodes_allowed", "info", mems[0] if mems else None)
    except OSError:
        record("mem_nodes_allowed", "unknown", note="/proc unavailable")
    try:
        mem = [l for l in open("/proc/meminfo") if l.startswith("MemTotal")][0]
        record("ram_gib", "info", round(int(mem.split()[1]) / 1048576))
    except (OSError, IndexError):
        record("ram_gib", "unknown", note="/proc/meminfo unavailable")
    rc, out, err = run(["timedatectl", "show", "-p", "NTPSynchronized"])
    if rc == 0:
        synced = "yes" in out
        record("clock_ntp_synced", "pass" if synced else "warn", out,
               note=None if synced else "timing experiments prefer synced clocks")
    else:
        record("clock_ntp_synced", "unknown", error=err)


def sec_storage(args):
    disk("data_path_free", args.data_path, threshold_gib=args.min_free_gib)
    disk("tmp_free", tempfile.gettempdir())


GPU_FIELDS = ("index,uuid,pci.bus_id,name,memory.total,driver_version,"
              "persistence_mode,ecc.mode.current,mig.mode.current")


def sec_gpus(args):
    rc, out, err = run(["nvidia-smi", f"--query-gpu={GPU_FIELDS}",
                        "--format=csv,noheader"], timeout=20)
    if rc != 0 or not out:
        record("gpu_inventory", "fail" if rc is not None else "unknown",
               error=err or "nvidia-smi returned no output")
        return
    names = GPU_FIELDS.split(",")
    gpus, malformed = [], []
    for row in csv.reader(io.StringIO(out)):
        vals = [c.strip() for c in row]
        if len(vals) == len(names):
            gpus.append(dict(zip(names, vals)))
        elif any(vals):
            malformed.append(",".join(vals))
    record("gpu_inventory", "pass" if gpus and not malformed else
           ("warn" if gpus else "unknown"), gpus,
           error="; ".join(malformed) if malformed else None,
           note=f"{len(malformed)} malformed row(s) skipped" if malformed else None)
    if not gpus:
        return
    n = len(gpus)
    record("gpu_count", "pass" if n == args.expected_gpus else "warn",
           n, note=f"expected {args.expected_gpus}")
    models_ok = all(args.expected_gpu_model in g["name"] for g in gpus)
    record("gpu_model", "pass" if models_ok else "warn",
           sorted({g["name"] for g in gpus}),
           note=f"expected {args.expected_gpu_model}")
    # MIG: only an explicit 'Disabled' on every GPU establishes flat GPUs;
    # 'Enabled' anywhere is a warn; anything else ([N/A], unexpected) is unknown.
    mig = {g["index"]: g["mig.mode.current"] for g in gpus}
    states = {v.strip().lower() for v in mig.values()}
    if states <= {"disabled"}:
        record("mig_mode", "pass", mig, note="all flat (explicitly Disabled)")
    elif any("enabled" in s for s in states):
        record("mig_mode", "warn", mig,
               note="MIG enabled on some GPUs — changes our client-to-GPU "
                    "pinning; tell us")
    else:
        record("mig_mode", "unknown", mig,
               note="MIG state not reported as Enabled/Disabled — cannot "
                    "confirm flat GPUs")
    rc2, out2, err2 = run(["nvidia-smi", "--query-compute-apps=pid,process_name",
                           "--format=csv,noheader"], timeout=20)
    if rc2 == 0:
        nproc = len([l for l in out2.splitlines() if l.strip()])
        record("gpu_processes_now", "pass" if nproc == 0 else "warn", nproc,
               note="snapshot only — 0 now does not prove exclusivity; "
                    "we monitor during runs" if nproc == 0
                    else "machine appears shared")
    else:
        record("gpu_processes_now", "unknown", error=err2 or "query failed")
    rc3, out3, err3 = run(["nvidia-smi", "topo", "-m"], timeout=20)
    if rc3 == 0:
        RAW["gpu_topology"] = out3
        record("gpu_topology", "info", "captured",
               note="our clients are single-GPU processes; recorded for "
                    "contention diagnosis, not a requirement")
    else:
        record("gpu_topology", "unknown", error=err3)
    rc4, out4, err4 = run(["nvidia-smi", "-q"], timeout=25)
    if rc4 == 0 and out4:
        RAW["nvidia_smi_q"] = out4
        record("gpu_full_query", "pass", f"captured ({len(out4)} bytes)")
        cuda = [l for l in out4.splitlines() if "CUDA Version" in l]
        record("cuda_driver_reported", "info" if cuda else "unknown",
               cuda[0].split(":")[-1].strip() if cuda else None,
               note="driver ceiling, not the toolkit; our framework's "
                    "runtime ships inside the containers")
    else:
        record("gpu_full_query", "unknown", error=err4 or "no output")
        record("cuda_driver_reported", "unknown",
               error="full query unavailable")
    rc5, out5, err5 = run(["nvcc", "--version"])
    record("cuda_toolkit_host", "info" if rc5 == 0 else "unknown",
           out5.splitlines()[-1] if rc5 == 0 and out5 else None,
           error=None if rc5 == 0 else err5,
           note="host toolkit; informational only")


def sec_containers(args):
    rc, out, err = run(["docker", "--version"])
    if rc != 0:
        record("docker_version", "fail", error=err)
        return
    record("docker_version", "pass", out)
    # Which daemon are we even talking to? A remote context would make
    # DockerRootDir and CDI paths refer to a DIFFERENT machine.
    rc_c, ctx, _ = run(["docker", "context", "show"])
    rc_e, endpoint, _ = run(["docker", "context", "inspect", "--format",
                             "{{.Endpoints.docker.Host}}"])
    env_host = os.environ.get("DOCKER_HOST", "")
    endpoint = env_host or (endpoint if rc_e == 0 else "")
    local = endpoint.startswith("unix://") or endpoint == ""
    record("docker_endpoint", "pass" if local else "warn",
           {"context": ctx if rc_c == 0 else None, "endpoint": endpoint or None,
            "DOCKER_HOST": env_host or None},
           note=None if local else "endpoint may be a REMOTE daemon — "
                "host-local path and CDI checks below do not apply to it")
    rc2, out2, err2 = run(["docker", "info", "--format",
                           '{"runtimes":{{json .Runtimes}},'
                           '"cgroup":{{json .CgroupVersion}},'
                           '"root":{{json .DockerRootDir}},'
                           '"storage":{{json .Driver}}}'], timeout=20)
    if rc2 != 0:
        record("docker_daemon", "fail", error=err2 or "docker info failed",
               note="group membership / rootless?")
        return
    try:
        info = json.loads(out2)
    except json.JSONDecodeError:
        info = {}
    record("docker_daemon", "pass", "reachable")
    record("docker_cgroup", "info", info.get("cgroup"))
    record("docker_storage", "info",
           {"driver": info.get("storage"), "root": info.get("root")})
    if info.get("root"):
        if local:
            disk("docker_root_free", info["root"])
        else:
            record("docker_root_free", "unknown", {"path": info["root"]},
                   note="remote daemon — path is not on this machine")
    has_nvidia_rt = "nvidia" in (info.get("runtimes") or {})
    cdi, cdi_err = [], []
    for d in ("/etc/cdi", "/var/run/cdi"):
        try:
            if os.path.isdir(d) and any("nvidia" in f for f in os.listdir(d)):
                cdi.append(d)
        except OSError as e:
            cdi_err.append(f"{d}: {e}")
    cdi_note = ("host-local check" + ("" if local else
                " — does NOT establish the remote daemon's configuration"))
    if has_nvidia_rt or cdi:
        record("gpu_container_path", "pass",
               {"nvidia_runtime": has_nvidia_rt, "cdi_specs": cdi},
               error="; ".join(cdi_err) if cdi_err else None,
               note="configuration observed (" + cdi_note + "); functional "
                    "check happens in the step-2 smoke test")
    elif cdi_err:
        record("gpu_container_path", "unknown",
               {"nvidia_runtime": False}, error="; ".join(cdi_err),
               note="CDI directories unreadable; " + cdi_note)
    else:
        record("gpu_container_path", "warn",
               {"nvidia_runtime": False, "cdi_specs": []},
               note="neither nvidia runtime nor CDI specs found — install "
                    "nvidia-container-toolkit (or tell us your CDI setup); "
                    + cdi_note)
    rc3, out3, err3 = run(["docker", "compose", "version"])
    record("compose_v2", "pass" if rc3 == 0 else "fail",
           out3 if rc3 == 0 else None, error=None if rc3 == 0 else err3)
    record("container_probes", "info", "not run",
           note="by design: no containers are launched by this survey; "
                "NET_ADMIN + GPU access are verified functionally by the "
                "step-2 smoke test with our own pinned image")


def sec_netem(args):
    record("tc_present", "pass" if shutil.which("tc") else "fail",
           shutil.which("tc"))
    for m in ("sch_htb", "sch_netem"):
        rc, _, err = run(["modprobe", "-n", m])
        loaded = os.path.isdir(f"/sys/module/{m}")
        if loaded or rc == 0:
            record(f"module_{m}", "pass", "loaded" if loaded else "loadable",
                   note="availability, not permission — verified in smoke test")
        elif rc is None:
            record(f"module_{m}", "unknown", error=err)
        else:
            record(f"module_{m}", "warn", error=err, note="shaping may fail")


def sec_python(args):
    record("python3", "pass", sys.version.split()[0])
    try:
        import venv  # noqa: F401
        with tempfile.TemporaryDirectory() as td:
            rc, _, err = run([sys.executable, "-m", "venv",
                              os.path.join(td, "v")], timeout=120)
        record("venv_creation", "pass" if rc == 0 else "fail",
               error=None if rc == 0 else err,
               note=None if rc == 0 else "apt install python3-venv")
    except ImportError:
        record("venv_creation", "fail", error="venv module missing")
    except OSError as e:
        record("venv_creation", "unknown", error=e,
               note="could not create a temporary directory")
    for name, url in [("pypi_reachable", "https://pypi.org/simple/"),
                      ("cifar_url_reachable",
                       "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz")]:
        try:
            req = urllib.request.Request(url, method="HEAD")
            urllib.request.urlopen(req, timeout=8)
            record(name, "pass", "yes")
        except Exception as e:  # noqa: BLE001
            record(name, "warn", "no", error=e,
                   note="offline path needed (we ship wheels/dataset)")


SECTIONS = [("host", sec_host), ("storage", sec_storage), ("gpus", sec_gpus),
            ("containers", sec_containers),
            ("network emulation (configuration observations)", sec_netem),
            ("python & connectivity", sec_python)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--expected-gpus", type=int, default=4)
    ap.add_argument("--expected-gpu-model", default="A100")
    ap.add_argument("--min-free-gib", type=int, default=200)
    ap.add_argument("--data-path", default=".",
                    help="where campaign data will live (free-space check)")
    ap.add_argument("--out", default="machine_survey.json")
    args = ap.parse_args()

    print(f"=== machine survey v{SCHEMA} "
          f"{datetime.now(timezone.utc).strftime('%F %T UTC')} ===")
    for title, fn in SECTIONS:
        print(f"--- {title}")
        try:
            fn(args)
        except Exception as e:  # noqa: BLE001 — the report must always be written
            record(f"section_{title.split()[0]}_error", "unknown", error=e,
                   note="section aborted; remaining checks in it were skipped")

    doc = {"schema_version": SCHEMA,
           "generated_utc": datetime.now(timezone.utc).isoformat(),
           "hostname": socket.gethostname(),
           "argv": vars(args),
           "checks": CHECKS,
           "raw": RAW}
    try:
        tmp_out = args.out + ".tmp"
        with open(tmp_out, "w") as f:
            json.dump(doc, f, indent=1)
        os.replace(tmp_out, args.out)          # atomic: never a torn report
    except OSError as e:
        print(f"\nERROR: could not write {args.out}: {e}")
        return 1
    counts = {}
    for c in CHECKS.values():
        counts[c["status"]] = counts.get(c["status"], 0) + 1
    print("\nsummary: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print(f"survey complete — written: {args.out} (please send this file back)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
