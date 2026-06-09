import argparse
import json
import os
import sys
import time
import psutil
import shutil
import subprocess
import webbrowser

os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"

ROOT_DIR         = os.path.dirname(os.path.abspath(__file__))
VENV_PYTHON      = os.path.join(ROOT_DIR, ".venv", "Scripts", "python.exe")
DASH_VENV_PYTHON = os.path.join(ROOT_DIR, "master_dashboard", ".venv_dash", "Scripts", "python.exe")
LOGS_DIR         = os.path.join(ROOT_DIR, "logs")

NODE_DIR  = os.path.join(ROOT_DIR, "tools", "node")
NPM_CACHE = os.path.join(ROOT_DIR, "tools", "npm_cache")

if not os.path.exists(VENV_PYTHON):
    print(f"Warning: {VENV_PYTHON} not found. Using system Python.")
    VENV_PYTHON = sys.executable

if not os.path.exists(DASH_VENV_PYTHON):
    print(f"Warning: {DASH_VENV_PYTHON} not found. Dashboard backend may fail to start.")

KAFKA_BIN_DIR = r"C:\kafka\bin\windows"
KAFKA_LOG_DIR = r"C:\tmp\kafka-logs"
ZK_LOG_DIR    = r"C:\tmp\zookeeper-data"

DTC_HISTORY_FILE = os.path.join(ROOT_DIR, "data", "dtc_history.json")
REPLAY_PID_FILE  = os.path.join(ROOT_DIR, ".replay.pid")

KAFKA_TOPICS = [
    "telemetry.battery",
    "telemetry.body",
    "telemetry.engine",
    "telemetry.transmission",
    "telemetry.tyre",
]

os.makedirs(LOGS_DIR, exist_ok=True)

# --- Service Map ---

# Format: "key": ([cmd_list_or_string], is_always_detached, cwd)
SERVICE_MAP = {
    "api_observer":     ([VENV_PYTHON, r"telemetry_observer\api.py"],                          False, ROOT_DIR),
    "api_alerts":       ([VENV_PYTHON, r"alerts_service\api.py"],                              False, ROOT_DIR),
    "api_gold":         ([VENV_PYTHON, r"gold_service\api.py"],                                False, ROOT_DIR),
    "api_inference":    ([VENV_PYTHON, r"inference_service\api.py"],                           False, ROOT_DIR),
    "api_writer":       ([VENV_PYTHON, r"writer_service\api.py"],                              False, ROOT_DIR),
    "api_dtc":          ([VENV_PYTHON, r"dtc_service\api.py"],                                 False, ROOT_DIR),
    "api_analytics":    ([VENV_PYTHON, r"analytics_service\api.py"],                           False, ROOT_DIR),
    "engine_alerts":    ([VENV_PYTHON, r"alerts_service\app.py"],                              False, ROOT_DIR),
    "engine_gold":      ([VENV_PYTHON, r"gold_service\app.py"],                                False, ROOT_DIR),
    "engine_inference": ([VENV_PYTHON, r"inference_service\start_inference_cluster.py"],       False, ROOT_DIR),
    "engine_writer":    ([VENV_PYTHON, r"writer_service\src\start_writer_cluster.py"],         False, ROOT_DIR),
    # Ingest remains detached so its FastAPI logs are always visible in a cmd window
    "ingest": (f"{VENV_PYTHON} -m uvicorn ingest.app.main:app --port 8000 --reload", True, ROOT_DIR),
}

RESET_SCRIPTS = [
    r"tools\reset_alerts_gold.py",
    r"tools\reset_vehicle_health_gold.py",
    r"tools\reset_inference.py",
    r"tools\reset_writer.py",
    r"tools\reset_replay.py",
    r"tools\reset_dashboard_cache.py",  # Integrated RAM wipe
]

# Ports owned by the 7 API services (excludes 8000=ingest, 8005=dashboard backend)
API_PORTS = set(range(8001, 8009)) - {8005}

running_processes = []
open_log_files    = []
HEAD_MODE         = False  # overridden from --head flag before main() is called


# --- Helpers ---

def _make_env():
    env = os.environ.copy()
    env["PYTHONPATH"] = ROOT_DIR
    return env


# --- Execution Primitives ---

def run_background_task(cmd_list, name, cwd_path, wait_time=0):
    log_path = os.path.join(LOGS_DIR, f"{name}.log")
    print(f"--- Starting {name} (Logs: {log_path}) ---")
    log_file = open(log_path, "a", encoding="utf-8")
    open_log_files.append(log_file)
    proc = subprocess.Popen(cmd_list, cwd=cwd_path, stdout=log_file, stderr=subprocess.STDOUT, env=_make_env())
    running_processes.append({"proc": proc, "name": name, "detached": False})
    if wait_time > 0:
        time.sleep(wait_time)


def run_detached_console(cmd_str, name, cwd_path, wait_time=0):
    print(f"--- Starting {name} in a new window ---")
    proc = subprocess.Popen(
        ["cmd.exe", "/k", f"title {name} && {cmd_str}"],
        cwd=cwd_path,
        creationflags=subprocess.CREATE_NEW_CONSOLE,
        env=_make_env(),
    )
    running_processes.append({"proc": proc, "name": name, "detached": True})
    if wait_time > 0:
        time.sleep(wait_time)


def launch_service(cmd, name, cwd, delay=0, always_detach=False):
    """Routes to detached console (head mode / always_detach) or background task (headless)."""
    if always_detach or HEAD_MODE:
        cmd_str = cmd if isinstance(cmd, str) else subprocess.list2cmdline(cmd)
        run_detached_console(cmd_str, name, cwd, delay)
    else:
        cmd_list = cmd if isinstance(cmd, list) else cmd.split()
        run_background_task(cmd_list, name, cwd, delay)


# --- Master Dashboard Launchers ---

def launch_master_backend():
    cmd      = [DASH_VENV_PYTHON, "-m", "uvicorn", "backend.main_v2:app", "--port", "8005"]
    dash_dir = os.path.join(ROOT_DIR, "master_dashboard")
    if HEAD_MODE:
        run_detached_console(subprocess.list2cmdline(cmd), "Master_Dash_Backend", dash_dir)
    else:
        run_background_task(cmd, "Master_Dash_Backend", dash_dir)


def launch_master_frontend():
    print("--- Starting Master Dashboard Frontend (React/Vite) ---")
    frontend_dir = os.path.join(ROOT_DIR, "master_dashboard", "frontend")
    npm_cmd      = os.path.join(NODE_DIR, "npm.cmd")
    node_modules = os.path.join(frontend_dir, "node_modules")

    env = _make_env()
    env["PATH"] = NODE_DIR + os.pathsep + env["PATH"]
    env["npm_config_cache"] = NPM_CACHE

    log_path = os.path.join(LOGS_DIR, "Master_Dash_Frontend.log")
    log_file = open(log_path, "a", encoding="utf-8")
    open_log_files.append(log_file)

    if not os.path.exists(node_modules):
        print("   node_modules not found — running npm install (this takes ~1 min on first run)...")
        result = subprocess.run(
            f'"{npm_cmd}" install',
            shell=True, cwd=frontend_dir,
            stdout=log_file, stderr=subprocess.STDOUT,
            env=env,
        )
        if result.returncode != 0:
            print(f"   ERROR: npm install failed — check {log_path}")
            return
        print("   npm install complete.")

    if HEAD_MODE:
        proc = subprocess.Popen(
            ["cmd.exe", "/k", f'title Master_Dash_Frontend && "{npm_cmd}" run dev -- --port 5173 --strictPort'],
            cwd=frontend_dir,
            creationflags=subprocess.CREATE_NEW_CONSOLE,
            env=env,
        )
        running_processes.append({"proc": proc, "name": "Master_Dash_Frontend", "detached": True})
    else:
        proc = subprocess.Popen(
            f'"{npm_cmd}" run dev -- --port 5173 --strictPort',
            shell=True, cwd=frontend_dir,
            stdout=log_file, stderr=subprocess.STDOUT,
            env=env,
        )
        running_processes.append({"proc": proc, "name": "Master_Dash_Frontend", "detached": False})

    print("   Waiting for Vite to compile (6s)...")
    time.sleep(6)
    print("   Opening browser at http://localhost:5173")
    webbrowser.open("http://localhost:5173")


# --- Process Management ---

def kill_process(p_info):
    """Graceful terminate for background processes; force-kill for detached CMD windows."""
    pid  = p_info["proc"].pid
    name = p_info["name"]

    if p_info.get("detached", False):
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"   ✅ Closed window for {name}.")
    else:
        try:
            parent   = psutil.Process(pid)
            children = parent.children(recursive=True)
            for child in children:
                try: child.terminate()
                except: pass
            _, alive = psutil.wait_procs(children, timeout=3)
            for child in alive:
                try: child.kill()
                except: pass
            parent.terminate()
            try:
                parent.wait(timeout=3)
                print(f"   ✅ Closed {name}.")
            except psutil.TimeoutExpired:
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print(f"   ✅ Force-closed {name}.")
        except Exception:
            pass


def hunt_and_kill_port(port, name):
    for conn in psutil.net_connections(kind="inet"):
        if conn.pid is None:
            continue
        if conn.laddr.port == port and conn.status == "LISTEN":
            try:
                p = psutil.Process(conn.pid)
                p.terminate()
                p.wait(timeout=2)
                print(f"   ✅ Force-closed orphaned {name} on port {port}.")
            except psutil.TimeoutExpired:
                try:
                    p.kill()
                    print(f"   ✅ Force-killed orphaned {name} on port {port}.")
                except Exception:
                    pass
            except Exception:
                pass


def hunt_zombie_replay_workers(auto_kill=False):
    print("   🔍 Scanning OS for orphaned zombie replay workers...")
    zombies = []
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cmdline = p.info["cmdline"]
            if cmdline and "python" in p.info["name"].lower():
                cmd_str = " ".join(cmdline).lower()
                if "replay" in cmd_str and "run.py" not in cmd_str:
                    zombies.append(p)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

    if not zombies:
        print("   ✅ No zombie replay workers found. System is clean.")
        return

    print(f"\n⚠️  WARNING: Detected {len(zombies)} orphaned Replay Worker process(es) still running.")

    if not auto_kill:
        try:
            if input("Do you want to force-kill these zombie workers? (y/n): ").lower() != "y":
                print("   ℹ️ Leaving zombie workers alive.")
                return
        except KeyboardInterrupt:
            print("   ℹ️ Skipping zombie termination due to interrupt.")
            return

    for z in zombies:
        try:
            z.terminate()
            z.wait(timeout=2)
        except psutil.TimeoutExpired:
            z.kill()
        except Exception:
            pass
    print("   ✅ Zombie replay workers eliminated.")


def restart_service(target):
    if target == "master_dashboard":
        for internal_name in ["Master_Dash_Backend", "Master_Dash_Frontend"]:
            for i, p_info in enumerate(running_processes):
                if p_info["name"].lower() == internal_name.lower():
                    print(f"\n[RESTART] Terminating {p_info['name']} (PID: {p_info['proc'].pid})...")
                    kill_process(p_info)
                    running_processes.pop(i)
                    break
        time.sleep(2)
        launch_master_backend()
        time.sleep(4)
        launch_master_frontend()
        print("[RESTART] Master Dashboard is back online.\n")
        return

    if target == "dash_backend":
        internal_name = "Master_Dash_Backend"
    elif target == "dash_frontend":
        internal_name = "Master_Dash_Frontend"
    else:
        internal_name = f"Service_{target}"

    target_idx = -1
    for i, p_info in enumerate(running_processes):
        if p_info["name"].lower() == internal_name.lower():
            target_idx = i
            break

    if target_idx != -1:
        p_info = running_processes[target_idx]
        print(f"\n[RESTART] Terminating existing {p_info['name']} (PID: {p_info['proc'].pid})...")
        kill_process(p_info)
        running_processes.pop(target_idx)
        time.sleep(2)
    else:
        print(f"\n[RESTART] {internal_name} not found in running processes. Starting fresh.")

    if target == "dash_backend":
        launch_master_backend()
        print("[RESTART] Master_Dash_Backend is back online.\n")
    elif target == "dash_frontend":
        launch_master_frontend()
        print("[RESTART] Master_Dash_Frontend is back online.\n")
    else:
        cmd, is_detached, cwd = SERVICE_MAP[target]
        launch_service(cmd, internal_name, cwd, 2, always_detach=is_detached)
        print(f"[RESTART] {internal_name} is back online.\n")


def cleanup():
    print("\n" + "=" * 40)
    print("SHUTDOWN SEQUENCE INITIATED")
    print("=" * 40)

    for p_info in reversed(running_processes):
        print(f"Terminating {p_info['name']}...")
        kill_process(p_info)

    print("   ✅ All orchestrated processes terminated.")

    for f in open_log_files:
        try: f.close()
        except: pass

    hunt_and_kill_port(5173, "Node/Vite")
    hunt_zombie_replay_workers(auto_kill=True)

    print("\nStream offline. All background services and detached windows closed safely.")


# --- Flow Phases ---

def _check_api_liveness():
    listening_ports = {c.laddr.port for c in psutil.net_connections(kind="inet") if c.status == "LISTEN"}
    running_api_ports = API_PORTS & listening_ports
    if not running_api_ports:
        print("\n⚠️  WARNING: No backend API services detected on ports 8001-8008.")
        print("   Dashboard will launch but all data feeds will return errors.")
        print("   Consider running first: python run.py --backend")
    else:
        print(f"   ✅ Detected {len(running_api_ports)} API service(s) on ports: {sorted(running_api_ports)}")


def _boot_infra():
    print("\n--- Booting Infrastructure ---")
    run_detached_console(r"tools\kafka\start_zookeeper.bat", "Zookeeper", ROOT_DIR, 20)
    run_detached_console(r"tools\kafka\start_kafka.bat", "Kafka", ROOT_DIR, 30)


def _do_hard_reset(infra_is_running):
    if infra_is_running:
        print("Force closing Kafka/ZK to allow log deletion...")
        for port in [2181, 9092]:
            hunt_and_kill_port(port, f"Kafka/ZK Port {port}")
        time.sleep(2)

    print("\n--- Hard Resetting Infrastructure ---")
    print("Clearing in-memory API and frontend caches (Ports 8000-8008, 5173)...")
    for port in range(8000, 8009):
        hunt_and_kill_port(port, f"API Port {port}")
    hunt_and_kill_port(5173, "Node/Vite")
    time.sleep(2)

    shutil.rmtree(KAFKA_LOG_DIR, ignore_errors=True)
    shutil.rmtree(ZK_LOG_DIR, ignore_errors=True)

    run_detached_console(r"tools\kafka\start_zookeeper.bat", "Zookeeper", ROOT_DIR, 20)
    run_detached_console(r"tools\kafka\start_kafka.bat", "Kafka", ROOT_DIR, 30)

    print("\n--- Recreating Kafka Topics ---")
    for topic in KAFKA_TOPICS:
        subprocess.run(
            fr"{KAFKA_BIN_DIR}\kafka-topics.bat --create --topic {topic} --bootstrap-server localhost:9092 --partitions 6 --replication-factor 1",
            shell=True,
        )

    print("\n--- Resetting Spark/Stream Files ---")
    for script in RESET_SCRIPTS:
        subprocess.run([VENV_PYTHON, os.path.join(ROOT_DIR, script)], input="yes\n", text=True)

    print("\n--- Clearing DTC History & Replay State ---")
    if os.path.exists(DTC_HISTORY_FILE):
        with open(DTC_HISTORY_FILE, "w", encoding="utf-8") as fh:
            json.dump([], fh)
        print("   ✅ DTC history cleared.")
    if os.path.exists(REPLAY_PID_FILE):
        os.remove(REPLAY_PID_FILE)
        print("   ✅ Replay PID file cleared.")


def _start_all_services():
    for service_key, (cmd, is_detached, cwd) in SERVICE_MAP.items():
        service_name = f"Service_{service_key}"
        # APIs start quickly; engines load models/spark contexts and need more time
        delay = 5 if (is_detached or service_key.startswith("engine_")) else 1
        launch_service(cmd, service_name, cwd, delay, always_detach=is_detached)


def _interactive_loop():
    restartable_list = list(SERVICE_MAP.keys()) + ["dash_backend", "dash_frontend", "master_dashboard"]

    print("\nINTERACTIVE SERVICE MANAGER")
    print("Available services to restart:")
    for key in restartable_list:
        print(f"  - {key}")
    print("\nType a service name and press Enter to restart it.")
    print("Press Ctrl+C at any time to safely shut down the entire emulator.")

    while True:
        target = input("\nemulator> ").strip().lower()
        if target in restartable_list:
            restart_service(target)
        elif target:
            print(f"Unknown service '{target}'. Please choose from the list above.")


# --- Main Entrypoint ---

def main(args):
    interactive = not any([args.reset, args.start, args.backend, args.dashboard])

    # Dashboard-only mode
    if args.dashboard:
        print("\n--- Dashboard Mode ---")
        _check_api_liveness()
        launch_master_backend()
        time.sleep(4)
        launch_master_frontend()
        _interactive_loop()
        return

    # Infra state detection
    print("\nChecking Infrastructure State...")
    listening_ports = {c.laddr.port for c in psutil.net_connections(kind="inet") if c.status == "LISTEN"}
    infra_is_running = bool({2181, 9092} & listening_ports)

    if interactive and infra_is_running:
        if input("Zookeeper/Kafka are already running. Kill them? (y/n): ").lower() == "y":
            for port in [2181, 9092]:
                hunt_and_kill_port(port, f"Kafka/ZK Port {port}")
            infra_is_running = False
            time.sleep(2)

    # Automated --reset always kills infra first so log dirs can be deleted cleanly
    if args.reset and infra_is_running:
        for port in [2181, 9092]:
            hunt_and_kill_port(port, f"Kafka/ZK Port {port}")
        infra_is_running = False
        time.sleep(2)

    # Reset phase
    do_reset = args.reset or (interactive and input("\nReset stream files and topics (Hard Reset)? (y/n): ").lower() == "y")
    if do_reset:
        _do_hard_reset(infra_is_running)
        if not args.start:
            print("\nReset complete. Kafka and Zookeeper are running. Exiting.")
            return
    else:
        if not infra_is_running:
            _boot_infra()
        else:
            print("\n--- Resuming Existing Infrastructure ---")

    # Services phase
    do_start_services = args.start or args.backend or (interactive and input("\nStart Backend Services? (y/n): ").lower() == "y")
    if do_start_services:
        _start_all_services()

    # Dashboard phase — skipped only when --backend is the sole flag
    start_dashboard = args.start or (not args.backend and (interactive and input("\nStart Master Dashboard? (y/n): ").lower() == "y"))
    if start_dashboard:
        launch_master_backend()
        time.sleep(4)
        launch_master_frontend()

    print("\n" + "=" * 50)
    print("SYSTEM READY.")
    print("Action: Start replay using the Notebook.")
    print("=" * 50)

    _interactive_loop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Streaming Emulator Launcher")
    parser.add_argument("--reset",     action="store_true", help="Hard reset stream files and topics, then exit")
    parser.add_argument("--start",     action="store_true", help="Start all services and dashboard without prompts")
    parser.add_argument("--backend",   action="store_true", help="Start backend services only, no dashboard")
    parser.add_argument("--dashboard", action="store_true", help="Start master dashboard only (warns if APIs are not running)")
    parser.add_argument("--head",      action="store_true", help="Launch all services in visible terminal windows instead of log files")
    args = parser.parse_args()

    HEAD_MODE = args.head

    try:
        main(args)
    except KeyboardInterrupt:
        print("\n\n[Ctrl+C Detected: Interrupting sequence safely...]")
    except Exception as e:
        print(f"\n[Unexpected Error: {e}]")
    finally:
        try:
            cleanup()
        except KeyboardInterrupt:
            pass
