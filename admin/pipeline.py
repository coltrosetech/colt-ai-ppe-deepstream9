import os
import subprocess
import threading
from pathlib import Path

from deepstream.config import render_config, resolve_person_profile


class PipelineManager:
    def __init__(self):
        self.process = None
        self.lock = threading.RLock()
        self.dry_run = os.getenv("DEEPSAFE_DRY_RUN", "true").lower() == "true"
        root = Path(os.getenv("DEEPSAFE_DATA", "/tmp/deepsafe"))
        self.config_path = Path(os.getenv("DEEPSAFE_DS_CONFIG", root / "generated/deepstream-app.ini"))

    def status(self, state=None):
        running = self.process is not None and self.process.poll() is None
        result = {"running": running, "dry_run": self.dry_run,
                  "pid": self.process.pid if running else None, "config": str(self.config_path)}
        if state is not None:
            profile = resolve_person_profile(state)
            result["person_profile"] = profile
            result["source_plan"] = {
                "configured": len(state["sources"]),
                "engine_max_batch": profile["max_batch_size"],
                "benchmark_simulated_sources": profile["benchmark_sources"],
                "benchmark_duration_seconds": profile["benchmark_duration_seconds"],
            }
        return result

    def start(self, state):
        with self.lock:
            if not state["sources"]:
                raise ValueError("En az bir video/RTSP kaynagi gerekli")
            if self.status()["running"]:
                return self.status(state)
            render_config(state, self.config_path)
            if not self.dry_run:
                cmd = os.getenv("DEEPSAFE_DS_COMMAND", "deepstream-app")
                self.process = subprocess.Popen([cmd, "-c", str(self.config_path)])
            return self.status(state) | {"configured": True}

    def stop(self):
        with self.lock:
            if self.status()["running"]:
                self.process.terminate()
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.kill()
            self.process = None
            return self.status()
