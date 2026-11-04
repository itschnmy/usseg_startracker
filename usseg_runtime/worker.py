"""One spawned worker: cached database, bounded by the parent watchdog."""
from __future__ import annotations

import ctypes
import multiprocessing as mp
import os
from queue import Empty, Queue
import resource
import signal
import sys
from threading import Thread
from time import monotonic, perf_counter_ns

from .decode import FrameError, decode_job, load_dark


def worker_main(connection, database, profiles, cfg, expected_parent):
    # Linux PR_SET_PDEATHSIG also kills an orphan if supervisor gets SIGKILL.
    if sys.platform == "linux":
        ctypes.CDLL(None).prctl(1, signal.SIGKILL)
        if os.getppid() != expected_parent:
            return
    # Supervisor alone handles graceful termination; it controls the child.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    started = perf_counter_ns()
    try:
        import cv2
        from models.pipeline import PipelineConfig, StarTrackerPipeline

        cv2.setNumThreads(1)
        first = next(iter(profiles.values()))
        pipeline = StarTrackerPipeline(PipelineConfig(database=database, **first["pipeline"]))
        for profile in profiles.values():
            config = PipelineConfig(database=database, **profile["pipeline"])
            props = pipeline.solver.database_properties
            if (config.fov_deg+config.fov_max_error_deg < props["min_fov"] or
                    config.fov_deg-config.fov_max_error_deg > props["max_fov"]):
                raise ValueError("Profile FOV does not overlap database")
        darks = {key: load_dark(profile) for key, profile in profiles.items()}
        connection.send({"type": "ready", "database_load_ns": perf_counter_ns()-started})
    except Exception as exc:
        connection.send({"type": "startup_error", "message": f"{type(exc).__name__}: {exc}"})
        return
    try:
        while True:
            request = connection.recv()
            if request is None:
                return
            start = perf_counter_ns()
            metadata = request["metadata"]
            profile = profiles[metadata["camera_profile_id"]]
            try:
                image, conversion = decode_job(metadata, request["directory"], profile, cfg,
                                                darks[metadata["camera_profile_id"]])
                pipeline.config = PipelineConfig(database=database, **profile["pipeline"])
                # Native-depth calibration is owned by adapter; never subtract again.
                pipeline.master_dark = None
                result = pipeline.run_array(image, source=metadata["payload"])
                status = {"ok": "solved", "no_solve": "no_solve", "error": "runtime_error"}[result["status"]]
                connection.send({"status": status, "algorithm": result, "conversion": conversion,
                                 "worker_total_ns": perf_counter_ns()-start,
                                 "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
            except FrameError as exc:
                connection.send({"status": exc.status, "message": str(exc), "worker_total_ns": perf_counter_ns()-start})
            except (ValueError, KeyError, OSError) as exc:
                connection.send({"status": "invalid_input", "message": f"{type(exc).__name__}: {exc}",
                                 "worker_total_ns": perf_counter_ns()-start})
            except Exception as exc:
                connection.send({"status": "runtime_error", "message": f"{type(exc).__name__}: {exc}",
                                 "worker_total_ns": perf_counter_ns()-start})
    except (EOFError, BrokenPipeError):
        return
    finally:
        connection.close()


class Worker:
    def __init__(self, database, profiles, cfg):
        self.database, self.profiles, self.cfg = database, profiles, cfg
        self.process = self.connection = None
        self.reader = None
        self.starts = 0
        self.database_load_ns = []

    def start(self, stopping=lambda: False):
        if self.process is not None and self.process.is_alive():
            return
        self.close()
        context = mp.get_context("spawn")
        parent, child = context.Pipe()
        self.connection = parent
        self.process = context.Process(target=worker_main,
            args=(child, self.database, self.profiles, self.cfg, os.getpid()), daemon=True)
        self.process.start()
        child.close()
        self.starts += 1
        deadline = monotonic() + self.cfg.startup_timeout_s
        while monotonic() < deadline:
            if stopping():
                self.close()
                raise InterruptedError("Stop during worker startup")
            if self.connection.poll(.05):
                try:
                    message = self.connection.recv()
                except EOFError as exc:
                    self.close()
                    raise RuntimeError("Worker exited during startup") from exc
                if message.get("type") != "ready":
                    self.close()
                    raise ValueError(message.get("message", "Worker startup failed"))
                self.database_load_ns.append(message["database_load_ns"])
                return
            if not self.process.is_alive():
                break
        self.close()
        raise RuntimeError("Worker startup timed out or crashed")

    def execute(self, metadata, directory, stopping=lambda: None):
        # Pipe.poll only guarantees SOME bytes are ready. A blocking send/recv
        # can otherwise defeat a watchdog when a worker stops mid-message.
        connection = self.connection
        response = Queue(maxsize=1)
        def exchange():
            try:
                connection.send({"metadata": metadata, "directory": directory})
                response.put((True, connection.recv()))
            except (OSError, EOFError, ValueError) as exc:
                response.put((False, str(exc)))
        self.reader = Thread(target=exchange, daemon=True)
        self.reader.start()
        deadline = monotonic() + self.cfg.frame_timeout_s
        while True:
            stopped_at = stopping()
            if stopped_at is not None and monotonic()-stopped_at >= self.cfg.shutdown_grace_s:
                self.close()
                return {"status": "interrupted", "message": "Shutdown grace expired"}
            try:
                valid, result = response.get(timeout=.02)
                self.reader.join(.1)
                self.reader = None
                if not valid:
                    self.close()
                    return {"status": "runtime_error", "message": "Worker connection lost: " + result}
                return result
            except Empty:
                pass
            if not self.process.is_alive():
                self.close()
                return {"status": "runtime_error", "message": "Worker exited unexpectedly"}
            if monotonic() >= deadline:
                self.close()
                return {"status": "timeout", "message": "Hard frame deadline exceeded"}

    def close(self):
        if self.process is not None:
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(.2)
            if self.process.is_alive():
                self.process.kill()
                self.process.join(1)
            else:
                self.process.join()
        if self.connection is not None:
            self.connection.close()
        if self.reader is not None:
            self.reader.join(1)
            self.reader = None
        self.process = self.connection = None
