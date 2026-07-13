"""Minimal, extensible profiling infrastructure for mini-vllm.

Design principles:
1. Non-invasive: main flow only calls profiler.step() / profiler.scope() at key points.
2. Extensible: adding a new backend = adding a new class, zero changes to ModelRunner.
3. Fail-safe: backend registration failures are logged as warnings, never crash inference.
"""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import torch

from utils.logger import logger


@dataclass
class ProfileEvent:
    """A profiling event with arbitrary key-value metrics."""

    name: str
    step: int | None = None
    timestamp: float = field(default_factory=time.time)
    metrics: dict[str, float] = field(default_factory=dict)


@runtime_checkable
class ProfilerBackend(Protocol):
    """Protocol for profiler backends. Two methods only."""

    def on_event(self, event: ProfileEvent) -> None:
        """Receive a profiling event."""
        ...

    def finalize(self) -> None:
        """Cleanup / export / upload when inference ends."""
        ...


class TorchProfilerBackend:
    """Operator-level profiler using torch.profiler."""

    def __init__(self, profile_dir: str):
        import torch.profiler as tp

        self.profile_dir = profile_dir
        os.makedirs(profile_dir, exist_ok=True)
        schedule = tp.schedule(wait=0, warmup=2, active=20, repeat=1)
        self.prof = tp.profile(
            schedule=schedule,
            on_trace_ready=tp.tensorboard_trace_handler(profile_dir),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
        self._started = False

    def on_event(self, event: ProfileEvent) -> None:
        if event.name == "decode_step" and self.prof is not None:
            self.prof.step()

    def pause(self) -> None:
        if self.prof is not None and self._started:
            self.prof.stop()
            self._started = False

    def resume(self) -> None:
        if self.prof is not None and not self._started:
            self.prof.start()
            self._started = True

    def finalize(self) -> None:
        if self.prof is not None:
            self.prof.stop()


class SwanLabBackend:
    """System-level profiler uploading to SwanLab."""

    def __init__(self, project: str = "mini-vllm", experiment_name: str | None = None):
        import swanlab
        import time

        if experiment_name is None:
            experiment_name = time.strftime("%Y%m%d_%H%M%S")

        swanlab.init(project=project, experiment_name=experiment_name)
        self._swanlab = swanlab

    def on_event(self, event: ProfileEvent) -> None:
        if event.metrics:
            self._swanlab.log(event.metrics, step=event.step)

    def finalize(self) -> None:
        self._swanlab.finish()


class JSONFileBackend:
    """Local JSON file backend for offline analysis."""

    def __init__(self, path: str = "profile.json"):
        self._records: list[dict] = []
        self._path = path

    def on_event(self, event: ProfileEvent) -> None:
        self._records.append({
            "name": event.name,
            "step": event.step,
            "timestamp": event.timestamp,
            **event.metrics,
        })

    def finalize(self) -> None:
        with open(self._path, "w") as f:
            json.dump(self._records, f, indent=2, default=str)
        logger.info(f"[Profiler] JSON profile saved to {self._path}")


class ProfileSession:
    """Central profiler session. Manages multiple backends."""

    def __init__(self):
        self._backends: list[ProfilerBackend] = []
        self._has_operator_profiler = False
        self._has_system_profiler = False

        # Sliding-window state (mirrors progress.py _TokenThroughputColumn)
        self._decode_records: list[tuple[float, float]] = []  # (timestamp, latency_ms)
        self._last_inst_update: float = 0.0
        self._last_instant_throughput: float | None = None
        self._last_window_latency: float | None = None

    def add(self, backend: ProfilerBackend) -> ProfileSession:
        self._backends.append(backend)
        if isinstance(backend, TorchProfilerBackend):
            self._has_operator_profiler = True
        else:
            self._has_system_profiler = True
        return self

    def emit(self, name: str, step: int | None = None, **metrics: float) -> None:
        """Emit a profiling event to all backends."""
        if not self._backends:
            return
        event = ProfileEvent(name=name, step=step, metrics=metrics)
        for b in self._backends:
            try:
                b.on_event(event)
            except Exception as e:
                logger.warning(
                    f"[Profiler] Backend {type(b).__name__} on_event failed: {e}"
                )

    def pause(self) -> None:
        """Pause operator-level profilers (e.g., during CUDA Graph capture)."""
        for b in self._backends:
            if isinstance(b, TorchProfilerBackend):
                try:
                    b.pause()
                except Exception as e:
                    logger.warning(
                        f"[Profiler] Backend {type(b).__name__} pause failed: {e}"
                    )

    def resume(self) -> None:
        """Resume operator-level profilers."""
        for b in self._backends:
            if isinstance(b, TorchProfilerBackend):
                try:
                    b.resume()
                except Exception as e:
                    logger.warning(
                        f"[Profiler] Backend {type(b).__name__} resume failed: {e}"
                    )

    @contextmanager
    def scope(self, name: str):
        """Context manager for a profiling scope (e.g., decode phase).

        Handles operator-level profiler lifecycle (torch.profiler start/stop).
        """
        for b in self._backends:
            if isinstance(b, TorchProfilerBackend):
                try:
                    b.prof.__enter__()
                    b._started = True
                except Exception as e:
                    logger.warning(
                        f"[Profiler] Backend {type(b).__name__} start failed: {e}"
                    )
        try:
            yield self
        finally:
            for b in self._backends:
                if isinstance(b, TorchProfilerBackend):
                    try:
                        b.prof.__exit__(None, None, None)
                        b._started = False
                    except Exception as e:
                        logger.warning(
                            f"[Profiler] Backend {type(b).__name__} stop failed: {e}"
                        )
            self.finalize()

    @contextmanager
    def step(self, step: int | None = None, **extra_metrics: float):
        """Profile a single decode step. Measures latency and memory."""
        if not self._backends:
            yield
            return

        # Trigger operator profiler step before synchronize to keep timeline clean
        for b in self._backends:
            if isinstance(b, TorchProfilerBackend):
                try:
                    b.on_event(ProfileEvent(name="decode_step", step=step))
                except Exception as e:
                    logger.warning(
                        f"[Profiler] TorchProfilerBackend step failed: {e}"
                    )

        if not self._has_system_profiler:
            yield
            return

        # System-level measurement
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t0 = time.time()

        try:
            yield
        finally:
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            latency = (time.time() - t0) * 1000
            self._decode_records.append((time.time(), latency))

            # Sliding-window metrics (1s window, update at most once per second)
            now = time.time()
            if now - self._last_inst_update >= 1.0 and len(self._decode_records) >= 2:
                window_sec = 1.0
                cutoff = self._decode_records[-1][0] - window_sec
                start_idx = 0
                for i, (ts, _) in enumerate(self._decode_records):
                    if ts >= cutoff:
                        start_idx = i
                        break
                count = len(self._decode_records) - start_idx
                if count >= 2:
                    duration = (
                        self._decode_records[-1][0]
                        - self._decode_records[start_idx][0]
                    )
                    if duration > 0:
                        self._last_instant_throughput = (count - 1) / duration
                    window_latencies = [
                        lat
                        for _, lat in self._decode_records[start_idx:]
                    ]
                    self._last_window_latency = (
                        sum(window_latencies) / len(window_latencies)
                    )
                self._last_inst_update = now

            # Prune old records (keep last 5s)
            keep_cutoff = now - 5.0
            while (
                self._decode_records
                and self._decode_records[0][0] < keep_cutoff
            ):
                self._decode_records.pop(0)

            metrics = {
                **extra_metrics,
            }
            if self._last_window_latency is not None:
                metrics["latency_ms"] = self._last_window_latency
                metrics["throughput_tok_s"] = self._last_instant_throughput
            else:
                metrics["latency_ms"] = latency
            if torch.cuda.is_available():
                metrics["memory_mb"] = torch.cuda.memory_allocated() / 1024**2

            self.emit("decode_step", step=step, **metrics)

    def finalize(self) -> None:
        for b in self._backends:
            try:
                b.finalize()
            except Exception as e:
                logger.warning(
                    f"[Profiler] Backend {type(b).__name__} finalize failed: {e}"
                )


def build_profiler(cfg) -> ProfileSession:
    """Build a ProfileSession from GlobalConfig.

    Supports both new-style ``cfg.profiling.*`` and legacy ``cfg.profiling.torch_profiler.enabled``.
    """
    session = ProfileSession()

    # New-style config
    profiling = getattr(cfg, "profiling", None)
    if profiling is not None:
        torch_cfg = getattr(profiling, "torch_profiler", None)
        if torch_cfg and getattr(torch_cfg, "enabled", False):
            try:
                profile_dir = getattr(torch_cfg, "profile_dir", "./log/profile/")
                session.add(TorchProfilerBackend(profile_dir=profile_dir))
                logger.info(f"[Profiler] TorchProfiler enabled: {profile_dir}")
            except Exception as e:
                logger.warning(f"[Profiler] Failed to register TorchProfilerBackend: {e}")

        swanlab_cfg = getattr(profiling, "swanlab", None)
        if swanlab_cfg and getattr(swanlab_cfg, "enabled", False):
            try:
                session.add(
                    SwanLabBackend(
                        project=getattr(swanlab_cfg, "project", "mini-vllm"),
                        experiment_name=getattr(swanlab_cfg, "experiment_name", None),
                    )
                )
                logger.info(
                    f"[Profiler] SwanLab enabled: project={getattr(swanlab_cfg, 'project', 'mini-vllm')}"
                )
            except Exception as e:
                logger.warning(f"[Profiler] Failed to register SwanLabBackend: {e}")

        json_cfg = getattr(profiling, "json", None)
        if json_cfg and getattr(json_cfg, "enabled", False):
            try:
                session.add(
                    JSONFileBackend(path=getattr(json_cfg, "path", "profile.json"))
                )
                logger.info(
                    f"[Profiler] JSON backend enabled: {getattr(json_cfg, 'path', 'profile.json')}"
                )
            except Exception as e:
                logger.warning(f"[Profiler] Failed to register JSONFileBackend: {e}")

        # Legacy fallback: if no new-style backend is enabled
        if not session._backends and cfg.profiling.torch_profiler.enabled:
            try:
                profile_dir = getattr(cfg.path, "profile_dir", "./log/profile/")
                session.add(TorchProfilerBackend(profile_dir=profile_dir))
                logger.info(f"[Profiler] TorchProfiler enabled (legacy): {profile_dir}")
            except Exception as e:
                logger.warning(f"[Profiler] Failed to register TorchProfilerBackend: {e}")

        return session

    # Legacy fallback when profiling config does not exist at all
    if cfg.profiling.torch_profiler.enabled:
        try:
            profile_dir = getattr(cfg.path, "profile_dir", "./log/profile/")
            session.add(TorchProfilerBackend(profile_dir=profile_dir))
            logger.info(f"[Profiler] TorchProfiler enabled (legacy): {profile_dir}")
        except Exception as e:
            logger.warning(f"[Profiler] Failed to register TorchProfilerBackend: {e}")

    return session
