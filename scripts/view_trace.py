#!/usr/bin/env python3
"""Open the most recent PyTorch profiler trace in Speedscope (default) or Perfetto.

Usage:
    python scripts/open_profile.py
    python scripts/open_profile.py log/profile/some_trace.pt.trace.json
    python scripts/open_profile.py --viewer perfetto
    python scripts/open_profile.py --profile-dir ./my_traces/
    python scripts/open_profile.py --bind 0.0.0.0 --port 8080

WSL: auto-detected; opens browser on the Windows side via cmd.exe.
"""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from threading import Thread

PERFETTO_ORIGIN = "https://ui.perfetto.dev"
SPEEDSCOPE_CDN = "https://cdn.jsdelivr.net/npm/speedscope@latest/dist/release/index.html"

# ---------------------------------------------------------------------------
# HTML wrapper: hosts Perfetto in an iframe and passes the trace via
# postMessage (PING/PONG handshake required by Perfetto).
# ---------------------------------------------------------------------------

HTML_WRAPPER = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Perfetto Trace Loader</title>
  <style>
    body {{ margin: 0; overflow: hidden; font-family: sans-serif; }}
    #perfetto {{ width: 100vw; height: 100vh; border: none; }}
    #status {{
      position: fixed; top: 8px; left: 50%; transform: translateX(-50%);
      background: #333; color: #fff; padding: 6px 14px; border-radius: 4px;
      font-size: 13px; z-index: 100; opacity: 0.9;
    }}
  </style>
</head>
<body>
  <div id="status">Loading Perfetto...</div>
  <iframe id="perfetto" src="{origin}"></iframe>
  <script>
    const STATUS = document.getElementById('status');
    const IFRAME = document.getElementById('perfetto');
    const PERFETTO_ORIGIN = '{origin}';
    let handshakeDone = false;

    function setStatus(msg) {{
      STATUS.textContent = msg;
      console.log('[Perfetto Loader]', msg);
    }}

    // 1. PING/PONG handshake — Perfetto drops messages if not ready
    let pingInterval = null;
    function startPingPong() {{
      pingInterval = setInterval(() => {{
        if (handshakeDone) return;
        IFRAME.contentWindow.postMessage('PING', PERFETTO_ORIGIN);
      }}, 100);
    }}

    window.addEventListener('message', (evt) => {{
      if (evt.origin !== PERFETTO_ORIGIN) return;
      if (evt.data !== 'PONG') return;
      if (handshakeDone) return;
      handshakeDone = true;
      clearInterval(pingInterval);
      setStatus('Perfetto ready, fetching trace...');
      fetchAndSendTrace();
    }});

    function fetchAndSendTrace() {{
      fetch('/trace.json')
        .then(r => {{
          if (!r.ok) throw new Error('HTTP ' + r.status);
          return r.arrayBuffer();
        }})
        .then(buf => {{
          setStatus('Sending ' + (buf.byteLength / (1024*1024)).toFixed(1) + ' MB...');
          IFRAME.contentWindow.postMessage({{
            perfetto: {{
              buffer: buf,
              title: 'PyTorch Profile',
              fileName: 'trace.json'
            }}
          }}, PERFETTO_ORIGIN);
          setStatus('Trace sent to Perfetto');
          setTimeout(() => STATUS.style.display = 'none', 2000);
        }})
        .catch(e => {{
          setStatus('Error: ' + e.message);
          console.error(e);
        }});
    }}

    IFRAME.onload = () => {{
      setStatus('Perfetto iframe loaded, handshaking...');
      startPingPong();
    }};
  </script>
</body>
</html>
""".format(origin=PERFETTO_ORIGIN)


# ---------------------------------------------------------------------------
# Speedscope standalone downloader
# ---------------------------------------------------------------------------

def ensure_speedscope_html() -> Path:
    """Download and cache speedscope standalone HTML locally.

    Serving speedscope from localhost avoids the browser's Private Network
    Access restriction that blocks https://www.speedscope.app from fetching
    http://localhost URLs.
    """
    cache_dir = Path.home() / ".cache" / "mini-vllm"
    cache_file = cache_dir / "speedscope.html"
    if cache_file.exists():
        return cache_file

    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading speedscope standalone HTML to {cache_file} ...", flush=True)
    try:
        urllib.request.urlretrieve(SPEEDSCOPE_CDN, cache_file)
    except Exception as e:
        print(f"Error downloading speedscope: {e}", flush=True)
        print("Please check your network connection.", flush=True)
        sys.exit(1)
    return cache_file


# ---------------------------------------------------------------------------
# Environment detection
# ---------------------------------------------------------------------------

def _is_wsl() -> bool:
    """Detect WSL (not MSYS2/Git Bash)."""
    if os.environ.get("WSL_DISTRO_NAME"):
        return True
    # WSLInterop exists only on real WSL, not MSYS2
    if Path("/proc/sys/fs/binfmt_misc/WSLInterop").exists():
        return True
    return False


def _open_browser(url: str) -> None:
    """Open browser.  On WSL, try Windows-side commands first."""
    if _is_wsl():
        for cmd in [
            ["cmd.exe", "/c", "start", "", url],
            ["powershell.exe", "-Command", f'Start-Process "{url}"'],
        ]:
            try:
                subprocess.run(cmd, check=True, capture_output=True)
                print("  -> Opened browser on Windows side (WSL detected)", flush=True)
                return
            except subprocess.CalledProcessError as e:
                # Keep stderr visible for debugging but don't spam on every fallback
                err = e.stderr.decode("utf-8", errors="replace").strip() if e.stderr else ""
                if err:
                    print(f"  -> {cmd[0]} failed: {err}", flush=True)
                continue
        print("  -> WARNING: Could not open Windows browser from WSL.", flush=True)
        print(f"     Please open this URL manually:\n     {url}")
    else:
        try:
            webbrowser.open(url)
            print("  -> Opened browser", flush=True)
        except Exception as e:
            print(f"  -> WARNING: Could not open browser: {e}", flush=True)
            print(f"     Please open this URL manually:\n     {url}")


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------

class TraceHandler(SimpleHTTPRequestHandler):
    """Serve trace file and viewer HTML."""

    trace_path: Path
    viewer: str
    speedscope_html: Path | None = None

    def __init__(
        self,
        trace_path: Path,
        viewer: str,
        speedscope_html: Path | None,
        *args,
        **kwargs,
    ):
        self.trace_path = trace_path
        self.viewer = viewer
        self.speedscope_html = speedscope_html
        super().__init__(*args, **kwargs)

    def do_GET(self):
        if self.path in ("/trace", "/trace.json"):
            size = self.trace_path.stat().st_size
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(size))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            with open(self.trace_path, "rb") as f:
                self.wfile.write(f.read())
            return

        if self.viewer == "speedscope" and self.path.rstrip("/") == "/speedscope":
            if self.speedscope_html and self.speedscope_html.exists():
                with open(self.speedscope_html, "rb") as f:
                    body = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            else:
                self.send_response(500)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(b"speedscope.html not available\n")
                return

        if self.path in ("/", "/index.html"):
            if self.viewer == "perfetto":
                body = HTML_WRAPPER.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            else:
                # speedscope mode: redirect to /speedscope
                self.send_response(302)
                self.send_header("Location", "/speedscope")
                self.end_headers()
                return

        self.send_response(404)
        self.end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

    def log_message(self, format, *args):
        pass


def find_free_port(host: str, start: int = 18080, end: int = 65535) -> int:
    """Find an available TCP port on *host* in the given range."""
    for port in range(start, end):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex((host, port)) != 0:
                return port
    raise RuntimeError(f"No free port found on {host} in range {start}-{end}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

MAX_WARN_SIZE_MB = 200


def find_latest_trace(profile_dir: Path) -> Path | None:
    traces = sorted(profile_dir.glob("*.pt.trace.json"), key=lambda p: p.stat().st_mtime)
    return traces[-1] if traces else None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Open PyTorch profiler trace in Perfetto (default) or Speedscope"
    )
    parser.add_argument("trace", nargs="?", help="Path to a specific trace JSON file")
    parser.add_argument(
        "--viewer",
        choices=["speedscope", "perfetto"],
        default="perfetto",
        help="Profile viewer to use (default: perfetto)",
    )
    parser.add_argument("--profile-dir", default="./log/profile/", help="Trace directory")
    parser.add_argument("--port", type=int, default=0, help="HTTP server port (0 = auto)")
    parser.add_argument(
        "--bind",
        default="127.0.0.1",
        help="Bind address (default: 127.0.0.1). Use 0.0.0.0 to expose on all interfaces.",
    )
    parser.add_argument("--no-browser", action="store_true", help="Do not open browser")
    args = parser.parse_args()

    # Resolve trace file
    if args.trace:
        trace_path = Path(args.trace).expanduser().resolve()
        if not trace_path.exists():
            print(f"Error: Trace file not found: {trace_path}", flush=True)
            return 1
    else:
        profile_dir = Path(args.profile_dir).expanduser().resolve()
        trace_path = find_latest_trace(profile_dir)
        if trace_path is None:
            print(f"Error: No *.pt.trace.json found in {profile_dir}", flush=True)
            print("Hint: Enable profiling with use_profile: true in your config.", flush=True)
            return 1

    file_size_mb = trace_path.stat().st_size / (1024 * 1024)
    print(f"Trace file: {trace_path}", flush=True)
    print(f"File size:  {file_size_mb:.1f} MB", flush=True)

    if file_size_mb > MAX_WARN_SIZE_MB:
        print(
            f"WARNING: Trace > {MAX_WARN_SIZE_MB} MB. "
            "Browser may need significant RAM to load it.",
            flush=True,
        )

    # Prepare viewer assets
    speedscope_html: Path | None = None
    if args.viewer == "speedscope":
        speedscope_html = ensure_speedscope_html()

    # Server setup
    bind_host = args.bind
    port = args.port or find_free_port(bind_host)

    def handler_factory(*a, **kw):
        return TraceHandler(trace_path, args.viewer, speedscope_html, *a, **kw)

    try:
        server = HTTPServer((bind_host, port), handler_factory)
    except OSError as e:
        print(f"Error: Cannot start server on {bind_host}:{port} — {e}", flush=True)
        return 1

    server_thread = Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    # Build viewer URL
    if args.viewer == "speedscope":
        trace_url = f"http://localhost:{port}/trace.json"
        encoded = urllib.parse.quote(trace_url, safe="")
        url = f"http://localhost:{port}/speedscope/#profileURL={encoded}"
    else:
        url = f"http://localhost:{port}/"

    print(f"Server:     http://{bind_host}:{port}/", flush=True)
    print(f"Viewer:     {args.viewer}", flush=True)
    print(f"Open URL:   {url}", flush=True)

    if not args.no_browser:
        print("Opening browser...", flush=True)
        time.sleep(0.3)
        _open_browser(url)

    print("\nPress Ctrl+C to stop the server.", flush=True)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...", flush=True)
    finally:
        server.shutdown()

    return 0


if __name__ == "__main__":
    sys.exit(main())
