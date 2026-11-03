# Copyright (c) 2026 GodsQuantum
"""Native per-user desktop tray. No polling daemon, no network listener beyond Speaches."""

from __future__ import annotations

import contextlib
import os
from pathlib import Path
import plistlib
import socket
import sys
import threading
import webbrowser

APP_ID = "io.github.GodsQuantum.Cloud9-Speaches"


def _startup_file() -> Path:
    home = Path.home()
    if sys.platform == "darwin":
        return home / "Library" / "LaunchAgents" / f"{APP_ID}.plist"
    return home / ".config" / "autostart" / "cloud9-speaches.desktop"


def startup_enabled() -> bool:
    if os.name == "nt":
        import winreg

        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Run") as key:
                winreg.QueryValueEx(key, "Cloud9-Speaches")
                return True
        except FileNotFoundError:
            return False
    return _startup_file().exists()


def set_startup(enabled: bool) -> None:
    if os.name == "nt":
        import winreg

        with winreg.CreateKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Run") as key:
            if enabled:
                winreg.SetValueEx(key, "Cloud9-Speaches", 0, winreg.REG_SZ, f'"{sys.executable}"')
            else:
                with contextlib.suppress(FileNotFoundError):
                    winreg.DeleteValue(key, "Cloud9-Speaches")
        return
    path = _startup_file()
    if not enabled:
        path.unlink(missing_ok=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if sys.platform == "darwin":
        with path.open("wb") as stream:
            plistlib.dump(
                {"Label": APP_ID, "ProgramArguments": [sys.executable], "RunAtLoad": True, "KeepAlive": False}, stream
            )
    else:
        # Desktop-file launch syntax (quoted paths, no shell expansion).
        exe = str(Path(sys.executable).resolve()).replace("\\", "\\\\").replace('"', '\\"')
        path.write_text(
            f'[Desktop Entry]\nType=Application\nName=Cloud9-Speaches\nExec="{exe}"\n'
            "Terminal=false\nX-GNOME-Autostart-enabled=true\n",
            encoding="utf-8",
        )


def icon_file() -> Path:
    # sys._MEIPASS is PyInstaller's actual bundle data root for onedir builds.
    root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return root / "assets" / "app.png"


def request_show() -> bool:
    """Reopen hidden tray on second app launch without touching the server."""
    try:
        port = int((Path(os.environ["HF_HOME"]).parent.parent / "tray-ipc.port").read_text().strip())
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as client:
            client.sendto(b"SHOW", ("127.0.0.1", port))
        return True
    except (OSError, ValueError, KeyError):
        return False


def run_tray(port: int) -> None:
    from PIL import Image
    import pystray  # pyrefly: ignore[missing-import] - optional desktop-only packaging dependency
    import uvicorn

    from speaches.main import create_app

    server = uvicorn.Server(uvicorn.Config(create_app(), host="127.0.0.1", port=port, log_level="info"))
    worker = threading.Thread(target=server.run, name="Cloud9-Speaches API", daemon=False)
    worker.start()
    url = f"http://127.0.0.1:{port}/"

    def open_ui(_icon: object, _item: object) -> None:
        webbrowser.open(url)

    def toggle_startup(icon: pystray.Icon, _item: object) -> None:
        set_startup(not startup_enabled())
        icon.update_menu()

    def hide_icon(icon: pystray.Icon, _item: object) -> None:
        # Hide only for this session; relaunching the app restores its tray.
        icon.visible = False

    def quit_app(icon: pystray.Icon, _item: object) -> None:
        server.should_exit = True
        icon.stop()

    menu = pystray.Menu(
        pystray.MenuItem("Ouvrir Cloud9-Speaches", open_ui, default=True),
        pystray.MenuItem("Lancer avec la session", toggle_startup, checked=lambda _item: startup_enabled()),
        pystray.MenuItem("Masquer l'icône (jusqu'au prochain lancement)", hide_icon),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Quitter", quit_app),
    )
    icon = pystray.Icon("Cloud9-Speaches", Image.open(icon_file()).convert("RGBA"), "Cloud9-Speaches (local)", menu)
    inbox = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    inbox.bind(("127.0.0.1", 0))
    ipc_path = Path(os.environ["HF_HOME"]).parent.parent / "tray-ipc.port"
    ipc_path.write_text(str(inbox.getsockname()[1]), encoding="ascii")

    def receive_show() -> None:
        while True:
            try:
                packet, _ = inbox.recvfrom(32)
                if packet == b"SHOW":
                    icon.visible = True
            except OSError:
                break

    threading.Thread(target=receive_show, name="Cloud9-Speaches tray activation", daemon=True).start()
    try:
        # Main-thread icon loop is mandatory for macOS Cocoa.
        icon.run()
    finally:
        inbox.close()
        ipc_path.unlink(missing_ok=True)
        server.should_exit = True
        worker.join(timeout=15)
