"""Bilingual welcome screen for the SAR simulation environment."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import webbrowser

from i18n import set_language, tr


def _open_file(path: Path) -> None:
    if sys.platform.startswith("win"):
        os.startfile(str(path))  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])


def show_welcome_screen(base_dir: str | Path | None = None) -> str | None:
    """Display logo, title, authors and language selection.

    Returns ``"en"`` or ``"it"``. Closing the window or selecting Exit returns
    ``None``.
    """
    import tkinter as tk
    from tkinter import messagebox, ttk

    base = Path(base_dir or Path(__file__).resolve().parent)
    result: dict[str, str | None] = {"language": None}
    root = tk.Tk()
    root.title("SAR simulation environment")
    root.geometry("780x690")
    root.minsize(680, 610)
    root.configure(bg="#050505")

    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass
    style.configure("Welcome.TButton", font=("Segoe UI", 11, "bold"), padding=(15, 8))
    style.configure("Language.TRadiobutton", font=("Segoe UI", 11), background="#050505", foreground="#ffffff")
    style.map("Language.TRadiobutton", background=[("active", "#050505")], foreground=[("active", "#ffffff")])

    content = tk.Frame(root, bg="#050505", padx=28, pady=22)
    content.pack(fill="both", expand=True)

    logo_path = base / "sar_logo.png"
    logo_ref = None
    if logo_path.exists():
        try:
            original = tk.PhotoImage(file=str(logo_path))
            factor = max(1, int(round(original.width() / 640)))
            logo_ref = original.subsample(factor, factor)
            logo_label = tk.Label(content, image=logo_ref, bg="#050505", bd=0)
            logo_label.pack(pady=(0, 8))
        except tk.TclError:
            logo_ref = None

    title_var = tk.StringVar(value=tr("app_title"))
    subtitle_var = tk.StringVar(value=tr("welcome_subtitle"))
    authors_var = tk.StringVar(value=tr("authors"))
    language_label_var = tk.StringVar(value=tr("language"))
    continue_var = tk.StringVar(value=tr("continue"))
    exit_var = tk.StringVar(value=tr("exit"))
    manual_var = tk.StringVar(value=tr("open_manual"))

    tk.Label(content, textvariable=title_var, bg="#050505", fg="#ffffff",
             font=("Segoe UI", 25, "bold")).pack(pady=(6, 3))
    tk.Label(content, textvariable=subtitle_var, bg="#050505", fg="#d9dde5",
             font=("Segoe UI", 11)).pack(pady=(0, 14))
    tk.Label(content, textvariable=authors_var, bg="#050505", fg="#b9c3d0",
             font=("Segoe UI", 10, "italic")).pack(pady=(0, 18))

    language_frame = tk.Frame(content, bg="#15171b", padx=18, pady=12,
                              highlightbackground="#343a44", highlightthickness=1)
    language_frame.pack(pady=(0, 18))
    tk.Label(language_frame, textvariable=language_label_var, bg="#15171b",
             fg="#ffffff", font=("Segoe UI", 11, "bold")).pack(side="left", padx=(0, 18))
    language_var = tk.StringVar(value="en")

    def refresh_language(*_args) -> None:
        set_language(language_var.get())
        title_var.set(tr("app_title"))
        subtitle_var.set(tr("welcome_subtitle"))
        authors_var.set(tr("authors"))
        language_label_var.set(tr("language"))
        continue_var.set(tr("continue"))
        exit_var.set(tr("exit"))
        manual_var.set(tr("open_manual"))

    ttk.Radiobutton(language_frame, text="English", variable=language_var,
                    value="en", command=refresh_language,
                    style="Language.TRadiobutton").pack(side="left", padx=8)
    ttk.Radiobutton(language_frame, text="Italiano", variable=language_var,
                    value="it", command=refresh_language,
                    style="Language.TRadiobutton").pack(side="left", padx=8)

    button_row = tk.Frame(content, bg="#050505")
    button_row.pack(pady=(4, 0))

    def continue_application() -> None:
        selected = language_var.get()
        set_language(selected)
        result["language"] = selected
        root.destroy()

    def open_manual() -> None:
        filename = "SAR_User_Manual_EN.pdf" if language_var.get() == "en" else "Manuale_Utente_SAR_IT.pdf"
        path = base / "manuals" / filename
        if not path.exists():
            messagebox.showwarning(tr("app_title"), tr("manual_missing"), parent=root)
            return
        try:
            _open_file(path)
        except Exception as exc:
            messagebox.showerror(tr("app_title"), tr("manual_error", error=exc), parent=root)

    ttk.Button(button_row, textvariable=manual_var, command=open_manual,
               style="Welcome.TButton").pack(side="left", padx=6)
    ttk.Button(button_row, textvariable=exit_var, command=root.destroy,
               style="Welcome.TButton").pack(side="left", padx=6)
    ttk.Button(button_row, textvariable=continue_var, command=continue_application,
               style="Welcome.TButton").pack(side="left", padx=6)

    root.bind("<Return>", lambda _event: continue_application())
    root.bind("<Escape>", lambda _event: root.destroy())
    root.protocol("WM_DELETE_WINDOW", root.destroy)
    root.mainloop()
    # Keep the image reference alive for the full Tk lifecycle.
    _ = logo_ref
    return result["language"]
