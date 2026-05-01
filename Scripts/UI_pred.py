"""
ui_infer.py  –  HooperAI  |  Shot Prediction Launcher
======================================================
A tkinter GUI that wraps lstm_infer.py.

Layout
──────
  Left panel  : file paths + run settings
  Right panel : live log console  +  results summary card

Usage
──────
  python ui_infer.py
  (lstm_infer.py must live in the same folder, or adjust INFER_SCRIPT below)
"""

import os
import sys
import json
import threading
import subprocess
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from datetime import datetime


# ── path to the inference script ──────────────────────────────────────────────
INFER_SCRIPT = r"C:\Users\fridr\Documents\HooperAI\Scripts\LSTM\lstm_infer.py"

# ── default paths (pre-fill from the script's own defaults) ──────────────────
DEFAULTS = {
    "yolo":   r"C:\Users\fridr\Documents\HooperAI\data\model\yolo26m_best.pt",
    "lstm":   r"C:\Users\fridr\Documents\HooperAI\data\model\basketball_shot_lstm.keras",
    "scaler": r"C:\Users\fridr\Documents\HooperAI\data\model\basketball_shot_lstm_scaler.pkl",
    "config": r"C:\Users\fridr\Documents\HooperAI\data\model\basketball_shot_lstm_config.pkl",
    "video":  r"C:\Users\fridr\Documents\HooperAI\data\raw\videos\20260406_155804.mp4",
    "output": r"C:\Users\fridr\Documents\HooperAI\data\processed\predictions\2804",
}

# ── colour palette ────────────────────────────────────────────────────────────
C = {
    "bg":          "#0f1117",   # near-black background
    "panel":       "#181c27",   # card background
    "border":      "#252a38",   # subtle borders
    "accent":      "#f97316",   # orange accent (basketball)
    "accent_dim":  "#7c3910",   # muted accent
    "text":        "#e8e8e8",   # primary text
    "text_muted":  "#6b7280",   # secondary text
    "green":       "#22c55e",   # made shot
    "red":         "#ef4444",   # missed shot
    "yellow":      "#eab308",   # warning / medium conf
    "log_bg":      "#0a0d14",   # log console bg
    "log_text":    "#a3c4bc",   # log text colour
    "entry_bg":    "#1e2333",   # input field bg
    "btn":         "#f97316",   # primary button
    "btn_text":    "#ffffff",
    "btn_hover":   "#ea6800",
    "btn_stop":    "#374151",
    "btn_stop_txt":"#d1d5db",
}

FONT_MONO  = ("Consolas",  10)
FONT_BODY  = ("Segoe UI",  10)
FONT_LABEL = ("Segoe UI",  9)
FONT_TITLE = ("Segoe UI",  13, "bold")
FONT_STAT  = ("Segoe UI",  26, "bold")
FONT_STAT_S= ("Segoe UI",  11)


# ─────────────────────────────────────────────────────────────────────────────
class HooperUI(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("HooperAI  –  Shot Prediction")
        self.configure(bg=C["bg"])
        self.minsize(1060, 680)
        self.resizable(True, True)

        # state
        self._proc: subprocess.Popen | None = None
        self._running = False

        # path string vars
        self._paths = {k: tk.StringVar(value=v) for k, v in DEFAULTS.items()}

        # settings vars
        self._yolo_conf   = tk.DoubleVar(value=0.75)
        self._display     = tk.BooleanVar(value=True)
        self._save_video  = tk.BooleanVar(value=True)

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        # ── top title bar ────────────────────────────────────────────────────
        title_bar = tk.Frame(self, bg=C["bg"])
        title_bar.pack(fill="x", padx=20, pady=(16, 8))

        tk.Label(title_bar, text="🏀", font=("Segoe UI Emoji", 20),
                 bg=C["bg"], fg=C["accent"]).pack(side="left")
        tk.Label(title_bar, text="  HooperAI", font=("Segoe UI", 18, "bold"),
                 bg=C["bg"], fg=C["text"]).pack(side="left")
        tk.Label(title_bar, text="  Shot Prediction Launcher",
                 font=("Segoe UI", 11), bg=C["bg"], fg=C["text_muted"]).pack(side="left", pady=4)

        # ── main body (two columns) ──────────────────────────────────────────
        body = tk.Frame(self, bg=C["bg"])
        body.pack(fill="both", expand=True, padx=20, pady=(0, 16))
        body.columnconfigure(0, weight=0, minsize=400)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        self._build_left(body)
        self._build_right(body)

    def _build_left(self, parent):
        left = tk.Frame(parent, bg=C["bg"])
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        left.columnconfigure(0, weight=1)

        # ── file paths card ───────────────────────────────────────────────────
        self._file_card(left, "Model Files", [
            ("YOLO model (.pt)",    "yolo",   [("PyTorch weights", "*.pt")]),
            ("LSTM model (.keras)", "lstm",   [("Keras model", "*.keras")]),
            ("Scaler (.pkl)",       "scaler", [("Pickle", "*.pkl")]),
            ("Config (.pkl)",       "config", [("Pickle", "*.pkl")]),
        ]).pack(fill="x", pady=(0, 10))

        self._file_card(left, "Input / Output", [
            ("Video file",          "video",  [("Video", "*.mp4 *.avi *.mov *.mkv")]),
            ("Output folder",       "output", None),   # None = folder picker
        ]).pack(fill="x", pady=(0, 10))

        # ── settings card ────────────────────────────────────────────────────
        settings_outer = self._card_outer(left, "Run Settings")
        settings_outer.pack(fill="x", pady=(0, 10))
        settings = self._card_inner(settings_outer)

        # YOLO confidence slider
        conf_row = tk.Frame(settings, bg=C["panel"])
        conf_row.pack(fill="x", pady=(4, 8))
        tk.Label(conf_row, text="YOLO confidence",
                 font=FONT_LABEL, bg=C["panel"], fg=C["text_muted"]).pack(side="left")
        self._conf_label = tk.Label(conf_row, text="0.75",
                                    font=FONT_LABEL, bg=C["panel"], fg=C["accent"], width=5)
        self._conf_label.pack(side="right")
        tk.Scale(settings, from_=0.1, to=1.0, resolution=0.05,
                 orient="horizontal", variable=self._yolo_conf,
                 bg=C["panel"], fg=C["text"], troughcolor=C["border"],
                 highlightthickness=0, activebackground=C["accent"],
                 command=lambda v: self._conf_label.config(text=f"{float(v):.2f}"),
                 showvalue=False, length=340).pack(fill="x", pady=(0, 6))

        # checkboxes
        checks = tk.Frame(settings, bg=C["panel"])
        checks.pack(fill="x", pady=(0, 4))
        for text, var in [("Show OpenCV window", self._display),
                           ("Save annotated video", self._save_video)]:
            cb = tk.Checkbutton(checks, text=text, variable=var,
                                font=FONT_BODY, bg=C["panel"], fg=C["text"],
                                selectcolor=C["entry_bg"], activebackground=C["panel"],
                                activeforeground=C["text"], cursor="hand2")
            cb.pack(anchor="w", pady=2)

        # ── run / stop buttons ────────────────────────────────────────────────
        btn_row = tk.Frame(left, bg=C["bg"])
        btn_row.pack(fill="x", pady=(4, 0))
        btn_row.columnconfigure(0, weight=3)
        btn_row.columnconfigure(1, weight=1)

        self._run_btn = self._button(btn_row, "▶  Run Inference",
                                     self._run, C["btn"], C["btn_text"])
        self._run_btn.grid(row=0, column=0, sticky="ew", padx=(0, 6))

        self._stop_btn = self._button(btn_row, "■  Stop",
                                      self._stop, C["btn_stop"], C["btn_stop_txt"])
        self._stop_btn.grid(row=0, column=1, sticky="ew")
        self._stop_btn.config(state="disabled")

    def _build_right(self, parent):
        right = tk.Frame(parent, bg=C["bg"])
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)

        # ── results summary card ──────────────────────────────────────────────
        results_outer = self._card_outer(right, "Results")
        results_outer.pack(fill="x", pady=(0, 10))
        results = self._card_inner(results_outer)

        stats_row = tk.Frame(results, bg=C["panel"])
        stats_row.pack(fill="x", pady=(4, 8))
        for col in range(4):
            stats_row.columnconfigure(col, weight=1)

        self._stat_widgets = {}
        stat_defs = [
            ("total",     "Total shots",   C["text"],   "—"),
            ("made",      "Made",          C["green"],  "—"),
            ("missed",    "Missed",        C["red"],    "—"),
            ("make_pct",  "Make %",        C["accent"], "—"),
        ]
        for col, (key, label, colour, default) in enumerate(stat_defs):
            frame = tk.Frame(stats_row, bg=C["panel"])
            frame.grid(row=0, column=col, padx=8, pady=4, sticky="ew")
            val_lbl = tk.Label(frame, text=default, font=FONT_STAT,
                               bg=C["panel"], fg=colour)
            val_lbl.pack()
            tk.Label(frame, text=label, font=FONT_STAT_S,
                     bg=C["panel"], fg=C["text_muted"]).pack()
            self._stat_widgets[key] = val_lbl

        # shot timeline
        self._shot_frame = tk.Frame(results, bg=C["panel"])
        self._shot_frame.pack(fill="x", pady=(0, 4))

        # ── log console ───────────────────────────────────────────────────────
        log_outer = self._card_outer(right, "Live Log")
        log_outer.pack(fill="both", expand=True)
        log_card = self._card_inner(log_outer)

        log_inner = tk.Frame(log_card, bg=C["log_bg"], bd=0)
        log_inner.pack(fill="both", expand=True)

        self._log = tk.Text(log_inner, wrap="word", state="disabled",
                            bg=C["log_bg"], fg=C["log_text"],
                            font=FONT_MONO, relief="flat", bd=0,
                            insertbackground=C["log_text"],
                            selectbackground=C["accent_dim"])
        scroll = ttk.Scrollbar(log_inner, orient="vertical",
                               command=self._log.yview)
        self._log.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self._log.pack(fill="both", expand=True, padx=6, pady=6)

        # tag colours
        self._log.tag_config("info",  foreground=C["log_text"])
        self._log.tag_config("ok",    foreground=C["green"])
        self._log.tag_config("warn",  foreground=C["yellow"])
        self._log.tag_config("err",   foreground=C["red"])
        self._log.tag_config("head",  foreground=C["accent"])
        self._log.tag_config("dim",   foreground=C["text_muted"])

        # clear button
        tk.Button(log_outer, text="Clear log", font=FONT_LABEL,
                  bg=C["panel"], fg=C["text_muted"], relief="flat",
                  activebackground=C["border"], cursor="hand2",
                  command=self._clear_log).pack(anchor="e", padx=12, pady=(0, 6))

        self._log_line("HooperAI ready. Configure paths and press Run.", "head")

    # ── helper widget builders ────────────────────────────────────────────────

    def _card_outer(self, parent, title: str) -> tk.Frame:
        """Return a titled card outer Frame (use pack/grid on this)."""
        outer = tk.Frame(parent, bg=C["panel"],
                         highlightbackground=C["border"],
                         highlightthickness=1)
        tk.Label(outer, text=title.upper(), font=("Segoe UI", 8, "bold"),
                 bg=C["panel"], fg=C["accent"]).pack(anchor="w", padx=12, pady=(10, 4))
        return outer

    def _card_inner(self, outer: tk.Frame) -> tk.Frame:
        """Return the content frame inside a card outer frame."""
        inner = tk.Frame(outer, bg=C["panel"])
        inner.pack(fill="both", expand=True, padx=12, pady=(0, 10))
        return inner

    def _card(self, parent, title: str) -> tk.Frame:
        """Convenience: create card and return inner content frame.
        Only use this when the caller packs the *inner* frame directly
        (left panel usage). For right panel use _card_outer/_card_inner."""
        outer = self._card_outer(parent, title)
        return self._card_inner(outer)

    def _file_card(self, parent, title: str, fields: list) -> tk.Frame:
        """Card containing labelled path entries with browse buttons.
        Returns the outer frame so the caller can pack() it."""
        outer = self._card_outer(parent, title)
        card  = self._card_inner(outer)
        card.columnconfigure(0, weight=1)

        for label_text, key, filetypes in fields:
            row = tk.Frame(card, bg=C["panel"])
            row.pack(fill="x", pady=3)
            row.columnconfigure(1, weight=1)

            tk.Label(row, text=label_text, font=FONT_LABEL, width=20, anchor="w",
                     bg=C["panel"], fg=C["text_muted"]).grid(row=0, column=0, sticky="w")

            entry = tk.Entry(row, textvariable=self._paths[key],
                             font=FONT_LABEL, bg=C["entry_bg"], fg=C["text"],
                             insertbackground=C["text"], relief="flat",
                             highlightthickness=1,
                             highlightbackground=C["border"],
                             highlightcolor=C["accent"])
            entry.grid(row=0, column=1, sticky="ew", padx=(6, 4))

            btn = tk.Button(row, text="…", font=FONT_LABEL,
                            bg=C["border"], fg=C["text"], relief="flat",
                            activebackground=C["accent"], activeforeground="#fff",
                            cursor="hand2", width=3,
                            command=lambda k=key, ft=filetypes: self._browse(k, ft))
            btn.grid(row=0, column=2)

        return outer

    def _button(self, parent, text, cmd, bg, fg):
        btn = tk.Button(parent, text=text, command=cmd,
                        font=("Segoe UI", 11, "bold"),
                        bg=bg, fg=fg, relief="flat",
                        activebackground=C["btn_hover"], activeforeground="#fff",
                        cursor="hand2", pady=10)
        return btn

    # ── browse callbacks ──────────────────────────────────────────────────────

    def _browse(self, key: str, filetypes):
        if filetypes is None:
            # folder picker
            path = filedialog.askdirectory(title="Select output folder",
                                           initialdir=self._paths[key].get() or "/")
        else:
            path = filedialog.askopenfilename(title=f"Select {key}",
                                              filetypes=filetypes,
                                              initialdir=Path(self._paths[key].get()).parent
                                              if self._paths[key].get() else "/")
        if path:
            self._paths[key].set(path)

    # ── log helpers ───────────────────────────────────────────────────────────

    def _log_line(self, text: str, tag: str = "info"):
        self._log.configure(state="normal")
        ts = datetime.now().strftime("%H:%M:%S")
        self._log.insert("end", f"[{ts}] ", "dim")
        self._log.insert("end", text + "\n", tag)
        self._log.see("end")
        self._log.configure(state="disabled")

    def _clear_log(self):
        self._log.configure(state="normal")
        self._log.delete("1.0", "end")
        self._log.configure(state="disabled")

    # ── results update ────────────────────────────────────────────────────────

    def _update_stats(self, total, made, missed, pct):
        self._stat_widgets["total"].config(text=str(total))
        self._stat_widgets["made"].config(text=str(made))
        self._stat_widgets["missed"].config(text=str(missed))
        self._stat_widgets["make_pct"].config(text=f"{pct:.0f}%")

    def _add_shot_pill(self, shot_num: int, result: str, prob: float, conf: str):
        """Add a small coloured pill to the shot timeline."""
        colour = C["green"] if result == "MADE" else C["red"]
        pill = tk.Label(self._shot_frame,
                        text=f"#{shot_num} {result}\n{prob:.0%} [{conf}]",
                        font=("Segoe UI", 8), bg=colour, fg="#fff",
                        padx=6, pady=3, relief="flat", cursor="hand2")
        pill.pack(side="left", padx=3, pady=4)

    def _clear_shots(self):
        for w in self._shot_frame.winfo_children():
            w.destroy()
        for key in self._stat_widgets:
            self._stat_widgets[key].config(text="—")

    # ── run / stop ────────────────────────────────────────────────────────────

    def _validate(self) -> bool:
        """Check all required files exist before running."""
        checks = [
            ("yolo",   "YOLO model"),
            ("lstm",   "LSTM model"),
            ("scaler", "Scaler"),
            ("config", "Config"),
            ("video",  "Video"),
        ]
        missing = []
        for key, label in checks:
            p = self._paths[key].get().strip()
            if not p or not os.path.exists(p):
                missing.append(label)
        if missing:
            messagebox.showerror("Missing files",
                                 "The following files were not found:\n\n" +
                                 "\n".join(f"  • {m}" for m in missing))
            return False
        return True

    def _run(self):
        if self._running:
            return
        if not self._validate():
            return

        self._clear_shots()
        self._clear_log()
        self._log_line("Starting inference …", "head")

        # Build a temporary patched version of lstm_infer.py with the UI's settings
        script = self._build_script()
        self._run_btn.config(state="disabled")
        self._stop_btn.config(state="normal")
        self._running = True

        thread = threading.Thread(target=self._run_thread, args=(script,), daemon=True)
        thread.start()

    def _build_script(self) -> str:
        """
        Generate a runnable Python string that imports lstm_infer and calls it
        with the current UI settings, without touching the original file.
        """
        yolo   = self._paths["yolo"].get().replace("\\", "\\\\")
        lstm   = self._paths["lstm"].get().replace("\\", "\\\\")
        scaler = self._paths["scaler"].get().replace("\\", "\\\\")
        config = self._paths["config"].get().replace("\\", "\\\\")
        video  = self._paths["video"].get().replace("\\", "\\\\")
        output = self._paths["output"].get().replace("\\", "\\\\")
        conf   = self._yolo_conf.get()
        disp   = str(self._display.get())
        save   = str(self._save_video.get())

        return f"""
import sys
sys.path.insert(0, r"{Path(INFER_SCRIPT).parent}")

# Override module-level constants before importing
import lstm_infer as _m
_m.YOLO_MODEL   = r"{yolo}"
_m.LSTM_MODEL   = r"{lstm}"
_m.SCALER_PATH  = r"{scaler}"
_m.CONFIG_PATH  = r"{config}"
_m.VIDEO_PATH   = r"{video}"
_m.OUTPUT_DIR   = r"{output}"
_m.YOLO_CONF    = {conf}
_m.DISPLAY      = {disp}
_m.SAVE_VIDEO   = {save}

predictor = _m.RealTimeShotPredictor(
    yolo_path   = r"{yolo}",
    lstm_path   = r"{lstm}",
    scaler_path = r"{scaler}",
    config_path = r"{config}",
    output_dir  = r"{output}",
)

out_folder, stats = predictor.process_video(
    video_path = r"{video}",
    display    = {disp},
    save_video = {save},
    verbose    = True,
)

import json, sys
print("__HOOPER_STATS__" + json.dumps(stats.get("summary", {{}})))
print("__HOOPER_SHOTS__" + json.dumps(stats.get("shots", [])))
"""

    def _run_thread(self, script: str):
        """Run the inference in a subprocess, stream stdout to the log."""
        tmp = Path(__file__).parent / "_ui_runner_tmp.py"
        tmp.write_text(script, encoding="utf-8")

        # Force UTF-8 and suppress noisy TF / oneDNN logs
        env = os.environ.copy()
        env["PYTHONIOENCODING"]      = "utf-8"
        env["TF_ENABLE_ONEDNN_OPTS"] = "0"
        env["TF_CPP_MIN_LOG_LEVEL"]  = "2"

        try:
            self._proc = subprocess.Popen(
                [sys.executable, "-u", str(tmp)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )

            for line in self._proc.stdout:
                line = line.rstrip()
                if not line:
                    continue

                if line.startswith("__HOOPER_STATS__"):
                    self._parse_stats(line[len("__HOOPER_STATS__"):])
                    continue
                if line.startswith("__HOOPER_SHOTS__"):
                    self._parse_shots(line[len("__HOOPER_SHOTS__"):])
                    continue

                # skip low-value TF / absl spam
                ll = line.lower()
                if any(s in ll for s in ("absl", "i0000", "onednn", "port.cc", "tf_enable")):
                    continue

                # colour-code log lines
                tag = "info"
                if any(k in ll for k in ("ok", "succeeded", "ready", "done", "complete")):
                    tag = "ok"
                elif any(k in ll for k in ("warning", "cancelled", "userwarn", "warn")):
                    tag = "warn"
                elif any(k in ll for k in ("error", "failed", "traceback", "exception")):
                    tag = "err"
                elif any(k in ll for k in ("===", "[shot", "processing", "initialising", "loading")):
                    tag = "head"

                self.after(0, self._log_line, line, tag)

            self._proc.wait()
            rc = self._proc.returncode
            self.after(0, self._log_line,
                       f"Process finished (exit code {rc})",
                       "ok" if rc == 0 else "err")

        except Exception as e:
            self.after(0, self._log_line, f"Error: {e}", "err")
        finally:
            if tmp.exists():
                tmp.unlink()
            self.after(0, self._on_run_finished)

    def _parse_stats(self, raw: str):
        try:
            s = json.loads(raw)
            self.after(0, self._update_stats,
                       s.get("total_shots", 0),
                       s.get("predicted_made", 0),
                       s.get("predicted_miss", 0),
                       s.get("predicted_make_pct", 0.0))
        except Exception:
            pass

    def _parse_shots(self, raw: str):
        try:
            shots = json.loads(raw)
            for shot in shots:
                self.after(0, self._add_shot_pill,
                           shot["shot_number"],
                           shot["prediction_text"],
                           shot["probability"],
                           shot["confidence"])
        except Exception:
            pass

    def _on_run_finished(self):
        self._running = False
        self._proc    = None
        self._run_btn.config(state="normal")
        self._stop_btn.config(state="disabled")

    def _stop(self):
        if self._proc and self._running:
            self._log_line("Stopping process …", "warn")
            self._proc.terminate()

    def _on_close(self):
        self._stop()
        self.destroy()


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = HooperUI()
    app.mainloop()