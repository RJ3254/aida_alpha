"""Tkinter GUI for interacting with Coqui TTS models."""
from __future__ import annotations

import threading
from pathlib import Path
from tkinter import (
    BOTH,
    END,
    LEFT,
    RIGHT,
    VERTICAL,
    Button,
    Entry,
    Frame,
    Label,
    LabelFrame,
    Listbox,
    Scrollbar,
    StringVar,
    Text,
    Tk,
    filedialog,
    messagebox,
    ttk,
)

from .engine import TTSModelError, TTSModelManager


class TTSApp:
    """Main GUI application."""

    def __init__(self, root: Tk) -> None:
        self.root = root
        self.root.title("Coqui / Bark TTS Studio")
        self.manager = TTSModelManager()

        self.model_path_var = StringVar()
        self.config_path_var = StringVar()
        self.output_path_var = StringVar()
        self.speaker_var = StringVar()
        self.language_var = StringVar()

        self._build_ui()

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        container = Frame(self.root, padx=10, pady=10)
        container.pack(fill=BOTH, expand=True)

        # Model selection frame
        model_frame = LabelFrame(container, text="Model", padx=10, pady=10)
        model_frame.pack(fill=BOTH, expand=True)

        self._add_path_selector(
            model_frame,
            label="Model (.pth):",
            textvariable=self.model_path_var,
            command=lambda: self._browse_file(self.model_path_var, filetypes=[("Model checkpoint", "*.pth"), ("All files", "*.*")]),
        )
        self._add_path_selector(
            model_frame,
            label="Config (.json):",
            textvariable=self.config_path_var,
            command=lambda: self._browse_file(self.config_path_var, filetypes=[("Config", "*.json"), ("All files", "*.*")]),
        )

        Button(model_frame, text="Load Model", command=self._on_load_model).pack(anchor="e", pady=(5, 0))

        # Speaker / language selection
        options_frame = Frame(container)
        options_frame.pack(fill=BOTH, expand=True, pady=(10, 0))

        speaker_label = Label(options_frame, text="Speaker:")
        speaker_label.grid(row=0, column=0, sticky="w")
        self.speaker_combo = ttk.Combobox(options_frame, textvariable=self.speaker_var, state="readonly")
        self.speaker_combo.grid(row=0, column=1, padx=5, sticky="ew")

        language_label = Label(options_frame, text="Language:")
        language_label.grid(row=0, column=2, sticky="w", padx=(10, 0))
        self.language_combo = ttk.Combobox(options_frame, textvariable=self.language_var, state="readonly")
        self.language_combo.grid(row=0, column=3, padx=5, sticky="ew")

        options_frame.columnconfigure(1, weight=1)
        options_frame.columnconfigure(3, weight=1)

        # Text input area
        text_frame = LabelFrame(container, text="Text", padx=10, pady=10)
        text_frame.pack(fill=BOTH, expand=True, pady=(10, 0))

        self.text_box = Text(text_frame, height=6, wrap="word")
        self.text_box.pack(fill=BOTH, expand=True)

        # Output selection
        output_frame = Frame(container)
        output_frame.pack(fill=BOTH, expand=True, pady=(10, 0))

        self._add_path_selector(
            output_frame,
            label="Output file (.wav):",
            textvariable=self.output_path_var,
            command=lambda: self._browse_save_file(self.output_path_var, defaultextension=".wav", filetypes=[("Waveform audio", "*.wav"), ("All files", "*.*")]),
        )

        # Action buttons
        actions_frame = Frame(container)
        actions_frame.pack(fill=BOTH, expand=True, pady=(10, 0))

        Button(actions_frame, text="Synthesize", command=self._on_synthesize).pack(side=LEFT)
        Button(actions_frame, text="Clear", command=self._on_clear_text).pack(side=LEFT, padx=(10, 0))

        # Status display
        status_frame = LabelFrame(container, text="Status", padx=10, pady=10)
        status_frame.pack(fill=BOTH, expand=True, pady=(10, 0))

        self.status_list = Listbox(status_frame, height=6)
        self.status_list.pack(side=LEFT, fill=BOTH, expand=True)

        scrollbar = Scrollbar(status_frame, orient=VERTICAL, command=self.status_list.yview)
        scrollbar.pack(side=RIGHT, fill="y")
        self.status_list.config(yscrollcommand=scrollbar.set)

    # ------------------------------------------------------------------ Helpers
    def _add_path_selector(self, parent: Frame, *, label: str, textvariable, command) -> None:
        row = Frame(parent)
        row.pack(fill=BOTH, expand=True, pady=2)

        Label(row, text=label, width=15, anchor="w").pack(side=LEFT)
        entry = Entry(row, textvariable=textvariable)
        entry.pack(side=LEFT, fill=BOTH, expand=True)
        Button(row, text="Browse", command=command).pack(side=LEFT, padx=(5, 0))

    def _browse_file(self, variable, *, filetypes):
        path = filedialog.askopenfilename(filetypes=filetypes)
        if path:
            variable.set(path)

    def _browse_save_file(self, variable, *, defaultextension, filetypes):
        path = filedialog.asksaveasfilename(defaultextension=defaultextension, filetypes=filetypes)
        if path:
            variable.set(path)

    def _log(self, message: str) -> None:
        def append() -> None:
            self.status_list.insert(END, message)
            self.status_list.see(END)

        self.root.after(0, append)

    def _show_error(self, title: str, message: str) -> None:
        self.root.after(0, lambda: messagebox.showerror(title, message))

    # ------------------------------------------------------------------ Events
    def _on_load_model(self) -> None:
        model_path = Path(self.model_path_var.get())
        if not model_path.exists():
            self._show_error("Missing model", "Please choose a valid .pth file")
            return

        config_path = Path(self.config_path_var.get()) if self.config_path_var.get() else None
        if config_path and not config_path.exists():
            self._show_error("Missing config", "The selected config.json was not found")
            return

        def worker() -> None:
            try:
                self.manager.load_model(model_path, config_path, progress_callback=self._log)
                self.root.after(0, self._update_speakers)
                self.root.after(0, self._update_languages)
                self._log("Ready to synthesize!")
            except TTSModelError as exc:
                self._show_error("Model error", str(exc))

        threading.Thread(target=worker, daemon=True).start()

    def _on_synthesize(self) -> None:
        if not self.manager.is_loaded:
            self._show_error("Model not loaded", "Load a model before synthesizing audio")
            return

        output_path = self.output_path_var.get()
        if not output_path:
            self._show_error("Missing output", "Please choose an output file")
            return

        text = self.text_box.get("1.0", END)
        speaker = self.speaker_var.get() or None
        language = self.language_var.get() or None

        def worker() -> None:
            try:
                self.manager.synthesize(
                    text,
                    Path(output_path),
                    speaker=speaker,
                    language=language,
                    progress_callback=self._log,
                )
            except TTSModelError as exc:
                self._show_error("Synthesis error", str(exc))

        threading.Thread(target=worker, daemon=True).start()

    def _on_clear_text(self) -> None:
        self.text_box.delete("1.0", END)

    def _update_speakers(self) -> None:
        speakers = self.manager.speakers
        if speakers:
            self.speaker_combo["values"] = speakers
            self.speaker_combo.current(0)
        else:
            self.speaker_combo.set("")
            self.speaker_combo["values"] = []

    def _update_languages(self) -> None:
        languages = self.manager.languages
        if languages:
            self.language_combo["values"] = languages
            self.language_combo.current(0)
        else:
            self.language_combo.set("")
            self.language_combo["values"] = []


def main() -> None:
    root = Tk()
    app = TTSApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
