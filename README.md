# Coqui / Bark TTS Studio

A simple desktop application for generating speech from text using [Coqui TTS](https://github.com/coqui-ai/TTS) checkpoints. Load your `.pth` voice model, optionally provide the accompanying `config.json`, enter text, and export synthesized audio as a `.wav` file. The UI is built with Tkinter and can be packaged into a standalone executable with PyInstaller.

## Features

- Select custom `.pth` checkpoints and optional configuration files
- Automatically populates available speakers and languages for multi-speaker models
- Threaded synthesis with status logging so the UI stays responsive
- Simple controls for text entry, output selection, and clearing inputs
- PyInstaller build script for generating a one-file executable on Windows, macOS, or Linux

## Getting started

1. **Install dependencies**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Launch the app**
   ```bash
   python -m tts_app.main
   ```

3. **Use your model**
   - Click **Browse** to choose the `.pth` checkpoint and (optionally) the accompanying `config.json` file.
   - Press **Load Model**. Available speakers and languages will populate after the model is ready.
   - Type or paste the text you want to synthesize.
   - Select the output `.wav` path and click **Synthesize**.

> **Tip:** Many Coqui models require both the `.pth` checkpoint and the matching `config.json`. Some models also ship with a speaker embedding file; place it next to the model so the loader can find it automatically.

## Building an executable

Run the helper script to invoke PyInstaller with a sensible default configuration:

```bash
python build_exe.py
```

PyInstaller will create a `dist/coqui_tts_studio` executable (or `.exe` on Windows). You can then distribute this binary together with any model files your users need.

## Troubleshooting

- **Missing dependencies:** Ensure CUDA or other optional dependencies required by your specific model are installed.
- **Large models:** Initial loading may take a while. Status updates appear in the log panel.
- **Bark models:** Bark checkpoints compatible with the Coqui loader can be selected in the same way. For native Bark workflows, adapt `tts_app/engine.py` to instantiate the Bark pipeline instead of `TTS`.
