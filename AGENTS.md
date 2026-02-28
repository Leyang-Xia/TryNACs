## Cursor Cloud specific instructions

This is a Python-based Neural Audio Codec educational project (8 progressive stages). There is no web server, database, or Docker dependency — all code runs locally on CPU with synthetic audio data.

### Project structure

- `stage1_audio_basics/` through `stage8_full_codec/` — each stage has a single `.py` file with a `__main__` block for demo/verification
- `requirements.txt` — Python dependencies (torch, torchaudio, numpy, scipy, matplotlib, librosa, soundfile, einops)
- No build system, no test framework, no linter config

### Running stages

Each stage is self-contained and runnable via `python3 stageN_xxx/file.py`. Stage 8 is the full integration test that imports from stages 4-7.

### Key notes

- Stage 8 (`stage8_full_codec/neural_audio_codec.py`) uses `sys.path.insert` to import from sibling stage directories — always run from the repo root (`/workspace`).
- No GPU required; all stages complete on CPU in under 2 minutes total.
- There is no dedicated lint or test command; validation is done by running each stage's `__main__` block.
- Stage 2 training takes ~30s on CPU (50 epochs). Stage 8 training demo (5 steps) takes ~10s.
