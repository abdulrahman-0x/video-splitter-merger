# Plan (v2): Add DeepFilterNet noise reduction to the Python audio/video splitter app

**Audience:** AI coding agent with access to the app's source code.
**Platform:** Windows 10/11, Python app that splits audio/video (ffmpeg-based, to be confirmed in Phase 0).
**v2 note:** Section 2 now contains **measured facts** from a probe run on the user's machine (deep-filter 0.5.6, i7-7700HQ 4C/8T, 16 GB RAM, Windows 10). They replace the assumptions of v1. Section 3 lists the design decisions that follow from them. Do not re-derive them; only run the short smoke tests in Phase 1.

---

## 1. Goal and ground rules

**Goal:** Add an optional "Reduce noise" step that runs the `deep-filter` command-line tool on the app's audio, with user-adjustable strength, without changing anything when the option is off.

1. **Discover first, edit second.** Do Phase 0 and report before changing code.
2. **Backward compatible.** With the option disabled the app must behave exactly as before (same outputs, same filenames).
3. **Shell out to the exe.** Do not add PyTorch or the `deepfilternet` pip package.
4. **No `shell=True`.** Pass arguments as a list. Paths contain Arabic letters, Arabic-Indic digits and spaces.
5. **Small, local diff.** One new module (suggested `denoise.py`) plus minimal hooks. Follow the existing code style, config mechanism, logging, and GUI framework.
6. **ONE input file per `deep-filter` call. Never pass several files in one call.** (See 2.1: same-name inputs corrupt silently, and a failing file aborts the rest.)
7. **Never use `-a 0`.** (See 2.1: it is a literal passthrough and breaks alignment when combined with `-D`.)
8. **Drain the process output.** Do not leave stdout/stderr as undrained pipes (see 2.1: 322 KB of stderr for a one-hour hot recording; a full pipe would deadlock the child). Redirect to files in the temp dir, or read in threads.

---

## 2. Measured facts about `deep-filter` 0.5.6 (this machine)

### 2.1 Behavior
| Topic | Measured result |
|---|---|
| Default model | **DeepFilterNet3** (`-vv`: "Loading model DeepFilterNet3_onnx.tar.gz", "lookahead 2"). `-m` loads an external tar.gz. |
| Input containers | WAV only. mp3/flac/mp4 -> exit 1 (`Hound Error: no RIFF tag found`). |
| Input sample rates | 8k, 16k, 22.05k, 32k, 44.1k, 48k, 96k all accepted, **output keeps the input rate**, same duration. Audio **quality at non-48 kHz is unmeasured**. |
| Input bit depth | u8, s16, f32 accepted (output is always **s16**). **s24 and s32 crash with a Rust panic (exit 101, `TooWide`)**, no output. |
| Channels | 1, 2, 3, 6 accepted; output keeps channel count. |
| Output name/location | `<output-dir>/<basename of input>`. The input folder is discarded. Nested `-o` dirs are created automatically (even when processing later fails). |
| Overwrite | Re-running into the same `-o` silently overwrites. |
| **Same basename, several inputs, one call** | **Silent corruption** (exit 0, one output file, content matches neither input). Never do this. |
| Several inputs, one is bad | Aborts at the first bad file (exit 1). Earlier outputs remain, later files are not processed. |
| Exit codes | 0 success; 1 runtime/IO error; 2 argument parse error; 101 Rust panic. No partial output file is left by a failed single-file run. |
| **`-D` and length** | Without `-D` the enhanced output is **delayed 1440 samples (30 ms at 48 kHz)**. With `-D` it is aligned (lag 0) but **1440 samples shorter than the input** (the tail is dropped). |
| `-a` (attenuation limit) | Any float >= 0 works (fractions OK, 7.5 differs from 7 and 8). **100, 101, 150 give identical output** (clamped). `0` = bit-exact passthrough with a WARN, and with `-D` the output is `input[1440:]` (shifted 30 ms). Negative values only via `--atten-lim-db=-5` (no range check). Non-numbers -> exit 2. |
| Progress output | **None** when piped (no bar, no percentage, no ETA). `-v` only adds a final line: `Enhanced audio file <name> in <s> (RTF: <x>)`. |
| stderr volume | Silent on clean audio. On clipped/hot audio it prints `WARN df::tract Possible clipping detected (1.000).` per event, about **322,800 bytes per hour** of a real lecture. Exit code and output are unaffected; the output is not itself clipped. |
| Unprobed | `--pf` / `--pf-beta` behavior; latency at non-48 kHz rates. |

### 2.2 Performance (this machine: i7-7700HQ, 4 cores / 8 threads)
| Case | Result |
|---|---|
| 60 min, mono 48k s16 | wall 692 s -> **RTF 0.19** (about 5x faster than real time); **peak RAM 1424 MB**; about 1 core busy |
| 10 min mono | RTF 0.15 to 0.17, peak RAM 272 MB |
| 333 s stereo | **RTF 0.26** (about 1.4x the mono cost); stereo RAM not measured, estimate 2x mono |
| RAM scaling | roughly linear: **about 0.4 to 0.45 MB per second of mono audio** |
| 2 processes in parallel (10-min mono files) | 148.8 s total vs 203.8 s sequential: only **1.37x** faster, each process slows to RTF 0.25 |

### 2.3 Strength sweep (`-D -a N`, 60 s of real lecture speech, dBFS)
| N | overall RMS | noise floor (quietest 10% of 20 ms frames) | speech level (loudest 10%) | noise-floor change |
|---|---|---|---|---|
| input | -12.10 | -25.00 | -5.72 | - |
| 3 | -13.80 | -27.99 | -7.16 | -3.0 |
| 6 | -15.21 | -30.99 | -8.31 | -6.0 |
| 12 | -17.19 | -36.96 | -9.85 | -12.0 |
| 20 | -18.56 | -44.89 | -10.87 | -19.9 |
| 30 | -19.21 | -54.63 | -11.35 | -29.6 |
| 50 | -19.48 | -71.67 | -11.56 | -46.7 |
| 70 | -19.51 | -78.53 | -11.58 | -53.5 |
| 100 | -19.52 | -79.47 | -11.58 | -54.5 |

Reading: the noise-floor reduction is roughly **N dB up to about 50**, and the curve flattens between 70 and 100 (less than 1 dB). **Speech also gets quieter** (about 5.9 dB at full strength, 2.6 dB at N=6). This is one recording; other material will differ.

### 2.4 The user's real source format
The real lecture file used in the probe was **48 kHz, stereo, `pcm_s24le`**. That is exactly a format that makes `deep-filter` panic. **Conversion to 16-bit is therefore not an edge case for this user, it is the normal case.**

---

## 3. Design decisions that follow from the facts

1. **Always feed `deep-filter` a 48 kHz, `pcm_s16le`, WAV file** produced by ffmpeg (even though other rates are accepted, their quality is unmeasured, and s24/s32 crash). Source rate is restored afterwards if requested.
2. **Always work in a private temp dir with short ASCII names** (`in.wav` -> `out\in.wav`), one file per call. This removes the Unicode, name-collision, overwrite, and abort-on-bad-file problems at once.
3. **Fix the 30 ms tail loss without losing audio:** append silence to the input before denoising, run with `-D`, then trim the result to the exact original length.
   - Pad with at least 1440 samples; use **0.1 s (4800 samples at 48 kHz)** for margin.
   - After `-D`, the output has `padded_length - 1440` samples and is time-aligned with the input. Trim it to `N48` (the original length at 48 kHz, in samples). Nothing real is lost and A/V sync is preserved.
   - Smoke-test this in Phase 1 (lag 0, exact length) before relying on it.
4. **No `-a 0`.** Strength `<= 0` means "denoise off" for that item: skip `deep-filter` entirely. Valid strength is `1..100`; above 100 is clamped to 100 (identical output anyway).
5. **Process units of bounded length.** Memory grows about 0.45 MB/s per channel (about 1.6 GB per mono hour, about 3.2 GB per stereo hour estimated). Therefore:
   - **Preferred stage: per part after splitting** (each part is a short, bounded unit; also gives natural progress/cancel granularity).
   - Estimate `ram_mb = duration_s * channels * 0.45`. If a unit's estimate exceeds `min(4096 MB, 25% of available RAM)`, do not run it as one unit. If the app has no smaller unit available, ask the user how to proceed (chunking with overlap + crossfade is possible but is out of scope unless approved).
   - Trade-off to mention in the report: each part starts with a cold model state, so the first moments of a part may be less cleaned. Judge by listening.
6. **No progress from the tool -> estimate it.** Show an indeterminate or time-based progress using `ETA = duration * RTF`, with default RTF 0.20 (mono) / 0.27 (stereo) and a running average of RTFs measured on previous runs (store in settings). With `-v`, the last line gives the real RTF; optionally parse it for calibration.
7. **Concurrency:** default 1 worker. Allow 2 as an advanced setting (measured speedup only 1.37x, and RAM doubles). Never exceed 2.
8. **Treat clipping WARNs as one note**, never an error: count the `Possible clipping detected` lines and log one friendly message ("input is clipping, N events"). Keep only the last ~20 non-clipping stderr lines for diagnostics.

---

## 4. Phase 0: Discovery (read-only, then report)

Read the codebase and answer briefly:
1. Entry points, GUI framework (Tkinter / PyQt / PySide / CLI / other), main modules.
2. **Pipeline map:** how input becomes output; where audio is extracted; where splitting happens (time, silence, size); which intermediate files exist; final audio/video formats and codecs.
3. How ffmpeg/ffprobe are located and invoked.
4. Settings persistence, logging, progress, cancel, threading model.
5. Output naming conventions (e.g. `part_NNN.wav`, `*_parts` folders).
6. **Where the hook goes** (apply Section 3 rule 5; confirm unit lengths in practice) and how long a typical part is.
7. For video outputs: how the audio track is produced/muxed today.

Do not edit any file in this phase.

---

## 5. Phase 1: Smoke tests (short; the full probe is already done)

Use a throwaway script outside the app. Use a real speech file in the user's real format (48 kHz stereo `pcm_s24le`) and one 44.1 kHz mono file.
1. Run the Section 3 rule 3 procedure (pad -> `-D` -> trim) on both files.
2. Check: output length equals the source length in samples (+-1 sample); cross-correlation lag between source and output is 0 (+-2 samples) at `-a 100`; the last 100 ms of the output is not silent or clicking.
3. Check that source channel count is preserved and that the source rate is restored when requested.
4. Report the measured RTF and peak RAM for one stereo part of realistic length.

If any check fails, report before continuing and propose a fix.

---

## 6. Phase 2: Wrapper module (`denoise.py`)

```python
class DenoiseError(RuntimeError): ...
class DenoiseSkipped(Exception): ...        # e.g. auto_convert off and source not conformant

def find_deep_filter(configured_path: str | None = None) -> str | None: ...
def get_version(exe: str) -> str | None: ...            # runs: exe -V  -> "deep_filter 0.5.6"
def denoise_wav(src, dst, strength_db: float = 100, *,
                exe: str, ffmpeg: str, ffprobe: str,
                auto_convert: bool = True,
                restore_original_rate: bool = True,
                downmix_mono: bool = False,
                cancel_event=None, log=None) -> dict: ...   # returns {"rtf":..., "clip_events":...}
```

**Locating the exe:** explicit path from settings (if it exists) -> `shutil.which("deep-filter")` -> `shutil.which("deep-filter-0.5.6-x86_64-pc-windows-msvc")`. A process only sees PATH changes made before it started, so keep the explicit-path setting and log the resolved path. Cache the result; offer "Re-detect". (On this machine it resolves to `F:\DeepFilterNet\deep-filter.exe`.)

**`denoise_wav` algorithm**
1. Validate `strength_db`: `<= 0` -> skip (feature off for this item); `> 100` -> 100. Reject NaN.
2. `ffprobe` the source: format, codec, sample rate, channels, duration. Compute `N48 = round(duration_s * 48000)`.
3. **Conformant** = WAV container + `pcm_s16le` + 48000 Hz. If not conformant and `auto_convert=False` -> raise `DenoiseSkipped` with a clear message (e.g. "File is 48 kHz / 24-bit; enable automatic conversion to use noise reduction"). Never pass a non-conformant file to `deep-filter`.
4. Create `tempfile.TemporaryDirectory(prefix="dfn_")`. Build `tmp/in.wav` with ffmpeg (`-nostdin -y -vn`), filter chain `aresample=48000,apad=pad_dur=0.1` (add `pan=mono|c0=0.5*c0+0.5*c1` only if `downmix_mono`), codec `pcm_s16le`. Keep the channel layout otherwise. (For a conformant source this step only pads.)
5. Run `[exe, "-D", "-a", f"{strength_db:g}", "-o", str(tmp/"out"), str(tmp/"in.wav")]` with `Popen`, `stdin=DEVNULL`, stdout/stderr redirected to files in `tmp`, `creationflags=CREATE_NO_WINDOW` for GUI apps. Poll so `cancel_event` can kill the process promptly (no orphan `deep-filter.exe`). Record wall time -> RTF.
6. Success = exit code 0 **and** `tmp/out/in.wav` exists and is larger than a WAV header. Map failures: rc 101 (or `TooWide`) -> "unsupported bit depth"; rc 1 -> include the `Caused by:` line; rc 2 -> internal bug (bad arguments). Include the last ~20 stderr lines in `DenoiseError`.
7. Count clipping WARN lines; log one note (rule 8 in Section 3).
8. **Finalize with one ffmpeg call:** `atrim=end_sample=N48` (and `apad`/`atrim` safeguard so the length is exactly `N48`), then `aresample=<original rate>` if `restore_original_rate` and the source rate differs, then encode to the app's final audio format/codec. Before trimming, log a warning if the output length differs from `padded_length - 1440` by more than 2 samples (indicates changed tool behavior).
9. Write `dst` atomically (`dst.tmp` then `os.replace`). Temp dir is cleaned automatically, also on cancel and exceptions.

**Always pass `-D`.** The 1440-sample constant is a named constant (`DF_DELAY_SAMPLES = 1440`) with a comment pointing to the measurement.

---

## 7. Phase 3: Pipeline integration

- Hook at the stage chosen in Phase 0 (Section 3 rule 5). Reuse the app's progress, cancel, and logging mechanisms.
- **Audio outputs:** replace the audio with the denoised version, keeping existing naming (do not rename `part_NNN.wav`).
- **Video outputs:** copy the video stream untouched and replace only the audio (`-map 0:v -map 1:a -c:v copy -c:a <existing codec>`), keeping the app's existing trim/`-shortest` behavior. Never re-encode video for this feature. Because length is preserved (Section 3 rule 3), sync must hold; verify.
- **Batches:** sequential by default; optional `denoise_workers` (1 or 2).
- **Per-item failure** (`DenoiseError` / `DenoiseSkipped`): log it, fall back to the **un-denoised audio for that item** with a visible warning, unless the user chose "stop on error". Never silently produce a file that looks denoised but is not.
- **`deep-filter` missing:** the feature is disabled in the UI with a one-line explanation; the rest of the app works.

---

## 8. Phase 4: Settings and UI

Settings (use the app's existing mechanism):
- `denoise_enabled` (bool, default `false`)
- `denoise_strength_db` (number, `1..100`, default **100** = tool default)
- `denoise_auto_convert` (bool, default `true`)
- `denoise_restore_original_rate` (bool, default `true`)
- `deep_filter_path` (str, optional override)
- `denoise_workers` (int, 1 or 2, default 1; advanced)
- `denoise_rtf_history` (list of recent measured RTFs per channel count, internal)

UI (match existing style and localization):
- Checkbox **Reduce noise (DeepFilterNet)**.
- Strength control for `-a`: numeric field or slider `1..100`, with presets as starting points (measured on one lecture recording, noise-floor change / speech level change):
  - **Light = 6** (about -6 dB noise, speech about -2.6 dB)
  - **Medium = 12** (about -12 dB, speech about -4.1 dB)
  - **Strong = 20** (about -20 dB, speech about -5.2 dB)
  - **Very strong = 30** (about -30 dB, speech about -5.6 dB)
  - **Maximum = 100** (about -54 dB, speech about -5.9 dB)

  Hint text: "Lower keeps more of the original sound and room noise; higher removes more noise. Values above about 70 sound almost the same as 100. Try a short sample first." `0` is not offered (the checkbox is the off switch).
- Checkbox **Convert automatically to 48 kHz / 16-bit if needed** (`denoise_auto_convert`, on by default). Hint: "DeepFilterNet cannot read 24-bit audio. If this is off, such files are left un-denoised."
- Checkbox **Restore original sample rate after denoising** (`denoise_restore_original_rate`, on by default; enabled only when auto-convert is on).
- Status line: `deep-filter 0.5.6 found at <path>` or `not found`, with "Browse..." and "Re-detect".
- Progress/ETA text based on `duration * RTF` (Section 3 rule 6), e.g. "Reducing noise... about 4 min left".
- **Nice-to-have only if cheap:**
  - "Preview 30 s": process the first 30 s with the same function and open/play the result.
  - "Match loudness to the original" (default off): denoising lowers speech level by up to about 6 dB; offer a simple gain compensation (e.g. ffmpeg `loudnorm` or a measured RMS ratio).
  - "Mix down to mono before denoising" (default off, `downmix_mono`): stereo costs about 1.4x the time and 2x the RAM of mono, but it changes the output channel layout, so keep it opt-in with a clear label.
- Do not expose `--pf`, `--pf-beta`, `-m`, or the threshold options in the UI. An `extra_args` list in the config file only is acceptable (validated list of strings, appended after the managed arguments). `--pf` was not probed.

---

## 9. Phase 5: Robustness checklist

- All subprocess calls: list args, no shell, cancel support, captured output decoded as UTF-8 with `errors="replace"`.
- Temp dirs always cleaned in `finally`.
- Free-disk check before long jobs (48 kHz s16 stereo is about 11.5 MB per minute, and several temp copies exist at once).
- RAM guard from Section 3 rule 5 before starting a unit.
- Logging: full command line, exit code, wall time, RTF, clipping-event count, stderr tail on errors.
- No new third-party Python dependency unless truly necessary (psutil is acceptable for the RAM check if the app does not already have another way; justify in the report).

---

## 10. Acceptance tests (run them, report results)

1. **Regression:** feature off -> outputs identical to before (files, names, formats).
2. **Arabic path:** source under a path with Arabic letters, Arabic-Indic digits and spaces runs end to end.
3. **Real format:** 48 kHz stereo `pcm_s24le` source -> processed successfully (no panic), channel count preserved, length equals the source (+-1 sample).
4. **Other formats:** 44.1 kHz mono s16, 16 kHz mono, 32-bit float WAV, and an mp3 or mp4 source -> valid output with the same duration; with `restore_original_rate=True` the output rate equals the source rate.
5. **Alignment:** cross-correlation lag between source and output is 0 (+-2 samples) at strength 100 and 12.
6. **Strength:** 100 vs 12 differ measurably; 12 leaves a higher noise floor than 100 (compare quietest-10% RMS).
7. **Strength off:** `denoise_strength_db <= 0` -> `deep-filter` is not launched and the audio is untouched.
8. **Clipped input:** the clipping WARNs do not fail the job and appear as one note.
9. **Video part:** video untouched, audio replaced, duration equal, sync verified (compare durations; a transient/clap check if possible).
10. **Cancel:** mid-run cancel stops the process, leaves no temp files, no leftover `deep-filter.exe`.
11. **Not installed:** the app starts normally; the feature is disabled with a clear message.
12. **Bad input:** corrupt or empty input -> clear error; the batch continues (or stops, per setting). Verify the original audio is used as fallback.
13. **Same file names:** two parts with the same basename from different folders processed back to back (or with 2 workers) give correct, independent outputs.
14. **Conversion options** with a 24-bit source:
    - `auto_convert=True` + `restore_original_rate=True`: output at the source rate, same duration.
    - `auto_convert=True` + `restore_original_rate=False`: output at 48 kHz.
    - `auto_convert=False`: `deep-filter` is not launched; fallback to original audio with the explanatory warning.
    - A source already 48 kHz s16 WAV: no resampling or bit-depth conversion runs (only the padding step); verify in the log.
15. **Long unit:** one stereo unit of at least 30 minutes: report wall time, RTF, peak RAM; confirm the RAM guard triggers when the estimate exceeds the limit.

---

## 11. Out of scope
- Training, other models (only the optional `-m` path in config), GPU/PyTorch backend.
- Chunking long units with overlap/crossfade (only with user approval).
- Changing existing split logic, naming, or output formats.

---

## 12. Final report format
1. Pipeline map and the chosen hook stage (and why; typical part length).
2. Phase 1 smoke-test results.
3. Files added/changed, one line each.
4. Acceptance test results (Section 10), including anything not run and why.
5. Known limitations, deviations from this plan, and assumptions made.
