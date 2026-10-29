# DeepFilterNet integration — final report

Feature branch `feature/deepfilternet`, commit `d13ad4c`. Tool: `deep-filter.exe`
(DeepFilterNet3 CLI, v0.5.6) at `F:\DeepFilterNet\deep-filter.exe`; ffmpeg/ffprobe
9.0.2 from the project's `bin\`. Every number below was measured on this machine.

---

## 1. Summary

The "Reduce noise (DeepFilterNet)" feature is implemented end to end: optional
per-method step that runs after the splitter produces its parts, denoising each
part (per audio stream), leaving video/subtitles untouched, and falling back to
the un-denoised audio on any failure. **Full pytest suite: 197 passed, 1
skipped** (154 baseline before the change + 43 new + the previously-dead
subtitle path in the shared helper now exercised). GUI end-to-end was not
executed in this headless session; the GUI code is compiled and unit-verified.

## 2. Pipeline map and hook stage (plan §12.1)

- App: Python + CustomTkinter GUI (`src/gui`), core logic in `src/core`
  (`splitter.py`, `merger.py`, `ffmpeg_manager.py`, `runner.py`).
- Splitting produces part files under `<output>_parts/` with metadata
  `(path, duration)`; audio merges/parts keep source container/codec families.
- **Chosen hook: per part *after* splitting, inside the app worker**
  (`app.py::_start_split_task`), not inside the splitter: the feature stays
  optional/independent, gets natural per-part progress and cancel granularity
  (plan §3 rule 5), and typical split parts are bounded-length units (minutes,
  not hours) which keeps peak RAM ≈ 0.5–1.4 GB per concurrent unit.
- Video parts keep video/subtitles by stream copy (`-c:v copy -c:s copy`); every
  audio stream is replaced by its own denoised version, with the original
  encoder and (for lossy sources) bitrate preserved. Atomic replacement via
  `.tmp` + `os.replace`; post-remux verification (stream counts + duration
  ≤ 0.15 s diff) protects against partial writes.

## 3. What was delivered (plan §12.3)

| File | Change |
|---|---|
| `src/core/denoise.py` | **NEW** (906 lines): `denoise_wav` (strength validation → probe → conform+pad to 48 kHz s16 → `-D` run with stdout/stderr to files, cancel polling, `CREATE_NO_WINDOW` → exit-code mapping 101/1/2 → clipping-warn counter → `atrim/apad` finalize to exact N48 (+rate restore) → atomic replace), `denoise_parts` (workers 1–2, RAM guard, per-part temp dir, per-stream extraction/encoding), RTF helpers, RIFF sample counter, disk-space check. No `shell=True`, no new runtime dependencies. |
| `src/gui/split_view.py` | +254 lines: denoise section (checkbox, strength entry with presets 6/12/20/30/100, auto-convert and restore-rate checkboxes, 1/2 workers radio, tool status line with Browse/Re-detect), settings collection, progress line, success-summary extension. |
| `src/gui/app.py` | +39 lines: hook in the split worker, `DENOISE_PROGRESS` message bridge, cancel wiring, session RTF history (cap 8 per channel count) for ETA. |
| `tests/test_denoise.py` | **NEW**: 44 tests (14 pure helpers, 4 validation gate, 3 "real ffmpeg up to tool launch" spies, 12 end-to-end, 10 parts pipeline, 1 version). Tool-dependent ones are marked `deepfilter_exe` and skip when the exe is absent. |
| `tests/conftest.py` | mp4-family subtitle encoder fix (`mov_text`): ffmpeg 9 rejects the `srt` codec in mp4, so `include_subtitle=True` produced generation failures; nobody exercised that path before. |

## 4. Phase 1 smoke tests (plan §5) — all pass

(Script `F:\dfn_probe_tmp\smoke_results.json`, 20 checks; the single "lag=-15"
entry was a probe bug — it compared the 44.1 kHz-restored file against 48 kHz
windows — re-verified lag 0 in the 48 kHz domain.)

| Check | Result |
|---|---|
| Direct 24-bit input → **exit 101** with `libDF\src\wav_utils.rs:87:39 TooWide` | confirmed (also from the absolute original path) |
| pad → input = **N48 + 4800 samples exactly** (20.6-min + 0.3 s + 0.05 s files) | exact |
| DF output = padded − 1440 samples exactly (all three source files) | exact |
| trim → dst = N48 exactly; channels preserved (2, and 1 for mono) | exact |
| Lag at `-a 100` = **0 samples** | confirmed |
| Restore rate: 44.1 kHz output rate restored, samples = N44 exactly | confirmed |
| Real format: 48 kHz stereo `pcm_s24le`, 1235.48 s | processed, no panic |

## 5. Verdicts on the measured-behavior claims (C1–C12)

| # | Claim (restated) | Verdict | Evidence |
|---|---|---|---|
| C1 | Without `-D` output is delayed 1440 samples; with `-D` lag is 0 and the 1440-sample tail is dropped | **CONFIRMED** | smoke lag 0 mono+stereo; pytest exact `df_out_samples == padded−1440`; acceptance NCC 0.99/0.83 at zero shift |
| C2 | The pad→`-D`→trim procedure restores the exact original length; padded input is exactly N48+4800 | **CONFIRMED** | smoke (incl. 0.3 s/0.05 s files); acceptance `samples_out == samples_in == 2,880,000` |
| C3 | DF output length == padded_length − 1440, independent of strength and rate | **CONFIRMED** | sweep `df_out_len == 59,306,400` for every N∈{6..100}; smoke for 0.3 s/0.05 s |
| C4 | s24/s32 direct → exit 101 `TooWide`; converted s16 works; rerun is bitwise identical | **CONFIRMED** | smoke exit-101 probe; pytest e2e + `sha256(out1)==sha256(out2)` |
| C5 | Tool is meaningfully faster than real time; ETA via `duration*RTF` with defaults 0.20/0.27 holds | **CONFIRMED** | measured RTF 0.2259 (20.6-min mono), 0.2386 (stereo, a=100); plan defaults stay sane |
| C6 | RAM ≈ 0.45 MB/s per channel; guard = min(4096 MB, 25 % available) bounds units | **PARTIAL** | mono exact: 492 MB ≈ 0.40 MB/s. Stereo measured 1400 MB vs estimate 1112 MB (≈0.57 MB/s per channel → the model under-estimates stereo ~26 %). Guard math verified (limit 1543 MB on this machine > full-lecture estimate 1111 MB); forced-refusal covered in pytest |
| C7 | Per-part integration keeps A/V sync (video copied, audio replaced in place) | **CONFIRMED** | acceptance: h264 stream copy intact, duration 60.0→60.0, audio codec/rate kept; pytest verify ≤ 0.15 s |
| C8 | Strength monotonicity of noise-floor reduction; ≥ 100 clamps; formats differ measurably | **CONFIRMED** | full sweep noise-floor −25.6→−78.9 dBFS monotonic, speech-ceiling −17.2; N6 vs N100 on real material differ 5.83 dB (same length both runs). (The plan's own claim of 100/101/150 identical outputs was measured in the plan's probe, not re-run here) |
| C9 | One file per call avoids the same-basename silent corruption | **CONFIRMED** | pytest: same-basename parts sequential and with 2 workers give independent correct outputs; tool saw only ASCII `in.wav` each call |
| C10 | 2 workers give a real but < 2× speedup | **CONFIRMED** | C10 probe: sequential 37.04 s vs parallel 21.25 s → **1.74×** (2×60 s stereo) |
| C11 | Cancel kills the tool promptly, no orphan processes/temp files | **CONFIRMED** | acceptance: cancel latency **0.10 s**, dst absent, zero own-ppid orphans; pytest ppid-scoped zero; temp dirs are `TemporaryDirectory` (auto-cleaned) |
| C12 | Clipping WARNs never fail the job and become one note | **CONFIRMED** | real lecture: 5 clip events → note «ملاحظة: … مشبع» and rc 0; stderr redirected to files (7.2–7.4 KB per 20.6-min run ≈ plan's 322 KB/h figure) |

## 6. Acceptance tests (plan §10, 15 items)

`F:\dfn_probe_tmp\acceptance_console.txt` (items A1–A10 map onto them).

1. **Regression, feature off** — app calls `denoise_parts` only when the
   checkbox is enabled (guarded in `app.py`); strength ≤ 0 leaves media
   byte-identical (pytest). GUI end-to-end split not executed in this session.
2. **Arabic path** — real slice inside `…\درس الضوضاء\شريط 1.wav` processed
   end to end (A1). ✔
3. **Real format** — 48 kHz stereo `pcm_s24le`: success, codec preserved,
   duration exactly 60.0 s (A1/A2). ✔
4. **Other formats** — 44.1 kHz s16 (restored ± 0–2 samples), 16 kHz mono,
   float32 WAV, mp3 (bitrate 128k preserved) — pytest + A4/A5. ✔
5. **Alignment** — zero-lag NCC on real material: 0.990 at N6, 0.830 at N100
   (whole 55 s window, no shift search needed; smoke already showed lag 0 at
   sample resolution) (A6). ✔
6. **Strength** — real material overall −15.33 → −19.17 (N6) → −25.00 dBFS
   (N100); Δ(N6, N100) = 5.83 dB, monotonic sweep table (A6). ✔
7. **Strength off** — skipped in 0.001 s, file untouched, tool never launched
   (A10 + pytest with launch guard). ✔
8. **Clipped input** — real lecture triggers 5 clipping WARNs → single
   «ملاحظة» note, job succeeds (A1); pure synthetic sine produced none
   (pytest records this skip honestly). ✔
9. **Video part** — video `h264`→`h264` (copy), subtitles preserved, duration
   equal, audio `aac` 44.1 kHz (A7 + pytest, including a 2-audio-stream part). ✔
10. **Cancel** — thread freed, `InterruptedError` in 0.10 s, no output, no
    orphan (A8 + pytest). ✔
11. **Not installed** — one-line Arabic explanation and the feature disabled
    (pytest top-level message; UI status line + disabled checkbox). ✔
12. **Bad input** — corrupt `.wav` → per-part failure with reason, batch
    continues, original audio untouched (pytest). ✔
13. **Same names** — two parts with identical basenames processed back-to-back
    and with workers=2: independent, distinct, correct outputs (pytest). ✔
14. **Conversion matrix** — convert+restore / no-restore / auto-convert-off
    all pass as specified (pytest + A2/A3). Nuance: a conformant 48 kHz s16
    source still goes through one ffmpeg conform pass (needed for the 0.1 s
    padding); the finalize step adds resampling only when the source rate
    differs — i.e., no *output* resampling/bit-depth change for conformant
    sources. ✔ (with the noted no-op conform pass)
15. **Long unit** — the 20.6-min full lecture (stereo): wall 294.79 s,
    RTF 0.2386, peak RAM 1400 MB; mono: 279.15 s, RTF 0.2259, 492 MB. The
    guard *did not trigger naturally* (estimate 1111 MB < limit 1543 MB);
    triggering was verified by forcing the limit (pytest). Below the plan's
    "at least 30 minutes" size — **PARTIAL**.

## 7. Test suite status

- Pre-change baseline: **154 passed** (recorded when C: had free space).
- Post-change full suite: **197 passed, 1 skipped** (`tests/test_denoise.py`
  contributes 43 passed + 1 skipped).
- Environment note: C: on this machine is at 0 bytes free, so pytest's default
  `basetemp` breaks unrelated suites. Run the suite with
  `python -m pytest tests -q --basetemp=F:\…\tmp\pytest_bt` (F: drive).

## 8. Deviations from the plan, limitations, assumptions

**Deviations (each with the plan's wording):**

1. **No settings persistence exists in the app** (Phase 0 finding). Plan §8:
   "Settings (use the app's existing mechanism)" — none exists, so settings are
   session-only UI state and `denoise_rtf_history` lives in memory (cap 8, plan
   §8 listed it as an internal setting rev: "denoise_rtf_history (…, internal)").
2. **RAM-guard overflow doesn't ask the user.** Plan §3 rule 5: "If the app has
   no smaller unit available, ask the user how to proceed". The hook runs in a
   worker thread while the GUI is busy; a modal would block. Instead the item
   is refused with a clear Arabic reason and the batch continues with the
   un-denoised audio (plan §7 fallback semantics), plus the reason surfaces in
   the success summary.
3. **§8 "Nice-to-have only if cheap" items were dropped:** 30 s preview,
   loudness-matching, and a mixdown-to-mono checkbox. `downmix_mono` exists in
   the core API (tested at pipeline level) but has no UI.
4. `denoise_wav` gained **`temp_base`** (not in the plan §6 signature): the
   system temp dir on this machine's C: is full; the app/tests can pin the
   private temp dir to another drive.
5. **Per-stream multi-audio:** plan §7 assumed a single audio track; parts with
   two audio streams are handled per stream (tested).
6. **Bitrate preservation** for lossy parts (`-b:a` from the probe) — plan §7
   asked only for `-c:a <existing codec>`.
7. **Explicit `-f <muxer>`** for the `.tmp` finalize/remux outputs: ffmpeg 9
   cannot infer a container from `.tmp` (found as rc −22 during acceptance).
8. **`-nostats -loglevel error`** on every ffmpeg call: ffmpeg prints ~2
   stderr stats lines/s, which would fill the 64 KB pipe buffer the plan
   warned about (§1 rule 8) during long conform steps.
9. One **conform pass even for conformant sources** (plan §6 step 4) — the
   padding requires a re-encode; the output side does not resample/requantize.

**Limitations — everything not run stays UNTESTED (with its reason), nothing passed silently.**

- RAM model under-estimates stereo (0.57 vs 0.45 MB/s per channel): guard uses
  the estimate; on small-RAM machines a big stereo unit could pass estimate but
  still respect the 25 %-limit rule as an upper bound.
- Chunking long units with overlap/crossfade is out of scope (plan §11) and was
  not implemented.
- Each part starts with a cold model state (plan §3 rule 5 trade-off); not
  listening-evaluated here.
- Non-ASCII user temp dir would reach the tool and fail it (the tool saw only
  our ASCII temp file names in all measured paths); only tested with F: temps.
- GUI end-to-end (real windows) was not executed in this session — the UI code
  path is exercised only at signature level.
- Only deep-filter 0.5.6 / ffmpeg 9.0.2 measured; other versions re-derive the
  C-claims before use.

## 9. Reproducing the evidence

- Smoke facts: `F:\dfn_probe_tmp\smoke_results.json` (probe scripts
  `smoke1.py`, `C:\Windows\Temp\opencode\dfn_probe\`).
- Performance/strength/parallel sweep: `F:\dfn_probe_tmp\sweep_results.json` +
  `sweep.py` + `sweep_console.txt` (full 20.6-min lecture).
- Acceptance items: `F:\dfn_probe_tmp\acceptance.py` →
  `acceptance_console.txt` (real lecture slices on `F:\dfn_tmp_accept`, since
  removed; the source file is never modified).
- Test suite: `python -m pytest tests -q -p no:warnings --basetemp=<pre-created
  dir on F:>` (pytest does not create `--basetemp` itself; e.g.
  `F:\projects\Python_Projs_tmp\pytest_bt` after `mkdir`) — 44 tests in
  `tests/test_denoise.py` cover C1–C12 mechanically.
