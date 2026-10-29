# Task: Implement DeepFilterNet noise reduction in my Python app, and validate the plan while doing it

## What you have
- The app's source code is in your current working directory (likely `F:\projects\Python Projects2\video-splitter-merger`; verify). It splits audio/video on Windows.
- A written plan: **`deepfilternet_integration_plan.md` (v2)**. It was written by another AI assistant from documentation plus a measured probe of `deep-filter` 0.5.6. **Read it fully before doing anything.** If the file is not in your workspace, stop and ask me to provide it.
- `deep-filter` is installed and on the PATH (resolved earlier to `F:\DeepFilterNet\deep-filter.exe`). `ffmpeg`/`ffprobe` are installed.
- A real lecture sample (read-only, do not modify or move): `F:\projects\Python Projects2\video-splitter-merger\deep-filter-test` (48 kHz, stereo, `pcm_s24le`, about 353 s). Verify it exists; if not, tell me and use any real speech WAV you can find in `tests`.

## Two goals (both matter equally)
1. **Implement the feature** following the plan.
2. **Test the plan itself.** I want to know whether the plan's author understood the problem correctly. Treat the plan as a hypothesis, not as an order. Where reality contradicts it, say so with evidence. Do not silently work around a plan error, and do not silently "improve" the plan either.

## Rules
1. Work on a **new git branch** (`feature/deepfilternet`) if the project uses git; otherwise first copy the project to a backup folder and tell me where. Never delete or rewrite existing user data, tests, or outputs.
2. Follow the plan's phases in order: Phase 0 (discovery), Phase 1 (smoke tests), then Phases 2 to 5 (module, integration, UI/settings, robustness), then Section 10 (acceptance tests).
3. **Checkpoint after Phase 1.** If a smoke-test check fails, or Phase 0 shows that a plan assumption cannot hold in this codebase (for example the chosen hook stage is impossible), **stop and report** with evidence and a proposed fix. If everything is consistent, continue without waiting for me.
4. Do not add dependencies other than what the plan allows. Do not use `shell=True`. Use `pathlib`. Match the existing code style, GUI framework, config mechanism, logging, and localization approach.
5. Keep the diff small and local: one new module (`denoise.py` or the closest equivalent to the codebase's layout) plus minimal hooks. With the feature disabled, behavior must be identical to before.
6. Run the project's existing tests before and after. Report any new failures. Add tests for the new module where the project has a test setup (pure-logic tests that do not need the exe are enough; exe-dependent tests may be marked to skip when `deep-filter` is missing).
7. **Never guess.** If something is unknown, test it. If you cannot test it, mark it `UNTESTED` with the reason.

## Claims in the plan that I suspect are the riskiest: test each one explicitly
| ID | Claim from the plan | How to check |
|---|---|---|
| C1 | Padding the input with silence, running `deep-filter -D`, then trimming to the original length gives an output of **exactly** the original sample count, time-aligned (lag 0 within +-2 samples) with the source, with no lost tail audio. | Phase 1; use the real 24-bit stereo sample and a 44.1 kHz mono file. Cross-correlate and compare lengths. Also compare the last 100 ms. |
| C2 | The filter chain `aresample=48000,apad=pad_dur=0.1` yields exactly 4800 padded samples at 48 kHz, and `atrim=end_sample=N48` yields exactly `N48` samples, on this machine's ffmpeg (version 9.x). | Count samples with `ffprobe` (`-count_frames` or `nb_samples`). |
| C3 | After `-D` the output length is `padded_length - 1440` samples (within 2). | Measure for several lengths, including very short files (under 1 s) and files shorter than the pad. |
| C4 | Converting `pcm_s24le` to `pcm_s16le` before `deep-filter` avoids the Rust panic (exit 101) the plan documents for 24-bit input. | Run both (direct call vs converted) on a copy; record exit codes. |
| C5 | Redirecting `deep-filter`'s stdout/stderr to files (instead of pipes), with `CREATE_NO_WINDOW` and a cancel poll loop, works from the app's real threading model, and cancel leaves no orphan `deep-filter.exe` and no temp dir. | Run it inside the app's real worker context; check Task Manager equivalents via `psutil` or `tasklist`. |
| C6 | Per-part denoising after splitting is the right hook for this codebase, and part lengths are small enough for the RAM guard (`duration * channels * 0.45 MB`). | Phase 0 evidence: typical part length; Phase 1: measure peak RAM for a realistic stereo part and compare to the formula. |
| C7 | `-a 0` / strength `<= 0` should be treated as "off" and skipped. | Confirm the app never launches the tool in that case. |
| C8 | Presets (6 / 12 / 20 / 30 / 100) correspond to noise-floor reductions of about N dB (up to about 50), measured on one recording. | Re-measure on the real sample at N = 6, 12, 20, 30, 100 (quietest-10% of 20 ms frames RMS in dBFS) and compare to the plan's table. Differences larger than about 3 dB should be reported. |
| C9 | Mono RTF about 0.15 to 0.19 and stereo about 0.26 on this machine; ETA = `duration * RTF`. | Measure on one stereo part; report the actual RTF. |
| C10 | Running two denoise workers in parallel is only about 1.37x faster. | Optional; run only if the app gets a worker setting. |
| C11 | Two parts with the **same file name** from different folders, processed back to back or in parallel, never interfere (private temp dir per call). | Section 10, test 13. |
| C12 | The video path (copy video, replace audio only) keeps A/V sync because the length is preserved. | Compare stream durations; transient check if possible. |

Add any other plan statement that turns out to be wrong, ambiguous, missing, or impossible to follow.

## Deliverable
When finished, **print the full report as your final reply** and also save it as `deepfilternet_implementation_report.md` in the project root. Maximum about 250 lines. Structure:

1. **Summary** (max 10 lines): what works, what does not, branch/backup location.
2. **Plan validation table:** one row per claim C1 to C12 plus any extra, with columns `ID | Verdict (CONFIRMED / PARTIAL / REFUTED / UNTESTED) | Evidence (numbers, commands, exit codes)`.
3. **Plan defects:** a numbered list of everything in the plan that was wrong, unclear, missing, or contradictory (quote the plan section number), each with what you did instead and why.
4. **Pipeline map and hook decision:** where the hook was placed, typical part length, why.
5. **Phase 1 smoke test results** (numbers).
6. **Section 10 acceptance tests:** a compact table with `PASS / FAIL / NOT RUN` and a one-line reason.
7. **Files added/changed** with one line each.
8. **Known limitations and open questions.**
9. **Measured numbers worth knowing:** RTF, peak RAM, any difference from the plan's tables.

Be specific and honest. A report that says "the plan was wrong here" with evidence is more useful to me than one that says everything passed.
