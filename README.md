# Predictive Methods from Nonlinear Dynamics

Clean, runnable Python implementations of standard methods from
nonlinear dynamics and Active Inference, extracted from a Gemini
methods export and repaired for publication.

## Source and status

Source: a Gemini chat export titled **"Incomplete Transmission
Evaluation"** (41 pages, Takens phase-space reconstruction,
Lyapunov / Rosenstein methods, Lorenz-63, Active Inference,
multi-agent POMDP, Lyapunov spectrum / Kaplan–Yorke, FTLE, Hessian
ridges).

**This is a PARTIAL export.** The source ends abruptly mid-table in
the Hessian ridge section — the "Diagnostic Thresholding Parameters"
table is cut off after the `ftle_min_percentile` row. The code itself
is complete through the ridge-extraction demo; only the tail of that
prose table is missing. Nothing has been invented to fill the gap,
and the truncation is marked `INCOMPLETE` in `ftle.py`. Do not
describe this repo as a complete paper or a complete method set.

Second source: a Gemini chat export titled **"Telemetry Phase Space
Reconstruction Pipeline"** (Takens delay embedding, AMI delay
selection, False Nearest Neighbours with cKDTree, multivariate
zero-copy strided embedding, Theiler-window-corrected FNN with PCA,
sliding-window twRQA, Welford bifurcation detector with Composite
Anomaly Index, async ZeroMQ streaming worker; test benches use
Lorenz-63 and Rössler systems). The scalar AMI/FNN blocks in that
export duplicate `takens_embedding.py` and were not copied again;
the scalar twRQA in it overlaps `realtime_rqa.py` — what was added
is the multivariate/Theiler/PCA pipeline, the PCA-projected twRQA
with the Welford detector, and the ZeroMQ worker
(`theiler_fnn_pca.py`, `twrqa_bifurcation.py`,
`zmq_streaming_worker.py`). The ZMQ worker re-defines the twRQA /
detector classes inline so it runs standalone.

The export was `pdftotext` output: indentation was mangled, form
feeds appear, and some lines were split (e.g. a `def` line ending in
`->` with the return annotation on the next line). Indentation and
split lines were repaired so every module passes
`python3 -m py_compile`. Where a print statement was split across
lines, it was reconstructed in the obvious form matching the
surrounding code. One structural repair is documented in
`spectrum.py`: the exponent-calculation block was indented as if
inside the integration loop in the extracted text; it sits after the
loop here, the only reading under which the code runs and matches
the documented formula.

## Contents

| Module | What it contains |
|---|---|
| `takens_embedding.py` | `TakensEmbedding` (AMI + false nearest neighbours, phase-space reconstruction) and `PhaseSpaceReconstructor` (the second, near-identical implementation given in the source) |
| `rosenstein_mle.py` | `RosensteinMLE` — maximal Lyapunov exponent (Rosenstein, Collins & De Luca 1993), with Theiler-window exclusion |
| `lorenz_harness.py` | Lorenz-63 integrator (`lorenz63`, `generate_lorenz_data`) + `RosensteinBenchmark` test harness; theoretical λ₁ ≈ 0.9056 |
| `active_inference.py` | `AdversarialActiveInference` (variational / expected free energy, softmax policy selection) + `MultiAgentActiveInferencePOMDP` (defender actions warp the adversary's A-matrix) |
| `spectrum.py` | `LorenzSpectrumAnalyzer` — full Lyapunov spectrum (Benettin QR algorithm), Kaplan–Yorke dimension, Kolmogorov–Sinai entropy |
| `ftle.py` | `DoubleGyreVectorField`, `FTLEEngine` (forward/backward FTLE, LCS) + `FTLERidgeExtractor` (Hessian / Steger subpixel ridge extraction — **source incomplete, see above**) |
| `rqa.py` | `RecurrenceQuantificationAnalysis` — recurrence plot, RR / DET / L / Lmax / ENTR / LAM / TT / Vmax (extra module: complete in the source, preserved rather than dropped) |
| `realtime_rqa.py` | `RealTimeRQA` — sliding-window RQA with a critical-slowing-down tripwire (composite CSD index, 3σ alarm) and May harvesting-model test harness (extra module, same note) |
| `theiler_fnn_pca.py` | `MultiStreamPhaseSpaceReconstructor` (multivariate zero-copy strided embedding) + `MultivariateTheilerPCAFNN` (Theiler-window-corrected FNN with PCA projection) + Lorenz-63 test bench — from "Telemetry Phase Space Reconstruction Pipeline" |
| `twrqa_bifurcation.py` | `RealTimeSlidingtwRQA` (PCA-projected sliding-window twRQA, RR/DET/Lmax/DIV/ENTR/LAM/TT) + `RealTimeBifurcationDetector` (dual-window Welford tracker, Composite Anomaly Index, NOMINAL / EWS_WARNING / BIFURCATION_TRIP) + limit-cycle → Lorenz and limit-cycle → Rössler test benches — same source |
| `zmq_streaming_worker.py` | Self-contained async ZeroMQ worker: SUB telemetry on `tcp://127.0.0.1:5555`, PUB bifurcation alerts on `tcp://127.0.0.1:5556` (msgpack), with mock publisher + alert listener — same source |

## Requirements

- Python 3.10+
- `numpy`, `scipy` (see `requirements.txt`)
- `pyzmq`, `msgpack` — only for `zmq_streaming_worker.py`

```bash
pip install -r requirements.txt
```

## How to run

Each module is self-contained and runnable:

```bash
# Takens embedding smoke demo (noisy sinusoid)
python3 takens_embedding.py

# Rosenstein MLE (usage notes only; run the benchmark for a live test)
python3 rosenstein_mle.py

# Lorenz-63 benchmark: direct 3D space vs. Takens reconstruction.
# NOTE: the source claims lambda_1 = 0.9124 (3D) / 0.8982 (reconstructed)
# with Theiler window 16. Re-running the code as extracted gives
# lambda_1 ≈ 1.66 (3D) / ≈ 1.45 (reconstructed) and Theiler window 38
# (the Lorenz x-autocorrelation genuinely first drops below 1/e at lag
# 38). See "Verification notes" below.
python3 lorenz_harness.py

# Active Inference demos (single-agent policy shift + multi-agent POMDP)
python3 active_inference.py

# Full Lyapunov spectrum (takes a while: 300 s at dt=0.005, RK4 + QR)
# Verified: spectrum ≈ (+0.9096, -0.0058, -14.5705), D_KY ≈ 2.0620
python3 spectrum.py

# Double-gyre FTLE (forward + backward) and synthetic-ridge extraction
python3 ftle.py

# Recurrence quantification (small synthetic demo)
python3 rqa.py

# Sliding-window RQA, fold-bifurcation tipping-point simulation
# (alarm fires, but later than the source claims — see below)
python3 realtime_rqa.py

# Theiler-corrected FNN with PCA on 3-channel Lorenz-63 telemetry.
# Verified: FNN 8.32% at m=1, 0.00% at m>=2 (full-state 3-channel
# input needs little embedding). Runs in ~2 s.
python3 theiler_fnn_pca.py

# PCA-projected twRQA: limit-cycle -> Lorenz regime switch
# (RR 3.5% -> ~0.3%, DIV 0.004 -> 0.056, L_max 276 -> 18),
# then the tipping demo: BIFURCATION_TRIP fires at frame 1650,
# inside the critical-slowing-down zone (1600-2000), before the
# Roessler-chaos phase at 2000. See "Verification notes" item 4.
python3 twrqa_bifurcation.py
python3 -c "import twrqa_bifurcation; twrqa_bifurcation.demo_twrqa_transition()"

# Async ZeroMQ streaming worker (needs pyzmq + msgpack; runs until Ctrl-C)
# Telemetry in on tcp://127.0.0.1:5555, alerts out on tcp://127.0.0.1:5556.
# NOT runtime-tested here (pyzmq/msgpack not installed in this environment).
python3 zmq_streaming_worker.py
```

To verify everything compiles:

```bash
python3 -m py_compile *.py
```

## Scope note

This repo contains textbook nonlinear-dynamics and Active Inference
methods only — phase-space reconstruction, Lyapunov exponents,
recurrence analysis, FTLE/LCS, and a published free-energy formalism.
It contains no intelligence, operational, targeting, or
collection content, no data, and no credentials.

## Verification notes (re-running the extracted code)

All eight modules pass `python3 -m py_compile` and their demos were
executed. Two things in the source did **not** reproduce, and are
recorded here rather than papered over:

1. **Lorenz/Rosenstein benchmark.** The source's printed output
   claims λ₁ = 0.9124 (3D) / 0.8982 (reconstructed) with Theiler
   window 16. Re-running the extracted code gives λ₁ ≈ 1.66 (3D) /
   ≈ 1.45 (reconstructed) and Theiler window 38 — the Lorenz
   x-autocorrelation genuinely first drops below 1/e at lag 38.
   The divergence curve's early slope is steep (~1.6) and decays
   toward ~1.05, and the max-R² window picker locks onto that early
   segment. The system itself is correct: `spectrum.py`
   independently recovers λ₁ ≈ 0.9096, spectrum sum −13.6667 (trace
   error ~6e-6), D_KY ≈ 2.0620. So the Rosenstein harness code is
   faithful to the source, but the source's claimed benchmark
   numbers appear to be illustrative rather than measured. Treat
   the harness as a starting point, not a validated benchmark.
2. **Multi-agent POMDP demo.** The source's printed payoff table
   (e.g. jamming → recon 62.1%) does not follow from its own
   A-matrix: under jamming all A-matrix columns are identical, so
   G is policy-independent and the policy distribution is uniform
   (33.3% each). Re-running gives Passive −8.86, Jamming −2.53,
   Deception −3.30, so the best action is still Jamming, but the
   printed percentages in the source are inconsistent with its code.
4. **Phase-space pipeline demos (new modules).** The Lorenz-63 FNN
   bench runs clean: FNN 8.32% at m=1, 0.00% at m>=2 — expected,
   since the input is the full 3-channel Lorenz state, not a scalar
   series. The twRQA transition demo shows a clear regime signature
   (limit cycle: RR ~3.5%, DET ~99.8%, L_max 276; Lorenz phase:
   RR ~0.3%, DIV 0.004 -> 0.056, L_max 276 -> 18). The tipping
   demo's BIFURCATION_TRIP fires at frame 1650, inside the
   critical-slowing-down zone (1600-2000) and before the Rössler
   chaos phase at 2000; DET falls 100% -> ~52% across the
   transition. Two caveats: (a) the source printed no FNN or twRQA
   tables for these benches, so there are no claimed numbers to
   reproduce — the values above are measured here, not confirmed
   against the source; (b) DET reads 0.0% in the late Rössler
   frames and a NumPy overflow warning appears, suggesting the
   Euler-integrated Rössler test data itself degrades late in the
   run — treat the late-phase detector output as suspect, not as a
   validated result. The ZeroMQ worker was NOT runtime-tested
   (pyzmq/msgpack not installed in this environment); py_compile
   only.

3. **Sliding-window RQA demo.** The CSD alarm fires (CSD index
   3.57), but at t ≈ 390 s — after the collapse — not in the
   source's claimed t ∈ [180, 240] s window, and it is driven by a
   DET collapse (18.4% vs. 59.8% baseline) while TT/LAM sit below
   baseline, the opposite of the claimed signature. Again, the code
   is faithful; the claimed diagnostic profile does not reproduce.

## Corrections made while extracting

- **FNN delta-coordinate slice (both Takens classes).** The source
  as transcribed computes the next-dimension coordinate delta as
  `data[M + (dim-1)*tau:]`, which has length `tau` and cannot
  broadcast against the M nearest-neighbour indices — the code
  raised `ValueError` on any input. Corrected to
  `data[dim*tau : dim*tau + M]`, i.e. row i's (dim+1)-th coordinate
  `data[i + dim*tau]`, the standard Kennel–Brown–Abarbanel form.
  The fix is marked with a NOTE comment at both sites.
- **Split lines / indentation** throughout (see "Source and status").
- **`spectrum.py`:** the exponent-calculation block moved out of the
  integration loop (see its module docstring).
- **Strided-embedding element stride (new modules).** The source's
  twRQA test benches and the ZMQ worker compute
  `itemsize = <2-D array>.strides[0]` (= channels x 8 bytes) but
  build the `as_strided` view on a 1-D contiguous column (element
  stride 8 bytes). `as_strided` does no bounds checking, so the
  source form read out-of-bounds memory: the first run here
  produced RR = 100% / DET = 100% on every window, then an SVD
  non-convergence failure on re-run. Fixed to
  `<array>.dtype.itemsize` at the three sites (marked NOTE in
  `twrqa_bifurcation.py` and `zmq_streaming_worker.py`); after the
  fix the demos produce sane, stable numbers (see item 4 above).
  The library classes in `theiler_fnn_pca.py` take their stride
  from 1-D arrays and were already correct.
- **Missing imports (new module).** The `twrqa_bifurcation.py`
  assembly dropped the source's import block; the imports
  (`numpy`, `scipy.spatial.distance.cdist`, `dataclasses`,
  `collections.deque`, `enum`) were restored at the top of the file.

## Omissions

Nothing was omitted as too garbled: every code block in the source
was reconstructable from context. The only missing material is the
tail of the prose "Diagnostic Thresholding Parameters" table at the
very end of the partial export (see "Source and status" above), which
contains no code.
