# 📊 Differometor Baseline Benchmark and Workflow Report

Date: 2026-11-21  
Project: START Hack Munich 2026 — Faster Physics

## 📌 Conclusions and Results First

> ✅ **The original simulator runs successfully and passes the built-in checks.** The first complete evaluation took **26.224 seconds**; the median of three warm runs was **12.451 seconds**.  
> 🛠️ **No performance optimization has been implemented yet.** These measurements cover different timing scopes; they are not before-and-after optimization results.

| Question | Current answer |
|---|---|
| What have we completed? | Installation → starter example → saved baseline → new branch → core code review |
| Which number is the main competition metric? | First complete evaluation: **26.223708 seconds** |
| What does 12.451 seconds mean? | The median of three repeated evaluations after compilation |
| Why does the first evaluation take longer? | It includes loading, preparation, and compilation; execution time also varies |
| Have we achieved a speedup? | **No.** No modified version has been measured yet |
| What comes next? | Choose an optimization hypothesis, make a small change, then check correctness and timing |

### 📊 The Data at a Glance

![First evaluation, phase timings, and median calculation](baseline-explained.png)

**How to read the figure:** The top panel distinguishes the two timing scopes. The middle panel shows where time was spent during the first evaluation. The bottom panel sorts the three warm runs and highlights the median in green. The chart labels are in English; detailed explanations follow below.

### 🧭 Where Are We in the Process?

| Stage | Status | Outcome |
|---|---|---|
| ① Install the environment | ✅ Complete | The project can run locally |
| ② Run the starter example | ✅ Complete | The complete calculation pipeline works |
| ③ Measure the original version | ✅ Complete | 26.224 seconds / 12.451 seconds and PASS |
| ④ Create a branch | ✅ Complete | A separate branch is ready to track subsequent changes |
| ⑤ Read the code | 🔎 Started | Understand that `update` builds equations and `solve` solves them |
| ⑥ Modify and validate | ⏳ Not started | No candidate results yet |

### 🔄 Why Are There Two Different Times?

**First complete evaluation:** 📦 Load and prepare → ⚙️ Compile and execute the calculation stages → 📤 Retrieve results → **26.224 seconds**.

**Warm evaluation in the same process:** ♻️ Reuse prepared inputs and compiled programs → 🧮 Recalculate physical outputs and gradients → 📤 Retrieve results → **12.451 seconds, the median of three runs**.

These are conceptual workflows. The actual benchmark alternates compilation and execution by stage; the precise sequence and timings are retained in Section 5. It reuses the compiled program, not the previous physical answers.

### 🔢 Calculating the Median in Three Steps

| Step | Action | Results (seconds) |
|---|---|---|
| ① Record | Collect the three measurements in execution order | 13.607138, 12.180455, 12.450582 |
| ② Sort | Arrange them from smallest to largest | 12.180455 → **12.450582** → 13.607138 |
| ③ Select the middle | Take the second of the three values | **12.450582 ≈ 12.451** |

The first evaluation's **26.223708 seconds is excluded from this sorting**, because it includes preparation and compilation and therefore has a different timing scope.

---

The following sections retain all procedures, data, explanations, and reproduction commands from the original report, organized by topic.

## 1. ✅ Current Conclusions

Environment installation, the starter example, and the original baseline benchmark have been completed, and the benchmark's built-in correctness checks passed. The benchmark records that the source code was unmodified. The `optimize-simulator` branch was subsequently created, and core code review began. No performance optimization has yet been implemented or validated.

| Key metric | Original baseline |
|---|---:|
| First complete evaluation, including compilation and excluding validation | **26.223708 seconds** |
| Median warm evaluation, three runs | **12.450582 seconds** |
| Total tracing, lowering, and compilation | 8.155907 seconds |
| Correctness checks | PASS |

These are baseline measurements on this machine, not optimized results. They cannot be directly compared with results obtained on different hardware or under different CPU limits.

## 2. 🎯 Task Objective

Accelerate a complete simulation evaluation while preserving physical predictions and correct gradients. The fixed workload consists of one saved 3×3 optical setup, all 101 frequency points from 800–3000 Hz, and all 193 physical design variables. Double precision must be preserved.

This benchmark computes optical fields, signals, quantum noise, sensitivity, optical powers, a scalar objective, and gradients. It does not run a design-parameter optimization loop. The competition primarily compares the time for the first complete evaluation; warm runtime is reported separately.

## 3. 💻 Test Environment

| Item | Recorded configuration |
|---|---|
| CPU | Apple M4 |
| Operating system | macOS 15.7.4, ARM64 |
| Logical CPUs on host | 10; this does not imply that all cores were used |
| CPU affinity | Not configured; this is not a Linux single-logical-CPU benchmark |
| Python | 3.12.16 |
| Differometor | 0.0.6 |
| JAX / jaxlib | 0.9.0 / 0.9.0 |
| NumPy | 2.5.5 |
| Backend | CPU |
| Double precision | Enabled |
| Persistent compilation cache | Disabled |

Project location: `/Users/yuechen/Desktop/start-hack-munich-2026`.

## 4. 🧭 Completed Workflow

### 4.1 Install the Runtime Environment

Dependencies were installed in the project's `.venv`, and the project was installed in editable mode. Editable installation allows this environment to use subsequent source-code changes directly.

```bash
cd /Users/yuechen/Desktop/start-hack-munich-2026
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

The terminal displayed `Successfully installed differometor-0.0.6`.

### 4.2 Run the Starter Example

```bash
python start_here.py
```

Results recorded from the terminal screenshot:

| Stage | Seconds |
|---|---:|
| Setup | 12.290 |
| Compilation | 11.473 |
| Simulation + sensitivity + gradient | 17.763 |
| Total | 41.526 |

The output confirmed 101 frequencies and 193 gradient values. The mean logarithmic sensitivity relative to the reference was `9.692e-08`.

The starter example compiles simulation and differentiation together, whereas the full benchmark uses separate stages. Their times cannot be used directly to calculate optimization gains. An average metric close to zero also does not replace checks at every frequency.

### 4.3 Run the Full Original Baseline

The initial command used `--cpus 1`, but macOS does not support the Linux CPU-affinity interface required by that option, so it failed. Removing the option allowed this command to run successfully:

```bash
python benchmark.py --repeats 3 --output benchmark-baseline.json
```

The program completed the first evaluation, three warm runs, and its built-in correctness checks, then generated a JSON report and a PNG chart.

### 4.4 Create an Optimization Branch

```bash
git switch -c optimize-simulator
git status --short
```

The screenshot confirms that the branch was created successfully. The status command produced no output, meaning there were no file changes for Git to report at that point. Ignored generated files may still exist; creating a branch does not automatically commit reports or uncommitted changes.

### 4.5 Read the Core Code

```bash
sed -n '1,125p' differometor/simulate.py
```

This step only read the file and did not modify the program. Two core operations were explained:

- `update`: Extract component parameters, select physical formulas, calculate coefficients, and place them in the matrix.
- `solve`: Solve a system of linear equations to obtain complex optical fields.

No performance bottleneck has yet been isolated to a specific function or line of code.

## 5. ⏱️ Detailed Timings for the First Complete Evaluation

The following values come directly from the saved benchmark JSON. All times are in seconds.

| Stage | Time |
|---|---:|
| Python / library imports | 0.657532 |
| Initialize CPU backend | 0.012802 |
| Read and parse public inputs | 0.002705 |
| Build optical setup and prepare inputs | 0.982327 |
| Forward: trace / lower | 1.208071 |
| Forward: compile | 4.236405 |
| Forward simulation + AD residuals | 9.976687 |
| Sensitivity: trace / lower | 0.080248 |
| Sensitivity: compile | 0.067825 |
| Sensitivity, powers and scalar loss | 0.000574 |
| Gradient: trace / lower | 0.316576 |
| Gradient: compile | 2.246783 |
| Reverse-mode gradient | 6.411724 |
| Transfer results to host | 0.023451 |
| **First complete evaluation total** | **26.223708** |

The total includes imports, initialization, input loading, model construction, all tracing and compilation, forward simulation, sensitivity/power/objective calculations, gradients, and result transfer. It excludes validation, repeated measurements, report and chart generation, and interpreter startup before the first timer begins.

Most time was spent on forward simulation (about 9.98 seconds), tracing and compilation (about 8.16 seconds), and gradient calculation (about 6.41 seconds). This is evidence at the stage level; it does not establish that a particular function is the bottleneck.

## 6. 🔢 Warm Runs and the Median

After the first evaluation, the same process reuses the prepared and compiled program for three more measurements. Each run still recalculates physical outputs and gradients rather than returning the previous answers.

| Repetition | Time (seconds) |
|---|---:|
| 1 | 13.607138 |
| 2 | 12.180455 |
| 3 | 12.450582 |

Sorted from smallest to largest:

```text
12.180455 < 12.450582 < 13.607138
```

The middle value is the median: **12.450582 seconds**, or approximately **12.451 seconds**. The arithmetic mean of the three measurements is approximately 12.746058 seconds. The median and mean are different statistics.

The first complete evaluation's 26.223708 seconds includes preparation and compilation, so it is excluded from the warm-run median. Warm runs usually take less time because they do not repeat the full preparation and compilation work. Execution time also varies, so subtracting compilation time from the first total does not yield an exact warm runtime. The specific causes of variation were not measured separately in this test.

## 7. ✅ Correctness Validation

| Check | Recorded result |
|---|---|
| All physical outputs and gradients finite | Passed |
| Reference sensitivity check | Passed |
| Maximum absolute natural-log sensitivity error | 5.37304946135464e-06, below 1e-5 |
| Gradient shape | 193 values |
| Gradient norm | 70517.11274737977; nonzero |
| Parameter used for gradient spot check | mb11.tuning |
| Two central-difference step sizes | 0.001 and 0.0005 |
| Gradient spot check | Passed |
| Optical field and signal dtype | complex128 |
| Noise dtype | float64 |

`PASS` means the built-in checks passed for this workload. The numerical gradient spot check covers one parameter; it is not an independent element-by-element validation of all 193 gradients, nor does it prove correctness for every other optical setup.

## 8. 🔎 Established Findings and Open Questions

**Confirmed:** The environment works; the original version completes the full workload; built-in checks pass; baseline timings are saved; an optimization branch has been created.

**Not yet completed:** Detailed performance profiling, algorithm changes, candidate testing, and speedup calculation.

Potential directions include grouping component formulas, reusing matrix factorizations, and reducing intermediate-array memory overhead. These are candidate ideas only; their benefits have not been validated.

## 9. 🛠️ Next Experimental Steps

1. Choose a specific optimization hypothesis.
2. Make one reviewable change.
3. Measure the candidate on the same machine with the same test settings, saving a separate report:

```bash
python benchmark.py --repeats 3 --output benchmark-candidate.json
```

4. Confirm that correctness checks pass before comparing the first complete evaluation time and warm-run median.
5. For substantial numerical changes, compare output arrays or test another setup. Repeat promising measurements to distinguish actual improvements from runtime variation.

First-evaluation speedup factor = 26.223707664990798 / candidate first-evaluation time in seconds.

Time reduction percentage = (1 - candidate first-evaluation time in seconds / 26.223707664990798) × 100%.

No candidate measurements are available yet, so no speedup factor can currently be reported.

## 10. 📎 Sources and Attachments

- [Original baseline JSON](benchmark-baseline.json): The primary source for timings, environment details, and validation results in this report.
- [Original timing chart](benchmark-baseline.png): Generated during the same benchmark run.
- Starter-example timings, branch creation, and code inspection steps come from terminal screenshots provided by the user.

The attachments are copies of the original project's benchmark files; the original files remain in the project directory. Preparing this report did not rerun the simulation or modify the simulator code.

![Original timings by phase](benchmark-baseline.png)

The original Chinese text-only report is preserved as [baseline-report-original.md](baseline-report-original.md), allowing the added visual explanations to be checked against it. The enhanced Chinese report is also retained as [baseline-report.md](baseline-report.md).
