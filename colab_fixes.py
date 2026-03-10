# ═══════════════════════════════════════════════════════════════════════════════
# colab_fixes.py  —  Drop-in replacements for buggy cells in colab_experiment.ipynb
# Each section is labelled with the cell number to replace.
# ═══════════════════════════════════════════════════════════════════════════════


# ── CELL 2 FIX ────────────────────────────────────────────────────────────────
# BUG: %cd {REPO} runs unconditionally. If Cell 2 is re-run (or the kernel
#      is NOT restarted), CWD is already /content/LatentModelPositionalAnalysis.
#      Then os.path.exists("LatentModelPositionalAnalysis") → False (no subdir),
#      so it clones AGAIN into a nested subfolder and %cd puts you one level too deep.
# FIX: Anchor the cd to the absolute path, not the relative name.

import os

REPO = "LatentModelPositionalAnalysis"
REPO_PATH = f"/content/{REPO}"

if not os.path.exists(REPO_PATH):
    os.system(f"git clone https://github.com/Hieuuum/LatentModelPositionalAnalysis {REPO_PATH}")

# Always cd to the absolute path — safe to re-run
os.chdir(REPO_PATH)
print("CWD:", os.getcwd())
os.system("ls")


# ── CELL 6 FIX ────────────────────────────────────────────────────────────────
# BUG 1: run_probe() returns a command string but NEVER EXECUTES IT.
#         The callers in Cell 9 do `rc = run_probe(...)` and then check `rc`
#         as a return code — but rc is a string. No script ever runs.
# BUG 2: The arg "--lora_r 128 --lora_alpha 32 --lora_init" is ONE string in
#         the args list, so it gets treated as a single arg with spaces inside it,
#         which subprocess/IPython shell will pass correctly via ! but is fragile.
# FIX: Run the command inside run_probe() using get_ipython().system() so output
#      streams live to the cell, same pattern Cell 7 already uses correctly.

import os, sys

def run_probe(latent_iterations, use_hint=True, question_indices_file=None):
    """Build and EXECUTE the probe command, streaming output to the cell."""
    script = "probe_latent_cot_hint.py" if use_hint else "probe_latent_token.py"

    hint_str   = "CoT-hint" if use_hint else "no-hint"
    subset_str = f"subset({len(ORIGINAL_ONLY_INDICES)}q)" if question_indices_file else "full-dataset"
    print(f"\n{'='*60}")
    print(f"  latent={latent_iterations} | {hint_str} | {subset_str}")
    print(f"  CWD: {os.getcwd()}")
    print(f"{'='*60}\n")

    # Build arg string
    qi_arg = f"--question_indices {question_indices_file}" if question_indices_file else ""
    cmd = (
        f"python {script}"
        f" --data_name zen-E/GSM8k-Aug"
        f" --output_dir outputs"
        f" --model_name_or_path gpt2"
        f" --seed 11"
        f" --model_max_length 512"
        f" --bf16"
        f" --lora_r 128 --lora_alpha 32 --lora_init"
        f" --batch_size 32"
        f" --greedy True"
        f" --num_latent 6"
        f" --use_prj True"
        f" --prj_dim 768"
        f" --prj_no_ln False"
        f" --prj_dropout 0.0"
        f" --inf_latent_iterations {latent_iterations}"
        f" --inf_num_iterations 1"
        f" --remove_eos True"
        f" --use_lora True"
        f" --ckpt_dir {CKPT_DIR}"
        f" {qi_arg}"
    )
    print(f"CMD: {cmd}\n")

    # Execute and stream output (works in Colab cells)
    ip = get_ipython()
    rc = ip.system(cmd + " 2>&1")
    return rc   # now rc is actually the exit code (0 = success)

print("run_probe() ready — executes and streams output")
print(f"Python: {sys.executable}")
print(f"CWD:    {os.getcwd()}")


# ── CELL 9a FIX ───────────────────────────────────────────────────────────────
# BUG: run_probe() now runs the command internally, so the caller should just
#      call it directly and check the int return code, not treat rc as a string.
# (No code change needed here beyond Cell 6 fix above — this cell is now correct.)

print("Step 1: Regenerating original baseline (6 latents, no hint)...")
rc = run_probe(latent_iterations=6, use_hint=False, question_indices_file=None)
print(f"Baseline done (exit_code={rc})\n")


# ── CELL 9b FIX ───────────────────────────────────────────────────────────────
# BUG: The JSON filename built in the sweep loop uses "decoded_latent_cot_hint"
#      as the prefix even when use_hint=False. probe_latent_token.py writes to
#      "outputs/original_decoded_latent{suffix}.json" — so the json_path check
#      always fails and sweep_accuracies stays empty.
# FIX: Use the correct filename prefix depending on use_hint.

import os, json

sweep_accuracies = {}

for n_latent in LATENT_SWEEP:
    indices_file = "outputs/original_only_indices.txt" if RUN_SUBSET_ONLY else None
    rc = run_probe(
        latent_iterations=n_latent,
        use_hint=False,                       # no-hint sweep
        question_indices_file=indices_file,
    )
    suffix = f"_latent{n_latent}"
    if RUN_SUBSET_ONLY:
        suffix += f"_subset{len(ORIGINAL_ONLY_INDICES)}q"

    # FIX: probe_latent_token.py writes to "original_decoded_latent", NOT "decoded_latent_cot_hint"
    json_path = f"outputs/original_decoded_latent{suffix}.json"

    if os.path.exists(json_path):
        with open(json_path) as f:
            d = json.load(f)
        sweep_accuracies[n_latent] = d['accuracy']
        print(f"  latent={n_latent}: {d['accuracy']:.2f}% ({d['num_correct']}/{d['config']['num_questions']} correct)")
    else:
        print(f"  latent={n_latent}: output JSON not found at {json_path}")
