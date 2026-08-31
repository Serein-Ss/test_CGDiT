#!/usr/bin/env bash
# Generate and evaluate all eight MP-20 models in a reproducible sequence.
# Tasks 8 (stability/MLIP) and 9 (full property grid) are intentionally disabled.

set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
export PROJECT_ROOT
export HYDRA_JOBS="${HYDRA_JOBS:-${PROJECT_ROOT}/output}"
export WABDB_DIR="${WABDB_DIR:-${PROJECT_ROOT}/wandb}"
export WANDB_MODE="${WANDB_MODE:-offline}"

cd "${PROJECT_ROOT}"

SEED="${SEED:-42}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-2.0}"
FORMAL_BATCH_SIZE="${FORMAL_BATCH_SIZE:-64}"
FORMAL_NUM_BATCHES="${FORMAL_NUM_BATCHES:-64}"
TOTAL_SAMPLES=$((FORMAL_BATCH_SIZE * FORMAL_NUM_BATCHES))
NUM_WORKERS="${NUM_WORKERS:-1}"
RUN_PILOT="${RUN_PILOT:-1}"
RUN_AB_INITIO="${RUN_AB_INITIO:-1}"

GT_FILE="${PROJECT_ROOT}/data/mp_20/test.csv"
GENERATION_LOG_DIR="${PROJECT_ROOT}/logs/structure_generation/mp20"
METRIC_LOG_DIR="${PROJECT_ROOT}/logs/metric_evaluation/mp20"
TEST_LOG_DIR="${PROJECT_ROOT}/logs/tests/generation_evaluation"
STATE_LOG_DIR="${PROJECT_ROOT}/logs/other/orchestration/generation_evaluation"
mkdir -p \
    "${GENERATION_LOG_DIR}" \
    "${METRIC_LOG_DIR}" \
    "${TEST_LOG_DIR}" \
    "${STATE_LOG_DIR}" \
    "${WABDB_DIR}"

if command -v flock >/dev/null 2>&1; then
    exec 9>"${STATE_LOG_DIR}/run.lock"
    if ! flock -n 9; then
        echo "Another all-model generation/evaluation run is already active." >&2
        exit 1
    fi
fi

MODEL_NAMES=(
    mp20_base
    mp20_fe
    mp20_bg
    mp20_eh
    mp20_fe_bg
    mp20_fe_eh
    mp20_bg_eh
    mp20_fe_bg_eh
)

MODEL_PATHS=(
    "output/singlerun/2026-06-27/00-32-50-mp20_base"
    "output/singlerun/2026-06-28/16-46-08-mp20_fe"
    "output/singlerun/2026-06-30/11-28-44-mp20_bg"
    "output/singlerun/2026-07-02/04-07-02-mp20_eh"
    "output/singlerun/2026-08-07/07-54-15-mp20_fe_bg"
    "output/singlerun/2026-08-08/18-02-14-mp20_fe_eh"
    "output/singlerun/2026-08-10/02-53-51-mp20_bg_eh"
    "output/singlerun/2026-08-05/13-12-29-mp20_fe_bg_eh"
)

declare -A CONDITION_SPECS=(
    [mp20_fe]="formation_energy_per_atom=-1.5"
    [mp20_bg]="band_gap=2.0"
    [mp20_eh]="e_above_hull=0.0"
    [mp20_fe_bg]="formation_energy_per_atom=-1.5 band_gap=2.0"
    [mp20_fe_eh]="formation_energy_per_atom=-1.5 e_above_hull=0.0"
    [mp20_bg_eh]="band_gap=2.0 e_above_hull=0.0"
    [mp20_fe_bg_eh]="formation_energy_per_atom=-1.5 band_gap=2.0 e_above_hull=0.0"
)

declare -A CONDITION_LABELS=(
    [mp20_fe]="fe_m1p5"
    [mp20_bg]="bg_2"
    [mp20_eh]="eh_0"
    [mp20_fe_bg]="fe_m1p5_bg_2"
    [mp20_fe_eh]="fe_m1p5_eh_0"
    [mp20_bg_eh]="bg_2_eh_0"
    [mp20_fe_bg_eh]="fe_m1p5_bg_2_eh_0"
)

FAILURES=()

log_header() {
    echo
    echo "============================================================"
    echo "$1"
    echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================================"
}

run_logged() {
    local log_dir="$1"
    local step_name="$2"
    shift 2
    local log_file="${log_dir}/${step_name}.log"

    echo "[RUN] ${step_name}"
    if "$@" 2>&1 | tee "${log_file}"; then
        echo "[OK] ${step_name}"
    else
        local status=$?
        echo "[FAILED:${status}] ${step_name}" | tee -a "${STATE_LOG_DIR}/failures.log"
        FAILURES+=("${step_name}")
    fi
}

condition_args_for() {
    local model_name="$1"
    local spec
    for spec in ${CONDITION_SPECS[${model_name}]-}; do
        printf '%s\n' "${spec}"
    done
}

generation_file_for() {
    python -c 'import sys; from cgdit.common.output_paths import resolve_generation_output_path; print(resolve_generation_output_path(sys.argv[1], sys.argv[2]))' "$1" "$2"
}

metrics_file_for() {
    python -c 'import sys; from cgdit.common.output_paths import resolve_evaluation_output_path; print(resolve_evaluation_output_path(sys.argv[1], "structural_metrics", sys.argv[2]))' "$1" "$2"
}

run_generation() {
    local model_name="$1"
    local model_path="$2"
    local label="$3"
    local batch_size="$4"
    local num_batches="$5"
    local mode="$6"
    local use_conditions="$7"
    local output_file
    output_file="$(generation_file_for "${model_path}" "${label}")"

    if [[ -s "${output_file}" ]]; then
        echo "[SKIP] Existing generation output: ${output_file}"
        return
    fi

    local cmd=(
        python -m scripts.cli.generation.generate
        --model_path "${model_path}"
        --batch_size "${batch_size}"
        --num_batches_to_samples "${num_batches}"
        --seed "${SEED}"
        --label "${label}"
    )

    if [[ "${mode}" == "abinitio" ]]; then
        cmd+=(--ab_initio --use_empirical_prior)
    fi

    if [[ "${use_conditions}" == "1" ]]; then
        cmd+=(--guidance_scale "${GUIDANCE_SCALE}")
        while IFS= read -r assignment; do
            [[ -n "${assignment}" ]] && cmd+=(--condition "${assignment}")
        done < <(condition_args_for "${model_name}")
    else
        cmd+=(--guidance_scale 1.0)
    fi

    run_logged "${GENERATION_LOG_DIR}" "generate_${model_name}_${label}" "${cmd[@]}"
}

run_metrics() {
    local model_name="$1"
    local model_path="$2"
    local label="$3"
    local input_file
    input_file="$(generation_file_for "${model_path}" "${label}")"
    local metrics_file
    metrics_file="$(metrics_file_for "${input_file}" "eval_metrics_gen_${label}.json")"

    if [[ ! -s "${input_file}" ]]; then
        echo "[FAILED] Missing generation output for metrics: ${input_file}" \
            | tee -a "${STATE_LOG_DIR}/failures.log"
        FAILURES+=("metrics_${model_name}_${label}_missing_input")
        return
    fi

    if [[ -s "${metrics_file}" ]]; then
        echo "[SKIP] Existing metrics: ${metrics_file}"
        return
    fi

    run_logged "${METRIC_LOG_DIR}" "metrics_${model_name}_${label}" \
        python -m scripts.cli.evaluation.evaluate_metrics \
        --root_path "${model_path}" \
        --tasks gen \
        --label "${label}" \
        --gt_file "${GT_FILE}" \
        --num_workers "${NUM_WORKERS}" \
        --seed "${SEED}" \
        --calc_prop false
}

log_header "Preflight"

echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "Git commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "Seed=${SEED}"
echo "Formal samples per setting=${TOTAL_SAMPLES} (${FORMAL_BATCH_SIZE} x ${FORMAL_NUM_BATCHES})"
echo "Conditional CFG scale=${GUIDANCE_SCALE}"
python --version
python -c "import torch; print('torch=', torch.__version__); print('cuda=', torch.cuda.is_available()); print('gpu=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

for required_file in \
    data/mp_20/train.csv \
    data/mp_20/test.csv \
    data/mp_20/train_sym.pt \
    data/mp_20/test_sym.pt; do
    [[ -s "${required_file}" ]] || { echo "Missing required data file: ${required_file}" >&2; exit 1; }
done

for i in "${!MODEL_NAMES[@]}"; do
    model_path="${MODEL_PATHS[$i]}"
    [[ -s "${model_path}/hparams.yaml" ]] || { echo "Missing hparams: ${model_path}" >&2; exit 1; }
    compgen -G "${model_path}/*.ckpt" >/dev/null || { echo "Missing checkpoint: ${model_path}" >&2; exit 1; }
done

run_logged "${TEST_LOG_DIR}" "preflight_pytest" python -m pytest tests -q

if [[ ${#FAILURES[@]} -gt 0 ]]; then
    echo "Preflight failed; formal generation will not start." >&2
    exit 1
fi

if [[ "${RUN_PILOT}" == "1" ]]; then
    log_header "Task 1: pilot generation"

    for i in "${!MODEL_NAMES[@]}"; do
        model_name="${MODEL_NAMES[$i]}"
        model_path="${MODEL_PATHS[$i]}"
        run_generation "${model_name}" "${model_path}" \
            "pilot_template_uncond_seed${SEED}" 1 1 template 0

        if [[ -n "${CONDITION_SPECS[${model_name}]-}" ]]; then
            condition_label="${CONDITION_LABELS[${model_name}]}"
            run_generation "${model_name}" "${model_path}" \
                "pilot_template_${condition_label}_seed${SEED}" 1 1 template 1
        fi
    done

    if [[ ${#FAILURES[@]} -gt 0 ]]; then
        echo "Pilot generation failed; formal generation will not start." >&2
        exit 1
    fi
fi

log_header "Task 2: formal unconditional template generation for all models"

for i in "${!MODEL_NAMES[@]}"; do
    model_name="${MODEL_NAMES[$i]}"
    model_path="${MODEL_PATHS[$i]}"
    run_generation "${model_name}" "${model_path}" \
        "template_uncond_n${TOTAL_SAMPLES}_seed${SEED}" \
        "${FORMAL_BATCH_SIZE}" "${FORMAL_NUM_BATCHES}" template 0
done

log_header "Task 3: formal common-target template generation for conditional models"

for i in "${!MODEL_NAMES[@]}"; do
    model_name="${MODEL_NAMES[$i]}"
    model_path="${MODEL_PATHS[$i]}"
    [[ -n "${CONDITION_SPECS[${model_name}]-}" ]] || continue
    condition_label="${CONDITION_LABELS[${model_name}]}"
    run_generation "${model_name}" "${model_path}" \
        "template_${condition_label}_n${TOTAL_SAMPLES}_seed${SEED}" \
        "${FORMAL_BATCH_SIZE}" "${FORMAL_NUM_BATCHES}" template 1
done

log_header "Task 4: structural metrics for template generation"

for i in "${!MODEL_NAMES[@]}"; do
    model_name="${MODEL_NAMES[$i]}"
    model_path="${MODEL_PATHS[$i]}"
    run_metrics "${model_name}" "${model_path}" \
        "template_uncond_n${TOTAL_SAMPLES}_seed${SEED}"

    if [[ -n "${CONDITION_SPECS[${model_name}]-}" ]]; then
        condition_label="${CONDITION_LABELS[${model_name}]}"
        run_metrics "${model_name}" "${model_path}" \
            "template_${condition_label}_n${TOTAL_SAMPLES}_seed${SEED}"
    fi
done

log_header "Tasks 5-6: target-property prediction and joint hit rate"
cat <<'EOF'
[PENDING] These tasks are not executed by the current repository because no
independent FE, BG, and Ehull predictor checkpoints are available, and the
current wdist_prop metric is not a multi-target hit-rate metric. Structural
metrics above use --calc_prop false intentionally.
EOF

if [[ "${RUN_AB_INITIO}" == "1" ]]; then
    log_header "Task 7: empirical-prior ab initio generation and structural metrics"

    for i in "${!MODEL_NAMES[@]}"; do
        model_name="${MODEL_NAMES[$i]}"
        model_path="${MODEL_PATHS[$i]}"
        condition_label="uncond"
        use_conditions=0

        if [[ -n "${CONDITION_SPECS[${model_name}]-}" ]]; then
            condition_label="${CONDITION_LABELS[${model_name}]}"
            use_conditions=1
        fi

        if [[ "${RUN_PILOT}" == "1" ]]; then
            run_generation "${model_name}" "${model_path}" \
                "pilot_abinitio_empirical_${condition_label}_seed${SEED}" \
                1 1 abinitio "${use_conditions}"
        fi

        formal_label="abinitio_empirical_${condition_label}_n${TOTAL_SAMPLES}_seed${SEED}"
        run_generation "${model_name}" "${model_path}" \
            "${formal_label}" \
            "${FORMAL_BATCH_SIZE}" "${FORMAL_NUM_BATCHES}" \
            abinitio "${use_conditions}"
        run_metrics "${model_name}" "${model_path}" "${formal_label}"
    done
fi

log_header "Tasks 8-9 disabled"
echo "Task 8 stability/MLIP evaluation: NOT RUN (requested)."
echo "Task 9 full property target grid: NOT RUN (requested)."

log_header "Run summary"
if [[ ${#FAILURES[@]} -gt 0 ]]; then
    printf 'Failed steps (%d):\n' "${#FAILURES[@]}"
    printf '  - %s\n' "${FAILURES[@]}"
    echo "See ${STATE_LOG_DIR}/failures.log and the per-step logs."
    exit 1
fi

echo "All executable tasks completed successfully."
echo "Outputs: generated_structures/ and evaluations/ under each model directory."
echo "Generation logs: ${GENERATION_LOG_DIR}"
echo "Metric logs: ${METRIC_LOG_DIR}"
echo "Test logs: ${TEST_LOG_DIR}"
echo "Run state: ${STATE_LOG_DIR}"
