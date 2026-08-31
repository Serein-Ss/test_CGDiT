#!/usr/bin/env bash

rl_resolve_run_id() {
    if [[ -n "${RUN_ID:-}" ]]; then
        printf '%s\n' "${RUN_ID}"
        return
    fi

    local job_id="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-}}"
    local submit_time=""
    if [[ -n "${job_id}" ]] && command -v scontrol >/dev/null 2>&1; then
        submit_time="$(scontrol show job "${job_id}" -o 2>/dev/null \
            | sed -n 's/.*SubmitTime=\([^ ]*\).*/\1/p')"
    fi
    if [[ -n "${submit_time}" && "${submit_time}" != "Unknown" ]]; then
        date -d "${submit_time}" '+%Y%m%d-%H%M%S'
    else
        date '+%Y%m%d-%H%M%S'
    fi
}

rl_run_root() {
    local project_root="$1"
    local scope="$2"
    local experiment="$3"
    local run_id="${4:-$(rl_resolve_run_id)}"

    if ! [[ "${run_id}" =~ ^[0-9]{8}-[0-9]{6}$ ]]; then
        echo "Invalid RL run id: ${run_id}" >&2
        return 2
    fi
    if ! [[ "${experiment}" =~ ^[A-Za-z0-9._-]+$ ]]; then
        echo "Invalid RL experiment name: ${experiment}" >&2
        return 2
    fi

    local run_date="${run_id:0:4}-${run_id:4:2}-${run_id:6:2}"
    local run_time="${run_id:9:2}-${run_id:11:2}-${run_id:13:2}"
    printf '%s/output/%s/%s/%s-%s\n' \
        "${project_root}" "${scope}" "${run_date}" "${run_time}" "${experiment}"
}
