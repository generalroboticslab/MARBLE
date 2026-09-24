#!/bin/bash
set -u

# Ensure we're in the right directory
cd "$(dirname "$0")"

# Path to the py312 python interpreter
PYTHON="${BALLBOT_PYTHON:-/home/orangepi/repo/micromamba/envs/py312/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
    echo "Python 3.12 environment not found: $PYTHON"
    echo "Set BALLBOT_PYTHON to the correct interpreter."
    exit 1
fi

export PYTHONPATH="$(pwd)/src:$(pwd)/xiao_can:${PYTHONPATH:-}"

# Checkpoint options 5/6 preselect; other defaults live in ballbot_terminal.py only.
DEFAULT_POLICY_DIR="${BALLBOT_POLICY:-policies/BallbotVelComplexFlatDRLatency}"
DEFAULT_POLICY_DIR="${DEFAULT_POLICY_DIR%/policy_deployed.pt}"   # accept either form

# Extra flags appended to every ballbot_terminal.py launch, so a one-off knob does not
# require abandoning the menu. Word-split on purpose: BALLBOT_ARGS="--target-rate-limit 0.6".
read -r -a EXTRA_TERMINAL_ARGS <<< "${BALLBOT_ARGS:-}"

prompt_yn() {
    # Usage: prompt_yn "question" -> echoes "y" or "n"; default is "n" on bare ENTER.
    local reply
    read -r -p "$1 [y/N]: " reply
    case "$reply" in
        y|Y|yes|Yes) echo "y" ;;
        *) echo "n" ;;
    esac
}

pick_policy() {
    # Sets PICKED_POLICY to a policy directory. Returns 1 if nothing usable was chosen.
    # Lists policies/*/policy_deployed.pt; a requires_smoothing marker adds
    # --no-no-trajectory-smoothing (option 5).
    PICKED_POLICY=""
    local dirs=() d i tag default_idx=1 reply
    for d in policies/*/; do
        [[ -f "$d/policy_deployed.pt" ]] && dirs+=("${d%/}")
    done
    if (( ${#dirs[@]} == 0 )); then
        echo "No checkpoints found (looked for policies/*/policy_deployed.pt)."
        return 1
    fi

    echo "Available trained policies:"
    for i in "${!dirs[@]}"; do
        tag=""
        [[ -f "${dirs[$i]}/requires_smoothing" ]] && tag="   [legacy - runs with software PD]"
        [[ "${dirs[$i]}" == "$DEFAULT_POLICY_DIR" ]] && default_idx=$((i + 1))
        printf "  %d) %s%s\n" "$((i + 1))" "$(basename "${dirs[$i]}")" "$tag"
    done

    read -r -p "Select [1-${#dirs[@]}, ENTER = $default_idx]: " reply
    [[ -z "$reply" ]] && reply="$default_idx"
    if ! [[ "$reply" =~ ^[0-9]+$ ]] || (( reply < 1 || reply > ${#dirs[@]} )); then
        echo "Invalid selection."
        return 1
    fi
    PICKED_POLICY="${dirs[$((reply - 1))]}"
}

launch_control() {
    # Usage: launch_control <label> <policy-args...>
    local label="$1"
    shift
    local extra_args=()
    [[ "$(prompt_yn "Force a full rehome first (default: skip; arm at the reported pose, then centre)?")" == "y" ]] \
        && extra_args+=(--home)
    echo "Starting $label..."
    "$PYTHON" src/ballbot_terminal.py "$@" "${extra_args[@]}" "${EXTRA_TERMINAL_ARGS[@]}"
}

while true; do
    clear 2>/dev/null || true
    echo "================================================"
    echo "            BALLBOT CONTROL CENTER              "
    echo "================================================"
    echo "1) Offline heuristic readiness check"
    echo "2) Verify fixed Sim_Model.xml IMU placement"
    echo "3) Zero and map motors to model slider axes"
    echo "4) Start terminal control (heuristic)"
    echo "5) Start terminal control (trained policy)"
    echo "6) Preflight trained policy"
    echo "7) Exit"
    echo "================================================"
    echo -n "Select an option [1-7]: "
    read -r opt

    case $opt in
        1)
            echo "Checking heuristic files and dependencies..."
            "$PYTHON" scripts/preflight_heuristic.py
            echo ""
            echo "Press ENTER to return to menu..."
            read -r
            ;;
        2)
            viz_args=()
            if [[ "$(prompt_yn "Open the live 3D orientation view in a browser?")" == "y" ]]; then
                viz_args+=(--viz)
                echo "Serving on http://<PI_IP>:8080 (viser picks the next free port if busy)."
            fi
            echo "Starting fixed-pose IMU diagnostic..."
            "$PYTHON" scripts/calibrate_imu.py "${viz_args[@]}"
            echo ""
            echo "Press ENTER to return to menu..."
            read -r
            ;;
        3)
            echo "Starting Slider Calibration..."
            "$PYTHON" scripts/calibrate_sliders.py
            echo ""
            echo "Press ENTER to return to menu..."
            read -r
            ;;
        4)
            launch_control "heuristic terminal control" --policy heuristic
            echo ""
            echo "Press ENTER to return to menu..."
            read -r
            ;;
        5)
            if pick_policy; then
                smoothing_args=()
                [[ -f "$PICKED_POLICY/requires_smoothing" ]] \
                    && smoothing_args+=(--no-no-trajectory-smoothing)
                launch_control "trained policy: $(basename "$PICKED_POLICY")" \
                    --policy trained --policy-path "$PICKED_POLICY/policy_deployed.pt" \
                    "${smoothing_args[@]}"
            fi
            echo ""
            echo "Press ENTER to return to menu..."
            read -r
            ;;
        6)
            if pick_policy; then
                echo "Checking PyTorch and policy compatibility..."
                "$PYTHON" src/ballbot_terminal.py --preflight \
                    --policy-path "$PICKED_POLICY/policy_deployed.pt"
            fi
            echo ""
            echo "Press ENTER to return to menu..."
            read -r
            ;;
        7)
            echo "Exiting..."
            exit 0
            ;;
        *)
            echo "Invalid option. Press ENTER to try again..."
            read -r
            ;;
    esac
done
