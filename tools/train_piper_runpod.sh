#!/usr/bin/env bash
#
# train_piper_runpod.sh
#
# Fine-tune a Piper voice on a rented NVIDIA GPU (RunPod or any Linux box with
# CUDA), from the dataset that generate_voice_dataset.py builds. Stages:
#
#	bash train_piper_runpod.sh setup
#		Installs build tools, clones piper1-gpl, builds its two compiled
#		pieces, downloads the Lessac medium checkpoint. Run once per pod.
#
#	bash train_piper_runpod.sh train <voice_name> <dataset_dir>
#		Starts training in the background (survives closing the terminal).
#		<dataset_dir> is the folder holding metadata.csv and wav/.
#		Run it again after a stop/restart and it resumes from last.ckpt.
#
#	bash train_piper_runpod.sh export <voice_name>
#		Exports the newest last.ckpt to <voice_name>.onnx + .onnx.json,
#		renders a test sentence, and packs everything into one tarball
#		ready for `runpodctl send`.
#
#	bash train_piper_runpod.sh status <voice_name>
#		Shows the current epoch and the tail of the training log.
#
# Settings (environment variables, all optional):
#	WORK_DIR      where everything lives          (default /workspace)
#	MAX_EPOCHS    run length, sets the LR anneal  (default 2000)
#	BATCH_SIZE    training batch size             (default 32)
#	NUM_WORKERS   dataloader workers              (default 4)
#	ACCELERATOR   gpu, or cpu for a smoke test    (default gpu)
#	PIPER_REF     piper1-gpl commit to check out  (default: the one this
#	              script was written against)
#	EXTRA_ARGS    anything else for `piper.train fit`
#
# /workspace is RunPod's persistent volume: it survives stopping the pod,
# the container disk does not. Keep everything there.

set -eo pipefail

WORK_DIR="${WORK_DIR:-/workspace}"
MAX_EPOCHS="${MAX_EPOCHS:-2000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
ACCELERATOR="${ACCELERATOR:-gpu}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

# piper1-gpl's training code changes often. Pin to the commit this script was
# checked against so a re-run months from now behaves the same.
PIPER_REF="${PIPER_REF:-10880972db9e0aca101fed350eadf8f49deefe9f}"
PIPER_REPO="https://github.com/OHF-Voice/piper1-gpl.git"
PIPER_DIR="${WORK_DIR}/piper1-gpl"
VENV_DIR="${PIPER_DIR}/.venv"

# Fine-tune base. Medium quality = 22050 Hz, which is what the dataset
# generator produces.
BASE_CKPT_URL="https://huggingface.co/datasets/rhasspy/piper-checkpoints/resolve/main/en/en_US/lessac/medium/epoch%3D2164-step%3D1355540.ckpt"
BASE_CKPT="${WORK_DIR}/checkpoints/lessac-medium.ckpt"

TEST_SENTENCE="${TEST_SENTENCE:-Hello there! This is my new offline voice. How does it sound to you?}"

log() {
	echo
	echo "==> $*"
}

die() {
	echo "ERROR: $*" >&2
	exit 1
}

as_root() {
	if [ "$(id -u)" -eq 0 ]; then
		"$@"
	else
		sudo "$@"
	fi
}

activate_venv() {
	[ -f "${VENV_DIR}/bin/activate" ] || die "No venv at ${VENV_DIR}. Run: bash $0 setup"
	# shellcheck disable=SC1091
	source "${VENV_DIR}/bin/activate"
	cd "${PIPER_DIR}"
}

voice_dir() {
	echo "${WORK_DIR}/train/$1"
}

# Newest last.ckpt across every lightning_logs/version_N (each resume makes a
# new version directory).
latest_last_ckpt() {
	local root
	root="$(voice_dir "$1")/lightning_logs"
	[ -d "${root}" ] || return 0
	find "${root}" -name 'last.ckpt' -printf '%T@ %p\n' 2>/dev/null \
		| sort -n | tail -1 | cut -d' ' -f2-
}

# -----------------------------------------------------------------------------

stage_setup() {
	log "Installing build tools"
	as_root apt-get update -qq
	as_root apt-get install -y -qq build-essential cmake ninja-build git wget >/dev/null

	mkdir -p "${WORK_DIR}/checkpoints"

	if [ ! -d "${PIPER_DIR}/.git" ]; then
		log "Cloning piper1-gpl"
		git clone -q "${PIPER_REPO}" "${PIPER_DIR}"
	fi
	log "Checking out piper1-gpl ${PIPER_REF}"
	git -C "${PIPER_DIR}" fetch -q origin
	git -C "${PIPER_DIR}" checkout -q "${PIPER_REF}"

	if [ ! -d "${VENV_DIR}" ]; then
		log "Creating venv (reusing the pod's preinstalled PyTorch)"
		# --system-site-packages lets the venv see the template's CUDA build of
		# torch, so pip doesn't download a second 2+ GB copy that may not
		# match the pod's driver.
		python3 -m venv --system-site-packages "${VENV_DIR}"
	fi
	activate_venv

	log "Installing piper training dependencies"
	python3 -m pip install -q --upgrade pip wheel
	python3 -m pip install -q scikit-build cmake ninja cython
	python3 -m pip install -q -e '.[train]'

	log "Building monotonic_align (Cython)"
	./build_monotonic_align.sh

	log "Building espeakbridge (downloads and compiles espeak-ng, takes a few minutes)"
	python3 setup.py build_ext --inplace >"${WORK_DIR}/espeakbridge_build.log" 2>&1 \
		|| { tail -40 "${WORK_DIR}/espeakbridge_build.log"; die "espeakbridge build failed (full log: ${WORK_DIR}/espeakbridge_build.log)"; }

	if [ ! -s "${BASE_CKPT}" ]; then
		log "Downloading Lessac medium checkpoint (~850 MB)"
		wget -q --show-progress -O "${BASE_CKPT}.part" "${BASE_CKPT_URL}"
		mv "${BASE_CKPT}.part" "${BASE_CKPT}"
	fi

	log "Checking the environment"
	python3 - <<'EOF'
import torch
import piper.espeakbridge  # noqa: F401
from piper.train.vits.monotonic_align import maximum_path  # noqa: F401
print(f"torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
	print(f"GPU: {torch.cuda.get_device_name(0)}")
EOF

	log "Setup complete. Next:"
	echo "    bash $0 train <voice_name> <dataset_dir>"
}

stage_train() {
	local voice="$1"
	local dataset="$2"
	[ -n "${voice}" ] && [ -n "${dataset}" ] || die "Usage: bash $0 train <voice_name> <dataset_dir>"

	dataset="$(cd "${dataset}" && pwd)" || die "Dataset directory not found: $2"
	[ -f "${dataset}/metadata.csv" ] || die "No metadata.csv in ${dataset}"
	[ -d "${dataset}/wav" ] || die "No wav/ folder in ${dataset}"

	local clips
	clips="$(grep -c . "${dataset}/metadata.csv" || true)"
	[ "${clips}" -gt 0 ] || die "metadata.csv is empty"

	activate_venv

	local vdir
	vdir="$(voice_dir "${voice}")"
	mkdir -p "${vdir}"

	if pgrep -f "piper.train fit.*--data.voice_name ${voice} " >/dev/null; then
		die "Training for '${voice}' is already running. Watch it with: bash $0 status ${voice}"
	fi

	# Either resume this voice's own run, or warm-start a new one from Lessac.
	# warmstart_ckpt copies the weights and starts at epoch 0 with a fresh
	# optimizer, so MAX_EPOCHS means what it says and the learning rate anneals
	# over this run. (--ckpt_path on the Lessac file would instead resume at
	# its epoch 2164 with its old, nearly flat learning rate schedule.)
	local resume_from start_args
	resume_from="$(latest_last_ckpt "${voice}")"
	if [ -n "${resume_from}" ]; then
		log "Resuming from ${resume_from}"
		start_args=(--ckpt_path "${resume_from}")
	else
		[ -s "${BASE_CKPT}" ] || die "Base checkpoint missing. Run: bash $0 setup"
		log "Starting a new fine-tune from Lessac medium"
		start_args=(--model.warmstart_ckpt "${BASE_CKPT}")
	fi

	echo "    voice:      ${voice}"
	echo "    dataset:    ${dataset} (${clips} clips)"
	echo "    epochs:     ${MAX_EPOCHS}   batch: ${BATCH_SIZE}   accelerator: ${ACCELERATOR}"
	echo "    output:     ${vdir}"

	local logfile="${vdir}/train.log"

	# shellcheck disable=SC2086
	nohup python3 -m piper.train fit \
		--data.voice_name "${voice}" \
		--data.csv_path "${dataset}/metadata.csv" \
		--data.audio_dir "${dataset}/wav" \
		--model.sample_rate 22050 \
		--data.espeak_voice en-us \
		--data.cache_dir "${vdir}/cache" \
		--data.config_path "${vdir}/config.json" \
		--data.batch_size "${BATCH_SIZE}" \
		--data.num_workers "${NUM_WORKERS}" \
		--trainer.accelerator "${ACCELERATOR}" \
		--trainer.devices 1 \
		--trainer.max_epochs "${MAX_EPOCHS}" \
		--trainer.default_root_dir "${vdir}" \
		"${start_args[@]}" \
		${EXTRA_ARGS} \
		>>"${logfile}" 2>&1 &

	local pid=$!
	sleep 5
	if ! kill -0 "${pid}" 2>/dev/null; then
		tail -40 "${logfile}"
		die "Training exited immediately. Full log: ${logfile}"
	fi

	log "Training is running in the background (pid ${pid})."
	echo "    It keeps going if you close this terminal."
	echo "    Progress:   bash $0 status ${voice}"
	echo "    Live log:   tail -f ${logfile}"
	echo "    When done:  bash $0 export ${voice}"
}

stage_status() {
	local voice="$1"
	[ -n "${voice}" ] || die "Usage: bash $0 status <voice_name>"
	local vdir logfile
	vdir="$(voice_dir "${voice}")"
	logfile="${vdir}/train.log"
	[ -f "${logfile}" ] || die "No training log at ${logfile}"

	if pgrep -f "piper.train fit.*--data.voice_name ${voice} " >/dev/null; then
		echo "Status: RUNNING"
	else
		echo "Status: not running"
	fi

	local ckpt
	ckpt="$(latest_last_ckpt "${voice}")"
	if [ -n "${ckpt}" ]; then
		activate_venv
		# Training rewrites last.ckpt every epoch, so a read can land mid-write.
		# Retry a few times, and never let a failed read stop the log tail below.
		python3 - "${ckpt}" "${MAX_EPOCHS}" <<'EOF' || true
import sys
import time
import torch
for attempt in range(5):
	try:
		ckpt = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
		break
	except Exception:
		time.sleep(2)
else:
	print("Checkpoint is being written right now; run status again in a moment.")
	sys.exit(0)
print(f"Last finished epoch: {ckpt['epoch']} of {sys.argv[2]}")
for state in ckpt.get("callbacks", {}).values():
	if state.get("monitor") and state.get("best_model_score") is not None:
		print(f"Best {state['monitor']}: {float(state['best_model_score']):.4f}")
EOF
	fi

	echo
	echo "--- last lines of ${logfile} ---"
	# The Lightning progress bar redraws with carriage returns; show the
	# latest redraw of each line rather than one enormous line.
	tail -c 20000 "${logfile}" | tr '\r' '\n' | grep -v '^\s*$' | tail -15
}

stage_export() {
	local voice="$1"
	[ -n "${voice}" ] || die "Usage: bash $0 export <voice_name>"
	activate_venv

	local vdir
	vdir="$(voice_dir "${voice}")"
	[ -f "${vdir}/config.json" ] || die "No config.json in ${vdir}. Has training started?"

	local ckpt
	ckpt="$(latest_last_ckpt "${voice}")"
	[ -n "${ckpt}" ] || die "No last.ckpt found under ${vdir}/lightning_logs"

	local out="${WORK_DIR}/export/${voice}"
	mkdir -p "${out}"

	# last.ckpt is rewritten every epoch. Copy it first so a training run that
	# is still going can't replace it halfway through the export, and check
	# the copy loads, since the copy itself can catch a write in progress.
	log "Exporting ${ckpt}"
	local tries=0
	while :; do
		cp "${ckpt}" "${out}/${voice}.ckpt"
		if python3 -c "import sys, torch; torch.load(sys.argv[1], map_location='cpu', weights_only=False)" \
			"${out}/${voice}.ckpt" 2>/dev/null; then
			break
		fi
		tries=$((tries + 1))
		[ "${tries}" -lt 10 ] || die "Couldn't get a complete copy of ${ckpt}. Try again, or stop training first."
		echo "    Checkpoint was mid-write, retrying..."
		sleep 3
	done

	python3 -m piper.train.export_onnx \
		--checkpoint "${out}/${voice}.ckpt" \
		--output-file "${out}/${voice}.onnx"
	cp "${vdir}/config.json" "${out}/${voice}.onnx.json"

	log "Rendering a test sentence"
	if echo "${TEST_SENTENCE}" | python3 -m piper -m "${out}/${voice}.onnx" -f "${out}/${voice}_test.wav"; then
		echo "    ${out}/${voice}_test.wav"
	else
		echo "    (test render failed; the model files are still fine to try on the Jetson)"
	fi

	local tarball="${WORK_DIR}/export/${voice}_piper.tar.gz"
	tar -czf "${tarball}" -C "${out}" "${voice}.onnx" "${voice}.onnx.json" \
		$( [ -f "${out}/${voice}_test.wav" ] && echo "${voice}_test.wav" )

	log "Done. To get it onto your Mac:"
	echo "    On the pod:   runpodctl send ${tarball}"
	echo "    On the Mac:   runpodctl receive <code it prints>"
	echo
	echo "    The .ckpt copy is in ${out} if you want to keep training later."
}

# -----------------------------------------------------------------------------

case "${1:-}" in
	setup)  stage_setup ;;
	train)  stage_train "$2" "$3" ;;
	status) stage_status "$2" ;;
	export) stage_export "$2" ;;
	*)
		sed -n '3,/^# Settings/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'
		exit 1
		;;
esac
