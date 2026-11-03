#!/usr/bin/env python3
"""
generate_voice_dataset.py

Build a Piper TTS training dataset by generating each prompt through the
ElevenLabs voice this project already uses. The output is a directory of WAVs
plus a metadata.csv, ready to hand to tools/train_piper_runpod.sh
(which runs `python3 -m piper.train fit` on a GPU machine).

Why generate rather than record: fine-tuning Piper needs (audio, transcript)
pairs. Recording them means a microphone, consistent takes, and either reading
from a script or transcribing and hand-correcting afterwards. Generating them
means the transcript is exactly what you asked for, every clip has identical
recording conditions, and there is nothing to segment. The tradeoff is that
you are cloning a clone — the fine-tune's ceiling is ElevenLabs' output, not
the original reference audio. For a fallback voice that's the right trade.

Audio is requested as raw PCM at 22050 Hz, which is exactly what a Piper
medium-quality voice expects, so there is no MP3 round trip and no resampling.

Runs on the Jetson or on a desktop — it only needs `requests` and `numpy`, not
pygame or any of the hardware modules.

Usage:
	# See what it would cost before spending anything
	python3 tools/generate_voice_dataset.py --dry-run

	# Generate a handful first and listen to them
	python3 tools/generate_voice_dataset.py --limit 5

	# Generate everything (resumable — safe to re-run)
	python3 tools/generate_voice_dataset.py

	# Check what you have
	python3 tools/generate_voice_dataset.py --stats
"""

import argparse
import configparser
import hashlib
import os
import sys
import time
import wave

import numpy as np
import requests

ELEVENLABS_API_URL = "https://api.elevenlabs.io/v1/text-to-speech"

# Piper medium-quality voices are 22.05kHz mono 16-bit. Ask ElevenLabs for
# exactly that and the bytes need no conversion at all.
SAMPLE_RATE = 22050
OUTPUT_FORMAT = "pcm_22050"

# Clips outside this range are flagged. Very short clips carry no prosody;
# very long ones get dropped by Piper's phoneme-length cap during training.
MIN_CLIP_S = 1.0
MAX_CLIP_S = 15.0

# Silence trim threshold, as int16 amplitude. ElevenLabs pads the head and tail
# of each generation; leaving that in teaches the model to produce it.
TRIM_THRESHOLD = 250
TRIM_MARGIN_S = 0.05

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_BASE_DIR = os.path.dirname(_SCRIPT_DIR)

PROMPTS_FILENAME = "voice_dataset_prompts.txt"


def resolve_prompts(explicit_path: str = "") -> str:
	"""Find the prompt list.

	Checked in order: an explicit --prompts path, next to this script, then
	the repo root. Either placement works, so it doesn't matter whether the
	file was dropped in tools/ with the script or at the top level.
	"""
	if explicit_path:
		if not os.path.exists(explicit_path):
			sys.exit(f"Prompt file not found: {explicit_path}")
		return explicit_path

	candidates = [
		os.path.join(_SCRIPT_DIR, PROMPTS_FILENAME),
		os.path.join(_BASE_DIR, PROMPTS_FILENAME),
	]
	for path in candidates:
		if os.path.exists(path):
			return path

	sys.exit(
		f"Prompt file '{PROMPTS_FILENAME}' not found. Looked in:\n"
		+ "\n".join(f"  {p}" for p in candidates)
		+ "\nPut it in either location, or pass --prompts /path/to/file"
	)


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

def load_tts_config(explicit_path: str = "") -> dict:
	"""Read [TextToSpeech] from config.cfg, matching text_to_speech.py."""
	candidates = []
	if explicit_path:
		candidates.append(explicit_path)
	else:
		candidates.append(os.path.join(_BASE_DIR, "config.cfg"))
		candidates.append("/mnt/usb/config.cfg")

	path = next((p for p in candidates if os.path.exists(p)), None)
	if path is None:
		sys.exit(f"No config found. Looked in: {', '.join(candidates)}")

	cfg = configparser.ConfigParser()
	try:
		cfg.read(path)
	except configparser.Error as e:
		sys.exit(f"Failed to parse config at '{path}': {e}")

	# Note: "Simularity" is spelled that way in config.cfg — matching the
	# existing key rather than fixing it, so both read the same file.
	conf = {
		"path": path,
		"key": cfg.get("TextToSpeech", "ElevenLabsKey", fallback="").strip(),
		"voice_id": cfg.get("TextToSpeech", "ElevenLabsVoiceID", fallback="").strip(),
		"stability": cfg.getfloat("TextToSpeech", "ElevenLabsStability", fallback=0.35),
		"similarity_boost": cfg.getfloat("TextToSpeech", "ElevenLabsSimularityBoost", fallback=0.75),
		"style": cfg.getfloat("TextToSpeech", "ElevenLabsStyle", fallback=0.3),
		"high_quality": cfg.getboolean("TextToSpeech", "ElevenLabsUseHighQualitySlowModel", fallback=False),
	}
	if not conf["key"]:
		sys.exit(f"No ElevenLabsKey in {path}")
	if not conf["voice_id"]:
		sys.exit(f"No ElevenLabsVoiceID in {path}")
	return conf


def load_prompts(path: str) -> list:
	lines = []
	seen = set()
	with open(path, "r", encoding="utf-8") as f:
		for raw in f:
			text = raw.strip()
			if not text or text.startswith("#"):
				continue
			if text in seen:
				# Duplicates would produce identical clips under the same
				# hash — harmless but wasteful, so drop them quietly.
				continue
			seen.add(text)
			lines.append(text)
	return lines


def clip_name(text: str) -> str:
	"""Stable filename from the text itself.

	Deliberately not an index: hashing means you can reorder, insert and
	delete prompt lines without invalidating clips you've already generated
	and paid for.
	"""
	return "utt_" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:12] + ".wav"


# -----------------------------------------------------------------------------
# Audio
# -----------------------------------------------------------------------------

def trim_silence(pcm: np.ndarray) -> np.ndarray:
	loud = np.flatnonzero(np.abs(pcm) > TRIM_THRESHOLD)
	if loud.size == 0:
		return pcm
	margin = int(TRIM_MARGIN_S * SAMPLE_RATE)
	start = max(0, loud[0] - margin)
	end = min(pcm.size, loud[-1] + margin)
	return pcm[start:end]


def write_wav(path: str, pcm: np.ndarray) -> float:
	with wave.open(path, "wb") as wf:
		wf.setnchannels(1)
		wf.setsampwidth(2)
		wf.setframerate(SAMPLE_RATE)
		wf.writeframes(pcm.tobytes())
	return pcm.size / SAMPLE_RATE


def wav_duration(path: str) -> float:
	try:
		with wave.open(path, "rb") as wf:
			return wf.getnframes() / wf.getframerate()
	except Exception:
		return 0.0


# -----------------------------------------------------------------------------
# Generation
# -----------------------------------------------------------------------------

class ElevenLabsError(RuntimeError):
	"""An error response from ElevenLabs, carrying its detail message.

	ElevenLabs puts the actual reason in a JSON "detail" object (which
	parameter was wrong, what it expected). requests' raise_for_status()
	throws that away and leaves only the status line.
	"""

	def __init__(self, response: requests.Response):
		self.status_code = response.status_code
		self.code = ""
		self.param = ""
		message = response.text.strip()[:300] or response.reason
		try:
			detail = response.json().get("detail")
			if isinstance(detail, dict):
				self.code = detail.get("code") or detail.get("status") or ""
				self.param = detail.get("param") or ""
				message = detail.get("message") or message
			elif isinstance(detail, list) and detail:
				# FastAPI-style validation errors: a list of {loc, msg}
				parts = []
				for item in detail:
					loc = ".".join(str(x) for x in item.get("loc", []))
					parts.append(f"{loc}: {item.get('msg', '')}".strip(": "))
				message = "; ".join(parts) or message
			elif isinstance(detail, str):
				message = detail
		except ValueError:
			pass
		self.message = message

		text = f"HTTP {self.status_code}"
		if self.code:
			text += f" [{self.code}]"
		if self.param:
			text += f" (param: {self.param})"
		text += f": {message}"
		super().__init__(text)


def synthesize(text: str, conf: dict, retries: int = 4) -> bytes:
	"""One ElevenLabs call, returning raw little-endian 16-bit PCM."""
	model = "eleven_multilingual_v2" if conf["high_quality"] else "eleven_turbo_v2_5"
	url = f"{ELEVENLABS_API_URL}/{conf['voice_id']}?output_format={OUTPUT_FORMAT}"

	delay = 2.0
	for attempt in range(retries):
		try:
			r = requests.post(
				url,
				headers={"xi-api-key": conf["key"], "Content-Type": "application/json"},
				json={
					"text": text,
					"model_id": model,
					"voice_settings": {
						"stability": conf["stability"],
						"similarity_boost": conf["similarity_boost"],
						"style": conf["style"],
						"use_speaker_boost": True,
					},
				},
				timeout=60,
			)
			if r.status_code == 429 or r.status_code >= 500:
				# Rate limited or a transient server fault — back off.
				if attempt < retries - 1:
					print(f"    HTTP {r.status_code}, retrying in {delay:.0f}s...")
					time.sleep(delay)
					delay *= 2
					continue
			if not r.ok:
				# Any other error is our request's fault and won't fix itself
				# on retry. Surface ElevenLabs' own explanation rather than a
				# bare "400 Bad Request".
				raise ElevenLabsError(r)
			return r.content
		except ElevenLabsError:
			raise
		except requests.RequestException as e:
			if attempt < retries - 1:
				print(f"    {e}, retrying in {delay:.0f}s...")
				time.sleep(delay)
				delay *= 2
				continue
			raise
	raise RuntimeError("exhausted retries")


def write_metadata(out_dir: str, prompts: list) -> int:
	"""Rebuild metadata.csv from whichever clips currently exist on disk.

	Regenerated wholesale each run rather than appended, so a partial or
	interrupted run always leaves a consistent file.
	"""
	rows = []
	for text in prompts:
		name = clip_name(text)
		if os.path.exists(os.path.join(out_dir, "wav", name)):
			# Piper's format is pipe-delimited: filename|text
			rows.append(f"{name}|{text}")
	with open(os.path.join(out_dir, "metadata.csv"), "w", encoding="utf-8") as f:
		f.write("\n".join(rows) + ("\n" if rows else ""))
	return len(rows)


def report(out_dir: str, prompts: list) -> None:
	wav_dir = os.path.join(out_dir, "wav")
	total = 0.0
	short = []
	long_ = []
	count = 0
	for text in prompts:
		path = os.path.join(wav_dir, clip_name(text))
		if not os.path.exists(path):
			continue
		d = wav_duration(path)
		total += d
		count += 1
		if d < MIN_CLIP_S:
			short.append((d, text))
		elif d > MAX_CLIP_S:
			long_.append((d, text))

	print()
	print("=" * 68)
	print(f"  Clips:          {count} of {len(prompts)} prompts")
	print(f"  Total audio:    {total / 60:.1f} minutes")
	if count:
		print(f"  Mean clip:      {total / count:.1f}s")
	print(f"  Target:         30-45 minutes ({'MET' if total >= 1800 else f'{(1800 - total) / 60:.1f} min short'})")

	if short:
		print(f"\n  {len(short)} clip(s) under {MIN_CLIP_S}s — too brief to carry prosody,")
		print(f"  consider lengthening these prompts:")
		for d, t in short[:5]:
			print(f"    {d:4.1f}s  {t[:56]}")
	if long_:
		print(f"\n  {len(long_)} clip(s) over {MAX_CLIP_S}s — may be dropped by Piper's")
		print(f"  phoneme cap during training, consider splitting:")
		for d, t in long_[:5]:
			print(f"    {d:4.1f}s  {t[:56]}")
	print("=" * 68)


# -----------------------------------------------------------------------------

def main() -> None:
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--config", default="", help="path to config.cfg (default: repo root, then /mnt/usb)")
	ap.add_argument("--prompts", default="",
	                help="prompt list (default: next to this script, then repo root)")
	ap.add_argument("--out", default=os.path.join(_BASE_DIR, "voice_dataset"),
	                help="output directory")
	ap.add_argument("--limit", type=int, default=0, help="stop after N new clips (0 = no limit)")
	ap.add_argument("--delay", type=float, default=0.4, help="seconds between API calls")
	ap.add_argument("--dry-run", action="store_true", help="show cost estimate, generate nothing")
	ap.add_argument("--stats", action="store_true", help="report on existing clips, generate nothing")
	ap.add_argument("--force", action="store_true", help="regenerate clips that already exist")
	args = ap.parse_args()

	prompts_path = resolve_prompts(args.prompts)
	prompts = load_prompts(prompts_path)
	wav_dir = os.path.join(args.out, "wav")

	if args.stats:
		report(args.out, prompts)
		return

	existing = sum(
		1 for t in prompts if os.path.exists(os.path.join(wav_dir, clip_name(t)))
	)
	todo = [t for t in prompts
	        if args.force or not os.path.exists(os.path.join(wav_dir, clip_name(t)))]
	if args.limit:
		todo = todo[:args.limit]

	chars = sum(len(t) for t in todo)
	print(f"Prompts:        {len(prompts)}  ({prompts_path})")
	print(f"Already done:   {existing}")
	print(f"To generate:    {len(todo)}")
	print(f"Characters:     {chars:,}  (ElevenLabs bills per character —")
	print(f"                check this against your quota before running)")

	if args.dry_run:
		print("\nDry run, nothing generated.")
		return
	if not todo:
		print("\nNothing to do.")
		report(args.out, prompts)
		return

	conf = load_tts_config(args.config)
	model = "eleven_multilingual_v2" if conf["high_quality"] else "eleven_turbo_v2_5"
	print(f"Config:         {conf['path']}")
	print(f"Voice:          {conf['voice_id']}  model={model}")
	print(f"Settings:       stability={conf['stability']} "
	      f"similarity={conf['similarity_boost']} style={conf['style']}")
	print()

	os.makedirs(wav_dir, exist_ok=True)
	total_s = 0.0
	failed = []

	for i, text in enumerate(todo, 1):
		name = clip_name(text)
		path = os.path.join(wav_dir, name)
		preview = text if len(text) <= 52 else text[:49] + "..."
		print(f"[{i}/{len(todo)}] {preview}")

		try:
			raw = synthesize(text, conf)
			pcm = np.frombuffer(raw, dtype="<i2")
			if pcm.size == 0:
				raise RuntimeError("empty audio returned")
			pcm = trim_silence(pcm)
			dur = write_wav(path, pcm)
			total_s += dur
			flag = ""
			if dur < MIN_CLIP_S:
				flag = "  <-- very short"
			elif dur > MAX_CLIP_S:
				flag = "  <-- very long"
			print(f"    {dur:.1f}s -> {name}{flag}")
		except ElevenLabsError as e:
			print(f"    FAILED: {e}")
			failed.append(text)
			if e.status_code in (400, 401, 402, 403, 404, 422):
				# Bad key, voice, setting or quota: every remaining prompt
				# would fail the same way, so stop instead of repeating it.
				print("\n  This error isn't specific to one prompt, so stopping here.")
				print(f"  Config used: {conf['path']}")
				print(f"  Voice ID:    {conf['voice_id']!r}")
				print(f"  Model:       {model}  output_format={OUTPUT_FORMAT}")
				print("  Fix the setting above and re-run; finished clips are kept.")
				break
		except Exception as e:
			print(f"    FAILED: {e}")
			failed.append(text)

		# Be polite to the API between calls.
		if i < len(todo):
			time.sleep(args.delay)

	n = write_metadata(args.out, prompts)
	print(f"\nWrote metadata.csv with {n} entries.")
	print(f"Generated {total_s / 60:.1f} minutes this run.")
	if failed:
		print(f"\n{len(failed)} prompt(s) failed — re-run to retry just those:")
		for t in failed[:10]:
			print(f"  {t[:60]}")

	report(args.out, prompts)
	print(f"\nDataset: {args.out}")
	print("Train on a rented NVIDIA GPU (e.g. RunPod) with:")
	print("  bash train_piper_runpod.sh setup")
	print("  bash train_piper_runpod.sh train <voice_name> /workspace/voice_dataset")
	print("  bash train_piper_runpod.sh export <voice_name>")
	print("See the header of tools/train_piper_runpod.sh for details.")


if __name__ == "__main__":
	main()
