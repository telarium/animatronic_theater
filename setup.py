#!/usr/bin/env python3
import getpass
import glob
import importlib
import json
import os
import platform
import pwd
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from typing import Dict, List, Optional, Tuple

class Setup:
	def __init__(self) -> None:
		# List of system packages to install (from apt)
		packages: List[str] = [
			"git", "build-essential", "cmake", "wget", "bzip2", "python3-dev", "flex", "bison",
			"mpv", "hostapd", "dnsmasq",
			# No python3-flask: Flask is pinned via pip below, and the apt copy
			# only drags in a Debian-owned Werkzeug 3.x for pip to fight with.
			"python3-smbus", "python3-evdev", "python3-setuptools", "python3-mido",
			"python3-pip",
			"python3-psutil", "python3-pydispatch", "python3-pygame", "iw",
			"iproute2",
			# ALSA audio
			"alsa-utils", "portaudio19-dev", "ffmpeg",
			# libusb backend for pyusb (ReSpeaker control via xvf_host)
			"libusb-1.0-0",
		]
		self._run_command("sudo apt update")
		self.install_packages(packages)
		self.install_python_packages([
			# Flask-SocketIO and its dependencies — pinned for compatibility.
			# Do NOT install python3-flask-socketio via apt; the system version
			# is too old and causes AttributeError on _request_ctx_stack.
			"werkzeug==2.3.7",
			"flask==2.3.3",
			"flask-socketio==5.3.6",
			"python-socketio==5.10.0",
			"python-engineio==4.8.0",
			# Required for WebSocket in async_mode='threading'. Without it
			# Flask-SocketIO can only serve polling, which adds an HTTP
			# round trip to every key event.
			"simple-websocket",
			# numpy pinned to <2.0 for openwakeword compatibility
			"numpy<2.0",
			# Other pip-only packages. openwakeword is NOT here — see
			# install_openwakeword() for why it needs its own install step.
			"pvporcupine", "rapidfuzz", "pydub", "scipy", "openai", "elevenlabs", "piper-tts",
			"pywifi", "flask-talisman", "requests", "pyudev", "anthropic", "smbus2",
			# USB — pip-only, no apt equivalents. Both are imported by xvf_host.py
			# (the repo copy and the one setup_respeaker() clones); without
			# libusb-package, ReSpeaker control is disabled at startup.
			"pyusb", "libusb-package",
			# STT. sherpa-onnx bundles its own onnxruntime, so nothing else is
			# needed here. webrtcvad and pyaudio were dropped with whisper.cpp:
			# the streaming transducer does its own endpointing, and both the
			# wakeword and STT capture paths use arecord rather than PortAudio.
			"sherpa-onnx",
			# USB-serial for ProgramBlue / PL2303 adapter
			"pyserial",
		])
		self.install_openwakeword()
		self.setup_sherpa_models()
		self.setup_llama()
		self.setup_openwakeword_models()
		self.setup_respeaker()
		self.setup_pl2303()
		self.setup_f_midi()
		self.setup_system_tuning()
		self.setup_bashrc()

	# ─── Who the app runs as ──────────────────────────────────────────────────

	@staticmethod
	def _real_user() -> str:
		"""The account the app runs as: SUDO_USER when run via sudo, otherwise
		whoever is running the script. Never a hardcoded name — each character
		runs under its own account."""
		return os.environ.get("SUDO_USER") or getpass.getuser()

	def _real_home(self) -> str:
		"""Home directory of _real_user(). Not expanduser("~"): under sudo on
		Ubuntu, HOME is /root, so "~" would point at root's files."""
		return pwd.getpwnam(self._real_user()).pw_dir

	# ─── apt / pip ────────────────────────────────────────────────────────────

	def install_packages(self, packages: List[str]) -> None:
		try:
			subprocess.check_call(["sudo", "apt", "install", "-y"] + packages)
		except subprocess.CalledProcessError as e:
			print(f"Failed to install packages: {e}")
			sys.exit(1)

	def _ensure_modern_pip(self) -> None:
		"""--break-system-packages needs pip >= 23. Ubuntu 24.04 ships pip 24,
		and there pip must NOT be upgraded: the Debian-managed pip can't be
		uninstalled by pip ("installed by debian"), so the upgrade itself
		fails. Only older systems (Ubuntu 22.04 ships pip 22) get upgraded."""
		result = subprocess.run(
			[sys.executable, "-m", "pip", "--version"],
			capture_output=True, text=True,
		)
		try:
			major = int(result.stdout.split()[1].split(".")[0])
		except (IndexError, ValueError):
			major = 0
		if major >= 23:
			return
		print(f"pip {major or 'unknown'} is too old for --break-system-packages, upgrading...")
		subprocess.check_call(["sudo", sys.executable, "-m", "pip", "install", "--upgrade", "pip"])

	def _pip_install(self, args: List[str]) -> None:
		# --ignore-installed: on Ubuntu 24.04, Python packages installed by apt
		# have no pip RECORD file, so pip can't uninstall them to put a pinned
		# or newer version in place ("Cannot uninstall Werkzeug 3.0.1, RECORD
		# file not found. Hint: The package was installed by debian."). With
		# this flag pip never uninstalls; it writes its own copy to
		# /usr/local/lib/python3.X/dist-packages, which comes before the apt
		# copies in sys.path, so ours are the ones imported.
		subprocess.check_call(
			["sudo", sys.executable, "-m", "pip", "install",
			 "--break-system-packages", "--ignore-installed"] + args
		)

	def install_python_packages(self, packages: List[str]) -> None:
		self._ensure_modern_pip()
		# One pip call for everything, so the resolver sees every pin at once.
		# Installing one package per call let a later package quietly upgrade
		# something an earlier pin held back (numpy<2.0 in particular).
		try:
			self._pip_install(packages)
		except subprocess.CalledProcessError as e:
			print(f"Failed to install Python packages: {e}")
			sys.exit(1)

	# openwakeword declares tflite-runtime as a hard dependency on Linux, and
	# tflite-runtime has no wheels for Python 3.12+ (Ubuntu 24.04 / JetPack 7),
	# so a plain install fails to resolve. Our models are .onnx and openwakeword
	# falls back to onnxruntime when tflite isn't installed, so on 3.12+ it is
	# installed without its declared deps plus the deps the onnx path needs.
	OPENWAKEWORD_SPEC = "openwakeword>=0.5.0,<1"
	OPENWAKEWORD_ONNX_DEPS = ["onnxruntime", "tqdm", "scipy", "scikit-learn", "requests"]

	def install_openwakeword(self) -> None:
		try:
			if sys.version_info < (3, 12):
				self._pip_install([self.OPENWAKEWORD_SPEC, "numpy<2.0"])
			else:
				print("openwakeword: Python 3.12+, installing without tflite-runtime (onnx only).")
				self._pip_install(["--no-deps", self.OPENWAKEWORD_SPEC])
				# numpy pin repeated so these deps can't drag numpy to 2.x.
				self._pip_install(self.OPENWAKEWORD_ONNX_DEPS + ["numpy<2.0"])
		except subprocess.CalledProcessError as e:
			print(f"Failed to install openwakeword: {e}")
			sys.exit(1)

	# ─── Speech-to-text model ─────────────────────────────────────────────────

	# Streaming transducer used by speech_to_text.py.
	SHERPA_MODEL = "sherpa-onnx-nemotron-speech-streaming-en-0.6b-560ms-int8-2026-04-25"
	SHERPA_MODEL_URL = (
		"https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
		f"{SHERPA_MODEL}.tar.bz2"
	)

	def setup_sherpa_models(self) -> None:
		"""Download the Nemotron streaming transducer into lib/sherpa_onnx/models/.
		"""
		script_dir = os.path.dirname(os.path.abspath(__file__))
		models_dir = os.path.join(script_dir, "lib", "sherpa_onnx", "models")
		model_dir  = os.path.join(models_dir, self.SHERPA_MODEL)

		# Check for an actual model file, not just the directory — a download
		# interrupted mid-extract leaves the directory behind and would
		# otherwise be treated as a completed install.
		if os.path.exists(os.path.join(model_dir, "encoder.int8.onnx")):
			print(f"Sherpa model {self.SHERPA_MODEL} already present, skipping.")
			return

		os.makedirs(models_dir, exist_ok=True)
		archive = os.path.join(models_dir, f"{self.SHERPA_MODEL}.tar.bz2")
		print(f"Downloading sherpa-onnx model: {self.SHERPA_MODEL} (~630MB)")
		try:
			subprocess.check_call(["wget", "-O", archive, self.SHERPA_MODEL_URL])
			subprocess.check_call(["tar", "xf", archive, "-C", models_dir])
		except subprocess.CalledProcessError as e:
			print(f"Failed to set up sherpa-onnx model: {e}")
			sys.exit(1)
		finally:
			if os.path.exists(archive):
				os.remove(archive)

		print(f"Sherpa model installed to {model_dir}.")

	# ─── Offline LLM (llama.cpp) ──────────────────────────────────────────────

	# llama.cpp moves fast and flag behaviour changes between releases, so
	# this is pinned rather than tracking master.
	LLAMA_REPO = "https://github.com/ggml-org/llama.cpp.git"
	LLAMA_COMMIT = "b96806d96"
	LLAMA_MODEL_REPO = "ggml-org/Qwen3-1.7B-GGUF"
	# Tried in order. The repo does not necessarily carry every quant, so the
	# actual filename is discovered from the HF API rather than hardcoded.
	LLAMA_QUANT_PREFERENCE = ["Q4_K_M", "Q8_0", "Q4_0", "BF16", "f16"]
	# Sanity floor for "is this actually a model?". The 1.7B Q4_K_M is ~1.2GB;
	# llama.cpp's own ggml-vocab-*.gguf test fixtures are 600KB-16MB, and a
	# wget that lands on an HTML error page is smaller still.
	LLAMA_MODEL_MIN_BYTES = 500 * 1024 * 1024
	LLAMA_PORT = 8081  # the port llm_service.py's offline fallback tier calls
	# Every Jetson Orin module is sm_87. Set explicitly rather than letting
	# cmake probe the GPU, so the build doesn't depend on detection working.
	LLAMA_CUDA_ARCH = "87"

	def _find_cuda_root(self) -> Optional[str]:
		"""Locate the CUDA toolkit. Deliberately does NOT rely on PATH: the
		exports setup_bashrc writes don't apply to the running process, and on
		JetPack the toolkit is always under /usr/local."""
		nvcc = shutil.which("nvcc")
		if nvcc:
			return os.path.dirname(os.path.dirname(nvcc))
		candidates = sorted(glob.glob("/usr/local/cuda*/bin/nvcc"), reverse=True)
		if candidates:
			return os.path.dirname(os.path.dirname(candidates[0]))
		return None

	def _cuda_env(self, cuda_root: str) -> dict:
		env = os.environ.copy()
		env["PATH"] = f"{cuda_root}/bin:" + env.get("PATH", "")
		env["LD_LIBRARY_PATH"] = f"{cuda_root}/lib64:" + env.get("LD_LIBRARY_PATH", "")
		return env

	def setup_llama(self) -> None:
		"""Build llama.cpp with CUDA into lib/llama/, fetch the GGUF, and
		install the llama-server unit. The build takes 10-15 minutes."""
		script_dir = os.path.dirname(os.path.abspath(__file__))
		llama_dir  = os.path.join(script_dir, "lib", "llama")
		binary     = os.path.join(llama_dir, "build", "bin", "llama-server")

		cuda_root = self._find_cuda_root()
		if cuda_root is None:
			print("llama: CUDA toolkit not found under /usr/local. Install it "
			      "(sudo apt install nvidia-jetpack) and re-run.")
			return
		print(f"llama: using CUDA at {cuda_root}")

		if os.path.isfile(binary):
			print("llama.cpp already built, skipping build.")
		else:
			self._build_llama(llama_dir, cuda_root)

		# NOT lib/llama/models/. The llama.cpp checkout ships ~25 vocab-only
		# ggml-vocab-*.gguf test fixtures in that directory; treating them as
		# candidate models makes the download step no-op and hands systemd a
		# file with no weights, which fails at load with the fairly opaque
		# "check_tensor_dims: tensor 'token_embd.weight' not found".
		model_path = self._download_llama_model(os.path.join(script_dir, "models"))
		if model_path is None:
			print("llama: no model available — skipping service install.")
			return

		self._install_llama_service(binary, model_path, cuda_root)

	def _build_llama(self, llama_dir: str, cuda_root: str) -> None:
		if not os.path.isdir(llama_dir):
			print("Cloning llama.cpp...")
			subprocess.check_call(["git", "clone", self.LLAMA_REPO, llama_dir])
		subprocess.check_call(["git", "-C", llama_dir, "fetch", "--all", "--tags"])
		subprocess.check_call(["git", "-C", llama_dir, "checkout", self.LLAMA_COMMIT])

		env = self._cuda_env(cuda_root)
		print("Building llama.cpp with CUDA (10-15 minutes)...")
		try:
			subprocess.check_call(
				["cmake", "-B", "build", "-DGGML_CUDA=ON",
				 f"-DCMAKE_CUDA_COMPILER={cuda_root}/bin/nvcc",
				 f"-DCMAKE_CUDA_ARCHITECTURES={self.LLAMA_CUDA_ARCH}"],
				cwd=llama_dir, env=env,
			)
			subprocess.check_call(
				["cmake", "--build", "build", "--config", "Release", "-j6"],
				cwd=llama_dir, env=env,
			)
		except subprocess.CalledProcessError as e:
			print(f"llama: build failed: {e}")
			sys.exit(1)

	def _existing_llama_model(self, models_dir: str) -> Optional[str]:
		"""Return a usable .gguf already in models_dir, or None.

		Sorted for determinism — os.listdir order is arbitrary, so an
		unsorted first-match is not reproducible between machines.
		"""
		for name in sorted(os.listdir(models_dir)):
			if not name.endswith(".gguf"):
				continue
			path = os.path.join(models_dir, name)
			size = os.path.getsize(path)
			if size >= self.LLAMA_MODEL_MIN_BYTES:
				print(f"llama: model already present ({name}), skipping download.")
				return path
			print(f"llama: ignoring {name} — {size} bytes is too small to be a model.")
		return None

	def _download_llama_model(self, models_dir: str) -> Optional[str]:
		os.makedirs(models_dir, exist_ok=True)

		existing = self._existing_llama_model(models_dir)
		if existing:
			return existing

		try:
			url = f"https://huggingface.co/api/models/{self.LLAMA_MODEL_REPO}"
			with urllib.request.urlopen(url, timeout=30) as response:
				data = json.load(response)
		except Exception as e:
			print(f"llama: could not list {self.LLAMA_MODEL_REPO}: {e}")
			return None

		available = [
			s["rfilename"] for s in data.get("siblings", [])
			if s.get("rfilename", "").endswith(".gguf")
		]
		if not available:
			print(f"llama: no .gguf files found in {self.LLAMA_MODEL_REPO}.")
			return None

		filename = None
		for quant in self.LLAMA_QUANT_PREFERENCE:
			match = next((f for f in available if quant.lower() in f.lower()), None)
			if match:
				filename = match
				break
		if filename is None:
			filename = available[0]
			print(f"llama: no preferred quant in {available}, using {filename}.")

		path = os.path.join(models_dir, os.path.basename(filename))
		url = f"https://huggingface.co/{self.LLAMA_MODEL_REPO}/resolve/main/{filename}"
		print(f"Downloading {filename} (~1.2GB)...")
		try:
			subprocess.check_call(["wget", "-O", path, url])
		except subprocess.CalledProcessError as e:
			print(f"llama: model download failed: {e}")
			if os.path.exists(path):
				os.remove(path)
			return None

		# wget exits 0 on a served error page, so check_call alone proves
		# nothing about what actually landed on disk.
		size = os.path.getsize(path)
		if size < self.LLAMA_MODEL_MIN_BYTES:
			print(f"llama: downloaded {filename} is only {size} bytes — expected ~1.2GB. "
			      "The server likely returned an error page rather than the model.")
			os.remove(path)
			return None

		print(f"llama: model installed to {path}.")
		return path

	# The unit is generated here rather than shipped as a file: every value in
	# it is install-specific, and lib/ is gitignored so a template couldn't
	# live next to the build it describes.
	#
	# --fit off and --cache-ram 0: on JetPack 7.2/7.2.1 the Orin Nano's GPU
	# memory allocator (NvMap) crashes under llama.cpp's automatic memory
	# fitting and host-RAM prompt cache. Turning both off, with an explicit
	# context size (-c), is the fix reported on NVIDIA's forums. Harmless on
	# JetPack 6, where the crash doesn't occur.
	LLAMA_SERVICE_NAME = "llama-server.service"
	LLAMA_SERVICE_UNIT = """[Unit]
Description=llama-server (offline LLM fallback for the animatronic)
After=network.target

[Service]
Type=simple
User={user}
# systemd does not source .bashrc, so CUDA has to be on the path here or the
# CUDA build fails to load libcudart at runtime.
Environment=PATH={cuda_root}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
Environment=LD_LIBRARY_PATH={cuda_root}/lib64
ExecStart={binary} -m {model_path} -ngl 99 -c 2048 --fit off --cache-ram 0 --jinja --host 127.0.0.1 --port {port}
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
"""

	def _install_llama_service(self, binary: str, model_path: str, cuda_root: str) -> None:
		unit = self.LLAMA_SERVICE_UNIT.format(
			user=self._real_user(),
			cuda_root=cuda_root,
			binary=binary,
			model_path=model_path,
			port=self.LLAMA_PORT,
		)

		staged = os.path.join(tempfile.gettempdir(), self.LLAMA_SERVICE_NAME)
		destination = f"/etc/systemd/system/{self.LLAMA_SERVICE_NAME}"
		try:
			with open(staged, "w") as f:
				f.write(unit)
			subprocess.check_call(["sudo", "cp", staged, destination])
			subprocess.check_call(["sudo", "systemctl", "daemon-reload"])
			# Re-run replaces the unit, so restart rather than start — enable
			# --now is a no-op on an already-running service and would leave
			# the old paths live.
			subprocess.check_call(["sudo", "systemctl", "enable", self.LLAMA_SERVICE_NAME])
			subprocess.check_call(["sudo", "systemctl", "restart", self.LLAMA_SERVICE_NAME])
		except (subprocess.CalledProcessError, OSError) as e:
			print(f"llama: could not install the service: {e}")
			return
		finally:
			if os.path.exists(staged):
				os.remove(staged)

		# A unit that fails to load its model exits after ~1s and is then
		# restarted, so an immediate is-active check reports "activating" and
		# looks like success. Model load itself takes a few seconds. Wait past
		# both before believing it.
		print("llama: waiting for the service to settle...")
		time.sleep(20)
		result = subprocess.run(
			["systemctl", "is-active", self.LLAMA_SERVICE_NAME],
			capture_output=True, text=True,
		)
		if result.stdout.strip() != "active":
			print(f"llama: service is not running (state: {result.stdout.strip()}). "
			      "Check: journalctl -u llama-server -n 40")
			return

		print(f"llama-server installed at {destination} and running on "
		      f"127.0.0.1:{self.LLAMA_PORT}.")

	# ─── Wakeword / ReSpeaker ─────────────────────────────────────────────────

	def setup_openwakeword_models(self) -> None:
		"""Download openWakeWord base models and install every custom wakeword
		model shipped in lib/openwakeword/ (one per character)."""
		try:
			# This process started before pip installed openwakeword, so the
			# import system's directory caches may predate it.
			importlib.invalidate_caches()
			import openwakeword
			import openwakeword.utils
			openwakeword.utils.download_models()
			print("openWakeWord base models downloaded.")
		except Exception as e:
			print(f"Failed to download openWakeWord models: {e}")
			sys.exit(1)

		# Copy the custom wakeword models into the openwakeword resources
		# folder. wakeword_detection.py loads them by absolute path from
		# lib/openwakeword/ (see the character JSON), so this copy is
		# belt-and-braces rather than required. The destination comes from
		# the installed package itself — it used to be hardcoded to
		# python3.10, which doesn't exist on Ubuntu 24.04 (Python 3.12).
		script_dir = os.path.dirname(os.path.abspath(__file__))
		src_dir = os.path.join(script_dir, "lib", "openwakeword")
		dst_dir = os.path.join(os.path.dirname(openwakeword.__file__), "resources", "models")
		models = sorted(glob.glob(os.path.join(src_dir, "*.onnx")))
		if not models:
			print(f"WARNING: no .onnx wakeword models in {src_dir} — copy them there manually.")
			return
		subprocess.check_call(["sudo", "mkdir", "-p", dst_dir])
		for src in models:
			subprocess.check_call(["sudo", "cp", src, dst_dir])
			print(f"{os.path.basename(src)} installed to openwakeword models folder.")

	def setup_respeaker(self) -> None:
		"""Clone ReSpeaker XVF3800 host control tools into lib/respeaker/."""
		script_dir = os.path.dirname(os.path.abspath(__file__))
		respeaker_dir = os.path.join(script_dir, "lib", "respeaker")

		if not os.path.isdir(respeaker_dir):
			print("Cloning ReSpeaker XVF3800 repo...")
			subprocess.check_call([
				"git", "clone",
				"https://github.com/respeaker/reSpeaker_XVF3800_USB_4MIC_ARRAY.git",
				respeaker_dir
			])
		else:
			print("ReSpeaker repo already present, skipping clone.")

		print("ReSpeaker xvf_host ready.")

	# ─── Out-of-tree kernel modules (shared) ──────────────────────────────────

	# Plain-text files from the stable tree, on the branch the running kernel
	# tracks. Building from the matching series keeps the source compatible
	# with the kernel's internal APIs, which change between major versions.
	#
	# Tried in order; all files for one build come from the same mirror.
	# git.kernel.org now answers scripted downloads with 403 Forbidden (bot
	# protection), so the primary source is Greg Kroah-Hartman's GitHub mirror
	# of the same stable tree, which carries every linux-X.Y.y branch.
	KERNEL_SOURCE_URLS = [
		"https://raw.githubusercontent.com/gregkh/linux/linux-{series}.y/{path}",
		"https://git.kernel.org/pub/scm/linux/kernel/git/stable/linux.git/plain/"
		"{path}?h=linux-{series}.y",
	]
	# urllib's default "Python-urllib/3.x" agent is exactly what bot filters
	# block; identify as curl, which both mirrors serve.
	KERNEL_SOURCE_HEADERS = {"User-Agent": "curl/8.5.0"}

	@staticmethod
	def _kernel_series(kernel_ver: str) -> str:
		"""'6.8.12-tegra' -> '6.8', the stable branch the Tegra kernel tracks."""
		return ".".join(kernel_ver.split("-")[0].split(".")[:2])

	@staticmethod
	def _kernel_version_tuple(kernel_ver: str) -> Tuple[int, int]:
		parts = kernel_ver.split("-")[0].split(".")
		try:
			return int(parts[0]), int(parts[1])
		except (IndexError, ValueError):
			return 0, 0

	@staticmethod
	def _module_dir(kernel_ver: str) -> str:
		"""Where our modules are installed. updates/ takes priority over the
		kernel's own modules in depmod, and JetPack kernel packages never
		write there — so an apt upgrade can't overwrite our builds with stock
		ones (that is how the MIDI port name regressed before)."""
		return f"/lib/modules/{kernel_ver}/updates"

	@staticmethod
	def _kernel_has_module(name: str) -> bool:
		"""True if the running kernel can load `name` (built in or as a module)."""
		result = subprocess.run(["modinfo", "-n", name], capture_output=True, text=True)
		return result.returncode == 0

	@staticmethod
	def _kernel_headers(kernel_ver: str, tag: str) -> Optional[str]:
		headers = f"/lib/modules/{kernel_ver}/build"
		if not os.path.exists(headers):
			print(f"{tag}: no kernel headers at {headers}. Install them "
			      "(sudo apt install nvidia-l4t-kernel-headers) and re-run.")
			return None
		return os.path.realpath(headers)

	def _download_kernel_sources(self, series: str, sources: Dict[str, str],
	                             build_dir: str, tag: str) -> bool:
		for template in self.KERNEL_SOURCE_URLS:
			host = urllib.parse.urlparse(template).hostname
			try:
				for name, path in sources.items():
					url = template.format(path=path, series=series)
					request = urllib.request.Request(url, headers=self.KERNEL_SOURCE_HEADERS)
					with urllib.request.urlopen(request, timeout=30) as response:
						data = response.read()
					if not data:
						raise RuntimeError(f"{name} downloaded empty from {url}")
					with open(os.path.join(build_dir, name), "wb") as f:
						f.write(data)
			except Exception as e:
				print(f"{tag}: could not download sources from {host} — {e}")
				continue
			print(f"{tag}: sources downloaded from {host}.")
			return True
		print(f"{tag}: no mirror had the linux-{series}.y sources.")
		return False

	@staticmethod
	def _module_vermagic(path: str) -> Optional[str]:
		try:
			result = subprocess.run(
				["modinfo", "-F", "vermagic", path],
				capture_output=True, text=True, check=True,
			)
		except (subprocess.CalledProcessError, OSError):
			return None
		fields = result.stdout.split()
		return fields[0] if fields else None

	@staticmethod
	def _install_module(module: str, install_dir: str, install_path: str, tag: str) -> None:
		print(f"{tag}: installing to {install_path}...")
		subprocess.check_call(["sudo", "mkdir", "-p", install_dir])
		subprocess.check_call(["sudo", "cp", module, install_path])
		subprocess.check_call(["sudo", "depmod", "-a"])

	# ─── PL2303 USB-serial (ProgramBlue) ──────────────────────────────────────

	PL2303_SOURCES = {
		"pl2303.c": "drivers/usb/serial/pl2303.c",
		"pl2303.h": "drivers/usb/serial/pl2303.h",
	}
	# Our adapter is a PL2303GC reporting bcdDevice 0x105. Upstream added that
	# ID in 5.16 (backported to 5.15 stable); the copy in lib/pl2303/ was
	# patched by hand because it came from the original 5.15 release.
	PL2303_GC_CASE = "case 0x105:"
	PL2303_GC_FIXED_IN = (5, 16)

	def setup_pl2303(self) -> None:
		"""Make sure a pl2303 driver that recognises our adapter is loaded.

		The stock JetPack 6 kernel has no pl2303 at all. Order of preference:
		our own build already installed for this kernel; the kernel's own
		driver if it's new enough to know the GC's 0x105 ID; a fresh build
		from the stable sources for this kernel's series; and, offline, the
		old patched copy in lib/pl2303/ (only builds against 5.15 kernels).
		"""
		script_dir   = os.path.dirname(os.path.abspath(__file__))
		legacy_dir   = os.path.join(script_dir, "lib", "pl2303")
		kernel_ver   = platform.release()
		install_dir  = self._module_dir(kernel_ver)
		install_path = os.path.join(install_dir, "pl2303.ko")

		if not self._kernel_has_module("usbserial"):
			print("pl2303: this kernel has no usbserial support — skipping. "
			      "ProgramBlue over USB-serial will not work.")
			return

		if self._module_vermagic(install_path) == kernel_ver:
			print(f"pl2303: our build is already installed for {kernel_ver}, skipping build.")
		elif (self._kernel_has_module("pl2303")
		      and self._kernel_version_tuple(kernel_ver) >= self.PL2303_GC_FIXED_IN):
			print(f"pl2303: kernel {kernel_ver} ships a pl2303 driver that already "
			      "recognises the GC adapter — using it, no build needed.")
		else:
			built = self._build_pl2303(kernel_ver)
			if built is not None:
				self._install_module(built, install_dir, install_path, "pl2303")
				shutil.rmtree(os.path.dirname(built), ignore_errors=True)
			elif not self._build_pl2303_legacy(legacy_dir, kernel_ver, install_dir, install_path):
				print("pl2303: no driver available — ProgramBlue over USB-serial will not work.")
				return

		print("pl2303: loading modules...")
		try:
			subprocess.check_call(["sudo", "modprobe", "usbserial"])
			subprocess.check_call(["sudo", "modprobe", "pl2303"])
		except subprocess.CalledProcessError as e:
			print(f"pl2303: modprobe failed — {e}. Check: sudo dmesg | tail")
			return

		print("pl2303: enabling on boot...")
		subprocess.check_call(
			"printf 'usbserial\\npl2303\\n' | sudo tee /etc/modules-load.d/pl2303.conf",
			shell=True,
		)

		# Add the real user (not root) to the dialout group so /dev/ttyUSB*
		# is accessible without sudo.
		real_user = self._real_user()
		subprocess.check_call(["sudo", "usermod", "-aG", "dialout", real_user])
		print(f"pl2303: added '{real_user}' to dialout group — re-login required to take effect.")

		print("pl2303: done.")

	def _build_pl2303(self, kernel_ver: str) -> Optional[str]:
		"""Download and build pl2303.ko for this kernel. Returns the built
		module's path (inside a temp dir the caller removes), or None."""
		headers = self._kernel_headers(kernel_ver, "pl2303")
		if headers is None:
			return None

		series = self._kernel_series(kernel_ver)
		build_dir = tempfile.mkdtemp(prefix="pl2303_")
		print(f"pl2303: building for {kernel_ver} from linux-{series}.y sources in {build_dir}...")

		if not self._download_kernel_sources(series, self.PL2303_SOURCES, build_dir, "pl2303"):
			shutil.rmtree(build_dir, ignore_errors=True)
			return None

		with open(os.path.join(build_dir, "pl2303.c"), "r") as f:
			if self.PL2303_GC_CASE not in f.read():
				# Shouldn't happen on any stable branch we'd see, but building
				# without it gives a driver that silently ignores the adapter.
				print(f"pl2303: linux-{series}.y source lacks '{self.PL2303_GC_CASE}' — "
				      "it would not recognise the GC adapter. Not building it.")
				shutil.rmtree(build_dir, ignore_errors=True)
				return None

		with open(os.path.join(build_dir, "Kbuild"), "w") as f:
			f.write("obj-m := pl2303.o\n")

		try:
			subprocess.check_call(["make", "-C", headers, f"M={build_dir}", "modules"])
		except subprocess.CalledProcessError as e:
			print(f"pl2303: build failed — {e}")
			shutil.rmtree(build_dir, ignore_errors=True)
			return None

		built = os.path.join(build_dir, "pl2303.ko")
		if self._module_vermagic(built) != kernel_ver:
			print("pl2303: the built module doesn't match the running kernel — not installing it.")
			shutil.rmtree(build_dir, ignore_errors=True)
			return None
		return built

	def _build_pl2303_legacy(self, legacy_dir: str, kernel_ver: str,
	                         install_dir: str, install_path: str) -> bool:
		"""Offline fallback: the hand-patched 5.15 source in lib/pl2303/."""
		if not os.path.isdir(legacy_dir):
			return False
		if self._kernel_series(kernel_ver) != "5.15":
			print(f"pl2303: lib/pl2303/ holds 5.15 source, which won't build "
			      f"against {kernel_ver}. Connect to the internet and re-run.")
			return False
		print("pl2303: falling back to the patched source in lib/pl2303/...")
		try:
			subprocess.check_call(["make", "-C", legacy_dir, "all"])
		except subprocess.CalledProcessError as e:
			print(f"pl2303: legacy build failed — {e}")
			return False
		self._install_module(
			os.path.join(legacy_dir, "src", "pl2303.ko"), install_dir, install_path, "pl2303"
		)
		return True

	# ─── USB MIDI gadget (usb_f_midi) ─────────────────────────────────────────

	# The interface string Windows (and Cakewalk) show as the MIDI port name.
	# Stock f_midi.c says "MIDI function"; the build below patches it to this.
	# This is NOT set by midi.py — its "product" string is a different field.
	F_MIDI_INTERFACE_NAME = "Animatronic MIDI"
	F_MIDI_STOCK_NAME     = "MIDI function"
	# u_f.h lives one directory up from the function sources in the kernel
	# tree. Fetching it from function/ returns an empty file and the build
	# fails on a missing header.
	F_MIDI_SOURCES = {
		"f_midi.c": "drivers/usb/gadget/function/f_midi.c",
		"u_midi.h": "drivers/usb/gadget/function/u_midi.h",
		"u_f.h":    "drivers/usb/gadget/u_f.h",
	}
	# NVIDIA's USB device-mode service binds the only UDC at boot, leaving
	# nothing for the MIDI gadget. JetPack upgrades re-enable it.
	L4T_USB_DEVICE_MODE_SERVICE = "nv-l4t-usb-device-mode.service"

	def setup_f_midi(self) -> None:
		"""Install usb_f_midi.ko for the running kernel, with the port name
		patched to F_MIDI_INTERFACE_NAME.

		The stock Tegra kernel ships without CONFIG_USB_CONFIGFS_F_MIDI, so the
		module is built out of tree. Even where a kernel does ship it, ours is
		still installed, because the stock one carries the generic port name.
		A module only loads on the exact kernel it was built for, so every
		JetPack kernel update needs this re-run:

			python3 -c "from setup import Setup; Setup.setup_f_midi(Setup.__new__(Setup))"

		Order of preference: the module already installed (if it matches this
		kernel and carries the patched name), then the copy in lib/f_midi/
		(same checks), then a fresh build from the upstream stable sources.
		"""
		script_dir   = os.path.dirname(os.path.abspath(__file__))
		repo_copy    = os.path.join(script_dir, "lib", "f_midi", "usb_f_midi.ko")
		kernel_ver   = platform.release()
		install_dir  = self._module_dir(kernel_ver)
		install_path = os.path.join(install_dir, "usb_f_midi.ko")

		self._disable_l4t_usb_device_mode()

		if self._f_midi_module_ok(install_path, kernel_ver):
			print(f"f_midi: patched module already installed for {kernel_ver}, skipping build.")
		elif self._f_midi_module_ok(repo_copy, kernel_ver):
			print(f"f_midi: installing the copy in lib/f_midi/ (built for {kernel_ver}).")
			self._install_module(repo_copy, install_dir, install_path, "f_midi")
		else:
			built = self._build_f_midi(kernel_ver)
			if built is None:
				print("f_midi: MIDI over USB-C will not be available until this succeeds.")
				return
			self._install_module(built, install_dir, install_path, "f_midi")
			# Keep a copy in the repo so a reinstall on this kernel needs no
			# network. Failure here doesn't affect the running install.
			try:
				os.makedirs(os.path.dirname(repo_copy), exist_ok=True)
				shutil.copy2(built, repo_copy)
				print(f"f_midi: saved a copy to {repo_copy}.")
			except OSError as e:
				print(f"f_midi: could not save a copy to lib/f_midi/ ({e}) — not fatal.")
			finally:
				shutil.rmtree(os.path.dirname(built), ignore_errors=True)

		print("f_midi: loading modules...")
		try:
			subprocess.check_call(["sudo", "modprobe", "libcomposite"])
			subprocess.check_call(["sudo", "modprobe", "usb_f_midi"])
		except subprocess.CalledProcessError as e:
			print(f"f_midi: modprobe failed — {e}. Check: sudo dmesg | tail")
			return

		print("f_midi: done. Reboot, then the gadget comes up when the app starts.")
		print("f_midi: on Windows, uninstall the old MIDI device in Device Manager "
		      "(View > Show hidden devices) and replug, or the old name stays cached.")

	def _f_midi_module_ok(self, path: str, kernel_ver: str) -> bool:
		"""True if path is a usb_f_midi.ko for this kernel with the patched
		name. A module that loads but still says "MIDI function" is NOT ok —
		that is exactly the regression the last kernel update caused."""
		if not os.path.isfile(path):
			return False
		vermagic = self._module_vermagic(path)
		if vermagic != kernel_ver:
			print(f"f_midi: {path} is built for {vermagic or 'an unknown kernel'}, "
			      f"not {kernel_ver}.")
			return False
		try:
			with open(path, "rb") as f:
				patched = self.F_MIDI_INTERFACE_NAME.encode() in f.read()
		except OSError:
			return False
		if not patched:
			print(f"f_midi: {path} matches the kernel but still uses the stock "
			      f"port name '{self.F_MIDI_STOCK_NAME}'.")
		return patched

	def _build_f_midi(self, kernel_ver: str) -> Optional[str]:
		"""Download, patch and build usb_f_midi.ko. Returns the path of the
		built module (inside a temp dir the caller removes), or None."""
		headers = self._kernel_headers(kernel_ver, "f_midi")
		if headers is None:
			return None

		series = self._kernel_series(kernel_ver)
		build_dir = tempfile.mkdtemp(prefix="usb_f_midi_")
		print(f"f_midi: building for {kernel_ver} from linux-{series}.y sources in {build_dir}...")

		if not self._download_kernel_sources(series, self.F_MIDI_SOURCES, build_dir, "f_midi"):
			shutil.rmtree(build_dir, ignore_errors=True)
			return None

		source_path = os.path.join(build_dir, "f_midi.c")
		with open(source_path, "r") as f:
			source = f.read()
		stock = f'"{self.F_MIDI_STOCK_NAME}"'
		count = source.count(stock)
		if count != 1:
			# Upstream changed the string. Building anyway would silently bring
			# back the generic port name, so stop and say why.
			print(f"f_midi: expected exactly one {stock} in f_midi.c, found {count}. "
			      "Upstream has changed — the name patch needs updating.")
			shutil.rmtree(build_dir, ignore_errors=True)
			return None
		with open(source_path, "w") as f:
			f.write(source.replace(stock, f'"{self.F_MIDI_INTERFACE_NAME}"'))
		print(f"f_midi: port name patched to '{self.F_MIDI_INTERFACE_NAME}'.")

		with open(os.path.join(build_dir, "Kbuild"), "w") as f:
			f.write("obj-m := usb_f_midi.o\nusb_f_midi-y := f_midi.o\n")

		try:
			subprocess.check_call(["make", "-C", headers, f"M={build_dir}", "modules"])
		except subprocess.CalledProcessError as e:
			print(f"f_midi: build failed — {e}")
			shutil.rmtree(build_dir, ignore_errors=True)
			return None

		built = os.path.join(build_dir, "usb_f_midi.ko")
		if not self._f_midi_module_ok(built, kernel_ver):
			print("f_midi: the built module failed its own checks — not installing it.")
			shutil.rmtree(build_dir, ignore_errors=True)
			return None
		return built

	def _disable_l4t_usb_device_mode(self) -> None:
		"""Stop NVIDIA's USB device-mode gadget from claiming the UDC.

		Masked rather than just disabled: a JetPack upgrade re-enables a
		disabled unit (that is how MIDI broke after the last apt upgrade), but
		it cannot enable a masked one.
		"""
		service = self.L4T_USB_DEVICE_MODE_SERVICE
		result = subprocess.run(
			["systemctl", "list-unit-files", service],
			capture_output=True, text=True,
		)
		if service not in result.stdout:
			return
		state = subprocess.run(
			["systemctl", "is-enabled", service], capture_output=True, text=True,
		).stdout.strip()
		if state == "masked":
			print(f"f_midi: {service} already masked.")
			return
		print(f"f_midi: disabling and masking {service} (it holds the USB-C port MIDI needs)...")
		print("f_midi: NOTE — this also removes the USB-C serial console and the "
		      "192.168.55.1 USB network. After reboot, reach this board over "
		      "Wi-Fi/Ethernet (or Tailscale) only.")
		subprocess.run(["sudo", "systemctl", "disable", "--now", service])
		subprocess.check_call(["sudo", "systemctl", "mask", service])

	# ─── System tuning ────────────────────────────────────────────────────────

	WIFI_POWERSAVE_CONF = "/etc/NetworkManager/conf.d/wifi-powersave-off.conf"

	def setup_system_tuning(self) -> None:
		"""Settings for an unattended appliance on networks we don't control.

		Wi-Fi power saving: the radio dozes between packets, which shows up as
		laggy web UI responses and dropped connections on busy venue Wi-Fi.
		Takes effect at the next reboot — NetworkManager isn't restarted here,
		since that would drop an SSH session running this script.

		unattended-upgrades: it can install a new kernel on its own, after
		which pl2303 and usb_f_midi no longer load until this script is
		re-run. Updates should happen when we choose, with a re-run after.
		"""
		subprocess.check_call(
			f"printf '[connection]\\nwifi.powersave = 2\\n' | sudo tee {self.WIFI_POWERSAVE_CONF} > /dev/null",
			shell=True,
		)
		print("system: Wi-Fi power saving disabled (applies after reboot).")

		installed = subprocess.run(
			["dpkg-query", "-W", "-f=${Status}", "unattended-upgrades"],
			capture_output=True, text=True,
		).stdout
		if "install ok installed" in installed:
			print("system: removing unattended-upgrades...")
			subprocess.check_call(["sudo", "apt", "purge", "-y", "unattended-upgrades"])
		else:
			print("system: unattended-upgrades not installed.")

	# ─── Shell environment ────────────────────────────────────────────────────

	def setup_bashrc(self) -> None:
		"""Add required environment variables to the app user's ~/.bashrc if
		not already present.

		Written to the real user's home, not "~": under sudo, "~" is /root.
		"""
		bashrc = os.path.join(self._real_home(), ".bashrc")
		# The CUDA exports are required by the llama.cpp build in setup_llama —
		# without them cmake won't find nvcc. Nothing in the STT path needs
		# them any more (sherpa-onnx runs the int8 model on CPU), so do not
		# drop them on that basis. llama-server itself gets its CUDA paths
		# from the systemd unit, not from here.
		#
		# The old PYTHONPATH=~/.local/lib/python3.10/site-packages export is
		# gone: everything is installed with sudo pip into /usr/local, which is
		# already on Python's path, and the python3.10 directory doesn't exist
		# on Ubuntu 24.04. Existing .bashrc files keep the old line; it's
		# harmless.
		exports = [
			"export PATH=/usr/local/cuda/bin:$PATH",
			"export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH",
		]
		try:
			with open(bashrc, "r") as f:
				current = f.read()
		except FileNotFoundError:
			current = ""
		with open(bashrc, "a") as f:
			for line in exports:
				if line not in current:
					f.write(f"\n{line}")
					print(f"Added to {bashrc}: {line}")
		print("bashrc updated. Run 'source ~/.bashrc' or re-login to apply.")

	def _run_command(self, command: str) -> None:
		try:
			subprocess.check_call(command, shell=True)
		except subprocess.CalledProcessError as e:
			print(f"Command failed: {command}\nError: {e}")
			sys.exit(1)

if __name__ == "__main__":
	Setup()
