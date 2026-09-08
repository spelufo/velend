"""Asking Blender about itself.

Shared by build.py and develop.py. The paths involved differ per platform and
per Blender version, so nothing here guesses at them: Blender is started once
and asked to print what it knows.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent

# Blender's Python ships these already. A second copy is at best dead weight and
# at worst a different build of numpy one sys.path entry away from the renderer.
PROVIDED_BY_BLENDER = {
	"numpy", "requests", "urllib3", "certifi", "idna", "charset-normalizer",
	"packaging", "typing-extensions", "zstandard",
}

_QUERY = """
import bpy, sys
print("VELEND_PYTHON", sys.executable)
print("VELEND_PYTAG", "%d.%d" % sys.version_info[:2])
print("VELEND_EXTENSIONS", bpy.utils.user_resource("EXTENSIONS"))
"""


def fail(message):
	raise SystemExit("error: " + message)


def extension_id():
	"""The `id` the manifest gives the extension, which names its directory."""
	for line in (ROOT / "blender_manifest.toml").read_text().splitlines():
		if line.startswith("id = "):
			return line.split("=", 1)[1].strip().strip('"')
	fail("no id in %s/blender_manifest.toml" % ROOT)


class Blender:
	"""The `blender` on PATH, and the parts of it these scripts drive."""

	def __init__(self):
		# A shell alias does not count: this has to be something exec'able.
		found = shutil.which("blender")
		if found is None:
			fail("no blender on PATH")
		# Resolved, because Blender locates its own resources relative to the
		# binary and comes up empty when it is reached through a symlink.
		self.path = str(Path(found).resolve())

		# One startup, several answers. `--factory-startup` keeps an already
		# installed copy of velend from loading and failing on the very
		# dependency the caller is about to set up.
		result = subprocess.run(
			[self.path, "--background", "--factory-startup", "--python-expr", _QUERY],
			capture_output=True, text=True,
		)
		if result.returncode != 0:
			sys.stderr.write(result.stderr)
			fail("%s exited with %d" % (self.path, result.returncode))

		answers = {}
		for line in result.stdout.splitlines():
			if line.startswith("VELEND_"):
				key, _, value = line[len("VELEND_"):].partition(" ")
				answers[key] = value
		missing = {"PYTHON", "PYTAG", "EXTENSIONS"} - set(answers)
		if missing:
			fail("%s did not report %s" % (self.path, ", ".join(sorted(missing))))

		self.python = answers["PYTHON"]
		self.pytag = answers["PYTAG"]
		self.extensions = Path(answers["EXTENSIONS"])

	def pip(self, *args):
		"""Run pip in Blender's own Python.

		PYTHONNOUSERSITE keeps pip away from ~/.local, which Blender's embedded
		interpreter ignores anyway (it starts Python with user site-packages
		disabled), and whose stale .pth files can break pip's startup outright.
		"""
		env = dict(os.environ, PYTHONNOUSERSITE="1")
		self.run([self.python, "-m", "pip", *args], env=env)

	def command(self, *args):
		"""Run one of `blender --command`'s subcommands."""
		self.run([self.path, "--command", *args])

	@staticmethod
	def run(argv, env=None):
		result = subprocess.run(argv, env=env)
		if result.returncode != 0:
			fail("%s exited with %d" % (argv[0], result.returncode))
