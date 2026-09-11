#!/usr/bin/env python3
"""Cut a release of velend.

	scripts/release.py v0.3.0

Sets the version in the manifest, builds the extension zips into build/, then
amends the manifest into HEAD and tags it.

Amending rewrites HEAD. If HEAD is already on the remote, the push at the end
needs --force-with-lease; the script says so when it notices.

Nothing is pushed. Push the branch and the tag yourself once the zips look right.
"""

import argparse
import re
import subprocess
import sys

from blender_env import ROOT, fail


MANIFEST = ROOT / "blender_manifest.toml"

TAG_RE = re.compile(r"^v[0-9]+\.[0-9]+\.[0-9]+$")
VERSION_RE = re.compile(r'^version = "(.*)"$', re.MULTILINE)


def run(*argv, capture=False):
	"""Run a command from ROOT, failing the release if it does.

	With `capture`, its stdout comes back stripped instead of going to the
	terminal. Anything whose failure is an answer rather than an error goes
	through succeeds() instead.
	"""
	result = subprocess.run(
		argv, cwd=ROOT, text=True,
		stdout=subprocess.PIPE if capture else None,
	)
	if result.returncode != 0:
		fail("%s exited with %d" % (argv[0], result.returncode))
	return result.stdout.strip() if capture else ""


def succeeds(*argv):
	"""Whether a command exits 0, with both of its outputs discarded."""
	result = subprocess.run(argv, cwd=ROOT, capture_output=True)
	return result.returncode == 0


def upstream_branch():
	"""The remote branch HEAD tracks, if it tracks one."""
	result = subprocess.run(
		["git", "rev-parse", "-q", "--verify", "--abbrev-ref", "@{upstream}"],
		cwd=ROOT, text=True, capture_output=True,
	)
	return result.stdout.strip() if result.returncode == 0 else ""


def current_version():
	match = VERSION_RE.search(MANIFEST.read_text(encoding="utf-8"))
	if not match:
		fail("no version in %s" % MANIFEST)
	return match.group(1)


def set_version(version):
	text = MANIFEST.read_text(encoding="utf-8")
	updated = VERSION_RE.sub('version = "%s"' % version, text, count=1)
	MANIFEST.write_text(updated, encoding="utf-8")
	if current_version() != version:
		fail("failed to set the version in %s" % MANIFEST)


def main(argv=None):
	parser = argparse.ArgumentParser(
		description=__doc__,
		formatter_class=argparse.RawDescriptionHelpFormatter,
	)
	parser.add_argument("tag", metavar="vMAJOR.MINOR.PATCH")
	args = parser.parse_args(argv)

	tag = args.tag
	if not TAG_RE.match(tag):
		fail("tag should look like v0.3.0, got '%s'" % tag)
	version = tag[1:]

	# A release has to be reproducible from the tag, so refuse to build one on
	# top of edits that are not in the commit being tagged.
	if not succeeds("git", "diff-index", "--quiet", "HEAD", "--"):
		fail("working tree has uncommitted changes")
	if succeeds("git", "rev-parse", "-q", "--verify", "refs/tags/" + tag):
		fail("tag %s already exists" % tag)

	# Amending a commit that is already published rewrites history someone else
	# may have. Not a refusal -- it is your branch -- but it should not be a
	# surprise.
	upstream = upstream_branch()
	pushed = bool(upstream) and succeeds("git", "merge-base", "--is-ancestor", "HEAD", upstream)

	print("== releasing %s -> %s as %s" % (current_version(), version, tag))
	if pushed:
		print("   HEAD is already on %s; amending it rewrites published history"
			% upstream)

	# Until the amend lands, any exit undoes the version bump and whatever
	# scripts/build.py rewrote in the manifest.
	amended = False
	try:
		set_version(version)

		print()
		run(sys.executable, str(ROOT / "scripts" / "build.py"))

		print()
		print("== amending HEAD and tagging")
		run("git", "add", str(MANIFEST))
		# --no-edit, because the commit keeps the message it already has.
		run("git", "commit", "--amend", "--no-edit")
		amended = True
	finally:
		if not amended:
			run("git", "checkout", "--", str(MANIFEST))

	run("git", "tag", "-a", tag, "-m", "velend " + tag)

	branch = run("git", "rev-parse", "--abbrev-ref", "HEAD", capture=True)
	print()
	print("tagged %s at %s"
		% (tag, run("git", "rev-parse", "--short", "HEAD", capture=True)))
	if pushed:
		print("push it with: git push --force-with-lease origin %s"
			" && git push origin %s" % (branch, tag))
	else:
		print("push it with: git push origin %s %s" % (branch, tag))


if __name__ == "__main__":
	sys.exit(main())
