#!/usr/bin/env python3
"""Set velend up for development.

Fills wheels/, and links this checkout into Blender's user extensions
directory, so that edits here are picked up by "Reload Scripts" without a
reinstall.

	scripts/develop.py
	scripts/develop.py --unlink

Dependencies come from the same wheels a released package ships: Blender
unpacks whatever the manifest lists into its own extensions site-packages when
the extension is enabled, whether that extension is an unzipped download or a
link to a checkout.

The manifest is left alone here. Its wheel list belongs to scripts/build.py,
so this downloads the whole list rather than just this machine's platform, and
says so when the two have drifted apart.
"""

import argparse
import sys

from blender_env import ROOT, Blender, extension_id, fail
from build import PLATFORMS, WHEELS_DIR, fetch_wheels, manifest_wheels


def check_manifest():
	"""Warn when wheels/ and the manifest's list have parted ways.

	They do whenever a dependency releases a new version, since pip fetches the
	newest match and the manifest still names the old file. Blender then looks
	for a wheel that is not there, so it is worth saying out loud.
	"""
	listed = set(manifest_wheels())
	present = {wheel.name for wheel in WHEELS_DIR.glob("*.whl")}
	if listed == present:
		return
	print()
	print("warning: wheels/ no longer matches the manifest's list.")
	for name in sorted(listed - present):
		print("  listed, missing: %s" % name)
	for name in sorted(present - listed):
		print("  present, unlisted: %s" % name)
	print("Run scripts/build.py to bring the manifest up to date.")


def link_path(blender):
	return blender.extensions / "user_default" / extension_id()


def unlink(link):
	if link.is_symlink():
		link.unlink()
		print("unlinked %s" % link)
	else:
		print("nothing to unlink at %s" % link)


def link(link_to_make):
	print("== linking %s -> %s" % (link_to_make, ROOT))
	link_to_make.parent.mkdir(parents=True, exist_ok=True)
	if link_to_make.is_symlink():
		link_to_make.unlink()
	elif link_to_make.exists():
		fail(
			"%s exists and is not a symlink.\n"
			"       It is probably a copy installed from a zip. Remove it in\n"
			"       Preferences > Add-ons, or delete it, then run this again."
			% link_to_make
		)
	link_to_make.symlink_to(ROOT)


def main(argv=None):
	parser = argparse.ArgumentParser(
		description=__doc__,
		formatter_class=argparse.RawDescriptionHelpFormatter,
	)
	parser.add_argument(
		"--unlink", action="store_true",
		help="remove the link again, leaving wheels/ in place",
	)
	args = parser.parse_args(argv)

	blender = Blender()
	if args.unlink:
		unlink(link_path(blender))
		return

	print("blender:    %s" % blender.path)
	print("python:     %s" % blender.pytag)
	print("extensions: %s" % blender.extensions)
	print()

	fetch_wheels(blender, list(PLATFORMS))
	check_manifest()
	print()
	link(link_path(blender))

	print()
	print('Done. Enable it in Blender under Preferences > Add-ons, search "%s".'
		% extension_id())
	print("Blender unpacks the wheels the first time it is enabled.")
	print('After editing the source, run Blender\'s "Reload Scripts" (F3 > Reload Scripts).')


if __name__ == "__main__":
	sys.exit(main())
