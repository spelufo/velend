"""The open data metadata: which samples, scans, volumes and segments exist.

The whole catalogue lives in one big `metadata.json` at the root of the open
data bucket. On startup we fetch it into a cache next to the extension's other
user files, reusing what is already cached when the server says it has not
changed, and parse it into the read-only objects below.

Nothing here mutates its objects after `load` built them, so the parsed
catalogue is safe to read from the loader threads.
"""

import gzip
import json
import os
import threading
import urllib.error
import urllib.request
from datetime import datetime, timezone
from email.utils import formatdate

import bpy


OPEN_DATA_BUCKET = "vesuvius-challenge-open-data"
OPEN_DATA_URL = "https://%s.s3.us-east-1.amazonaws.com" % OPEN_DATA_BUCKET
OPEN_DATA_MANIFEST_URL = OPEN_DATA_URL + "/metadata.json"

# The parsed catalogue, keyed by id. Ids are timestamps, unique across samples,
# so the flat dictionaries are enough to look anything up; each object also
# points at the ones it belongs to and holds the ones below it.
SAMPLES = {}
SCANS = {}
VOLUMES = {}
SEGMENTS = {}


class Sample:
	"""The physical sample. A scroll or a fragment."""

	def __init__(self, entry):
		sample = entry["sample"]
		properties = sample.get("properties") or {}
		self.id = sample["id"]
		self.type = properties.get("type")  # "scroll", "fragment", or None
		self.description = sample.get("description")
		# The matrices that take one of this sample's volumes into another,
		# straight from the manifest, keyed by the volume they come from.
		self.volume_transforms = {
			transform["from_volume_id"]: transform["transforms"]
			for transform in (properties.get("volume_transforms") or [])
		}
		self.data = sample.get("data") or []

		self.scans = {}
		self.volumes = {}
		self.segments = {}

	def __repr__(self):
		return "<Sample %s>" % self.id


class Scan:
	"""One scanning session at an imaging facility."""

	def __init__(self, scan, sample):
		creation = scan.get("creation") or {}
		metadata = creation.get("metadata") or {}
		properties = scan.get("properties") or {}
		self.id = scan["id"]
		self.long_id = scan["long_id"]
		self.sample = sample
		self.created_at = _parse_date(creation.get("date"))
		self.name = metadata.get("scan_name")
		self.location = metadata.get("location")
		self.energy_keV = properties.get("energy_keV")
		self.pixel_size_um = properties.get("pixel_size_um")
		self.detector_distance_mm = properties.get("detector_distance_mm")

		# The reconstructions made from this scan, filled in as they parse.
		self.volumes = {}

	def __repr__(self):
		return "<Scan %s>" % self.long_id


class Volume:
	"""A 3D volume, an OME-Zarr. One reconstruction of one scan, of which there
	may be several: at different resolutions, masked, or otherwise processed."""

	def __init__(self, volume, sample, scan):
		creation = volume.get("creation") or {}
		properties = volume.get("properties") or {}
		self.id = volume["id"]
		self.long_id = volume["long_id"]
		self.sample = sample
		self.scan = scan
		self.created_at = _parse_date(creation.get("date"))
		self.energy_keV = properties.get("energy_keV")
		self.pixel_size_um = properties.get("pixel_size_um")
		self.data_format = properties.get("data_format")
		# The manifest writes the shape the way zarr indexes it, (Z, Y, X).
		shape = properties.get("shape")
		self.shape = tuple(shape) if shape else None
		self.license = properties.get("license")
		self.data = volume.get("data") or []

		# The segments traced through this volume, filled in as they parse.
		self.segments = {}

	@property
	def zarr_url(self):
		"""The URL of the pyramid itself, the one `state.open_volume` takes."""
		return data_url(self.data, "ome-zarr")

	def __repr__(self):
		return "<Volume %s>" % self.long_id


class Segment:
	"""A mesh that follows a sheet of papyrus through a volume."""

	def __init__(self, segment, sample, volume):
		creation = segment.get("creation") or {}
		properties = segment.get("properties") or {}
		self.id = segment["id"]
		self.long_id = segment["long_id"]
		self.name = segment.get("suffix")
		self.sample = sample
		# The volume it was traced in. Its meshes are in that volume's voxels,
		# except for the "tifxyz-transformed" ones, which name their own.
		self.volume = volume
		self.created_at = _parse_date(creation.get("date"))
		self.width = properties.get("width")
		self.height = properties.get("height")
		self.data = segment.get("data") or []

	def tifxyz_url(self, volume=None):
		"""The URL of the segment's tifxyz, in the given volume's voxels.

		Defaults to the volume the segment was traced in, whose tifxyz needs no
		transform and is stored on its own.
		"""
		if volume is None or volume is self.volume or volume == getattr(self.volume, "id", None):
			return data_url(self.data, "tifxyz")
		volume_id = volume if isinstance(volume, str) else volume.id
		return data_url(self.data, "tifxyz-transformed", target_volume=volume_id)

	def __repr__(self):
		return "<Segment %s>" % self.long_id


def _parse_date(text):
	"""One of the manifest's UTC timestamps, or None if it holds none."""
	if not text:
		return None
	# Most dates end in "Z", which `fromisoformat` only learned to read in
	# Python 3.11; others spell the offset out, and a handful do both.
	if text.endswith("Z"):
		text = text[:-1]
		if not text.endswith("+00:00"):
			text += "+00:00"
	try:
		return datetime.fromisoformat(text).astimezone(timezone.utc)
	except ValueError:
		# A date we cannot read is not worth refusing the whole catalogue over.
		return None


def _origin_url(origin):
	"""An https URL for one of a data entry's origins, or None for one we
	cannot reach.

	Everything downloadable is public. All but a handful of origins name the
	open data bucket by its s3 URL, which we turn into the https one that zarr
	and urllib can both read; the rest name an https mirror already.
	"""
	for root in origin["access_roots"]:
		if root["type"] == "s3":
			if root["url"][len("s3://"):].strip("/") == OPEN_DATA_BUCKET:
				return OPEN_DATA_URL + "/" + origin["path"]
		elif root["type"] == "https":
			return root["url"].rstrip("/") + "/" + origin["path"]
	return None


def data_url(data, type, **parameters):
	"""The URL of the first entry of `type` in a `data` list whose parameters
	match the ones given, or None when the object has no such entry.

	An entry can be mirrored in several places. The open data bucket is the one
	kept up to date, so it wins over any other, whichever is listed first.
	"""
	fallback = None
	for entry in data:
		if entry["type"] != type:
			continue
		if any(
			(entry.get("parameters") or {}).get(key) != value
			for key, value in parameters.items()
		):
			continue
		for origin in entry["origins"]:
			url = _origin_url(origin)
			if url is not None and url.startswith(OPEN_DATA_URL):
				return url
			if fallback is None:
				fallback = url
	return fallback


def parse(manifest):
	"""Build the catalogue from a parsed `metadata.json`."""
	samples, scans, volumes, segments = {}, {}, {}, {}
	for entry in manifest["samples"].values():
		sample = Sample(entry)
		samples[sample.id] = sample

		for scan_json in entry["scans"].values():
			scan = Scan(scan_json, sample)
			scans[scan.id] = scan
			sample.scans[scan.id] = scan

		for volume_json in entry["volumes"].values():
			scan = scans.get(volume_json.get("scan_id"))
			volume = Volume(volume_json, sample, scan)
			volumes[volume.id] = volume
			sample.volumes[volume.id] = volume
			if scan is not None:
				scan.volumes[volume.id] = volume

		for segment_json in entry["segments"].values():
			volume = volumes.get(segment_json.get("original_volume_id"))
			segment = Segment(segment_json, sample, volume)
			segments[segment.id] = segment
			sample.segments[segment.id] = segment
			if volume is not None:
				volume.segments[segment.id] = segment

	return samples, scans, volumes, segments


def merge(base, overlay):
	"""Merge an overlay manifest into a parsed one, in place.

	Dictionaries merge key by key, so an overlay only has to spell out the path
	down to what it changes. Anything else, lists included, replaces what was
	there: a sample's "volume_transforms" is one list, and half of a list of
	matrices is not a useful thing to merge.
	"""
	for key, value in overlay.items():
		if isinstance(value, dict) and isinstance(base.get(key), dict):
			merge(base[key], value)
		else:
			base[key] = value
	return base


def overlay_path():
	"""The extra manifest named in the add-on preferences, if any."""
	addon = bpy.context.preferences.addons.get(__package__)
	if addon is None or not getattr(addon.preferences, "overlay_path", ""):
		return ""
	# Blender stores "//" relative paths, which only mean something once
	# resolved against the current .blend.
	return bpy.path.abspath(addon.preferences.overlay_path)


def cache_path():
	"""Where the downloaded manifest is kept between sessions."""
	directory = bpy.utils.extension_path_user(__package__, path="cache", create=True)
	return os.path.join(directory, "metadata.json")


def download(path):
	"""Refresh the cached manifest at `path` from the bucket.

	Sends the cached copy's date along, so a manifest that has not changed
	since costs one request and no download.
	"""
	request = urllib.request.Request(OPEN_DATA_MANIFEST_URL)
	if os.path.exists(path):
		request.add_header(
			"If-Modified-Since", formatdate(os.path.getmtime(path), usegmt=True)
		)
	try:
		with urllib.request.urlopen(request) as response:
			body = response.read()
			# The manifest is stored gzipped, and urllib does not ask for or
			# undo any encoding by itself. The cache keeps the plain JSON.
			if response.headers.get("Content-Encoding") == "gzip":
				body = gzip.decompress(body)
	except urllib.error.HTTPError as error:
		if error.code == 304:
			return False
		raise

	# Through a temporary file, so an interrupted download cannot leave a
	# truncated manifest behind for the next session to parse.
	temporary = path + ".part"
	with open(temporary, "wb") as file:
		file.write(body)
	os.replace(temporary, path)
	return True


def load(path=None, overlay=None, refresh=True):
	"""Fill the catalogue in, downloading the manifest when it is out of date.

	Falls back to the cached manifest when the download fails, so the add-on
	still works offline once it has run online. `overlay` names a JSON file
	shaped like the manifest, merged over it: a way to work with transforms,
	segments or volumes that have not reached the bucket yet.
	"""
	if path is None:
		path = cache_path()
	if refresh:
		try:
			download(path)
		except OSError as error:
			print("velend: could not fetch %s: %s" % (OPEN_DATA_MANIFEST_URL, error))
	if not os.path.exists(path):
		return False

	with open(path) as file:
		manifest = json.load(file)

	if overlay:
		try:
			with open(overlay) as file:
				merge(manifest, json.load(file))
		except (OSError, ValueError) as error:
			# A bad overlay leaves the catalogue as the bucket has it, rather
			# than leaving the add-on with none at all.
			print("velend: could not merge %s: %s" % (overlay, error))

	# Parsed whole before anything is published, so readers never see a
	# half-built catalogue.
	samples, scans, volumes, segments = parse(manifest)
	global SAMPLES, SCANS, VOLUMES, SEGMENTS
	SAMPLES, SCANS, VOLUMES, SEGMENTS = samples, scans, volumes, segments
	return True


def load_in_background():
	"""`load` on a thread, so downloading and parsing 15MB of JSON does not
	hold up Blender's startup.

	The paths are resolved here, on the calling thread, since they are the
	only calls in `load` that go through `bpy`.
	"""
	thread = threading.Thread(
		target=load,
		args=(cache_path(), overlay_path()),
		name="velend-metadata",
		daemon=True,
	)
	thread.start()
	return thread
