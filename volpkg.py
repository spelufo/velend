"""VC3D projects: the volumes a .volpkg.json lists, and where VC3D keeps them.

A project names its volumes by location, which for the open data ones is a
bucket URL rather than anything on disk. VC3D mirrors what it reads through
into a cache directory named after that URL, which is the same directory our
own mirror store reads through and fills in: pointing both at it means either
program's downloads count for the other.

Nothing writes that naming down anywhere the two could agree on, so it is
reimplemented here, after volume-cartographer's `deriveRemoteVolumeId` and
`remoteVolumeCacheRootForEntry` in `core/src/Volume.cpp` and
`core/src/VolumePkg.cpp`, and `remoteCachePath` in `RemoteCacheSettings.cpp`.
"""

import json
import os


# FNV-1a, 64 bit, the hash volume-cartographer names cache directories with.
_FNV_OFFSET_BASIS = 14695981039346656037
_FNV_PRIME = 1099511628211
_UINT64 = (1 << 64) - 1

_REMOTE_SCHEMES = ("s3://", "s3+", "http://", "https://")

# The open data tags a project entry carries about a volume. There are many
# more; these are the ones that say something we can use.
_SAMPLE_TAG = "vc-open-data-sample-id:"
_VOLUME_TAG = "vc-open-data-volume-id:"
_VOXEL_SIZE_TAG = "vc-open-data-voxel-size-um:"
_PREFERRED_TAG = "vc-open-data-preferred-source"

_BASE_SCALE_SELECTOR = "vc-base-scale="


def base_scale(location):
	"""The pyramid level a location's "#vc-base-scale=N" asks to be treated as
	level zero, or 0 for the locations that ask for nothing.

	It is a client side selector: the same pyramid, entered further down. We
	read every level a volume has, so it only tells us how much coarser than
	the pyramid's own level zero the entry's voxel size is quoted at.
	"""
	fragment = location.partition("#")[2]
	if not fragment.startswith(_BASE_SCALE_SELECTOR):
		return 0
	try:
		return int(fragment[len(_BASE_SCALE_SELECTOR):])
	except ValueError:
		return 0


def fnv1a(text):
	hash = _FNV_OFFSET_BASIS
	for byte in text.encode("utf-8"):
		hash = ((hash ^ byte) * _FNV_PRIME) & _UINT64
	return hash


def is_remote(location):
	"""Whether a project's location names something to fetch over the network
	rather than a path on this machine."""
	return location.startswith(_REMOTE_SCHEMES)


def source_url(location):
	"""The plain https URL VC3D would fetch a remote location through.

	S3 buckets are addressed over https all the same, with the region in the
	hostname; `s3://` leaves it out and means us-east-1.
	"""
	# A "#vc-base-scale=N" fragment picks the pyramid level to treat as level
	# zero. Neither the cache directory's name nor a request includes it.
	url = location.split("#", 1)[0]
	if url.startswith("s3://") or url.startswith("s3+"):
		scheme, separator, rest = url.partition("://")
		if separator:
			region = scheme[len("s3+"):] or "us-east-1"
			bucket, _, key = rest.partition("/")
			url = "https://%s.s3.%s.amazonaws.com" % (bucket, region)
			if key:
				url += "/" + key
	return url.rstrip("/")


def volume_id(url):
	"""The id VC3D gives a remote volume, and so the name of its cache
	directory: what the URL ends in, and a hash of the whole of it.

	The hash is what keeps two volumes that end in the same name, from
	different samples or mirrors, in cache directories of their own.
	"""
	return "%s-%016x" % (url.rsplit("/", 1)[-1] or "remote", fnv1a(url))


def sample_directory(sample_id):
	"""The directory VC3D groups a sample's cached volumes under: its id, with
	anything that is not a plain path character replaced."""
	name = "".join(
		character
		if (character.isalnum() and character.isascii()) or character in "-_."
		else "_"
		for character in sample_id
	)
	return name.lstrip("._") or "sample"


def settings_path():
	"""VC3D.ini, wherever this machine's VC3D would look for it."""
	directory = os.environ.get("VC3D_CONFIG_DIR")
	if not directory:
		directory = os.path.join(os.path.expanduser("~"), ".VC3D")
	return os.path.join(directory, "VC3D.ini")


def _decode_ini(value):
	"""One Qt settings value, unquoted and unescaped."""
	value = value.strip()
	if len(value) >= 2 and value.startswith('"') and value.endswith('"'):
		value = value[1:-1]
	escapes = {"\\": "\\", '"': '"', "n": "\n", "r": "\r", "t": "\t"}
	decoded = []
	index = 0
	while index < len(value):
		if value[index] != "\\" or index + 1 == len(value):
			decoded.append(value[index])
			index += 1
			continue
		# Anything else keeps its backslash: Qt writes "\x" escapes we have no
		# reason to read, and a path is likelier to want the backslash back.
		decoded.append(escapes.get(value[index + 1], value[index:index + 2]))
		index += 2
	return "".join(decoded)


def _ini_value(path, section, key):
	"""One value out of a settings file, or "" when it holds none.

	Enough of the ini dialect for the one key we read: Qt's own writer escapes
	and percent-encodes far more of it than a straight parse would survive.
	"""
	try:
		with open(path, encoding="utf-8", errors="replace") as file:
			lines = file.read().splitlines()
	except OSError:
		return ""
	in_section = False
	for line in lines:
		line = line.strip()
		if not line or line[0] in ";#":
			continue
		if line.startswith("[") and line.endswith("]"):
			in_section = line == "[%s]" % section
		elif in_section:
			name, separator, value = line.partition("=")
			if separator and name.strip() == key:
				return _decode_ini(value)
	return ""


def remote_cache_root():
	"""Where VC3D mirrors what it reads out of a remote volume.

	The same search `vc::settings::remoteCachePath` does. A project file may
	carry a "remote_cache_root" of its own, but VC3D stopped writing that and
	never reads it back, so neither do we.
	"""
	configured = _ini_value(settings_path(), "viewer", "remote_cache_dir")
	if configured:
		return os.path.abspath(os.path.expanduser(configured))
	# Deployments that mount one of these share a cache across users.
	for root in ("/volpkgs", "/ephemeral"):
		if os.path.isdir(root):
			return os.path.join(root, "remote_cache")
	return os.path.join(os.path.expanduser("~"), ".VC3D", "remote_cache")


def cache_limits():
	"""How much room VC3D's settings allow its cache: `(maximum, free)`, both
	in bytes, the maximum None when they put no ceiling on it.

	VC3D hands these to the budget that counts every chunk under the cache root
	and evicts the least recently read of them to make room, whoever downloaded
	it. It reads them once, at startup, and so do we, per volume opened.
	"""
	path = settings_path()
	# A maximum of zero means unlimited, and is VC3D's own default. The floor
	# under the free space it leaves the disk applies either way.
	maximum = _ini_gib(path, "remote_cache_max_gib", 0)
	return maximum or None, _ini_gib(path, "remote_cache_min_free_gib", 20)


def _ini_gib(path, key, default):
	"""One of the settings' GiB counts, in bytes."""
	try:
		count = int(_ini_value(path, "perf", key))
	except ValueError:
		count = default
	return max(count, 0) * (1 << 30)


def projects_dir():
	"""Where VC3D puts the open data projects it downloads."""
	return os.path.join(remote_cache_root(), "open_data", "projects")


def cache_dir(url, sample_id="", root=None):
	"""The directory VC3D mirrors a remote volume into.

	Open data volumes, the ones whose entry names the sample they belong to,
	are grouped by it; anything else sits directly under the cache root.
	"""
	if root is None:
		root = remote_cache_root()
	if sample_id:
		sample = sample_directory(sample_id)
		root = os.path.normpath(root)
		# A root already pointing into this sample's directory is left alone,
		# rather than growing a second copy of the same three components.
		grouping = os.path.join("open_data", "volumes", sample)
		if os.path.join(*root.split(os.sep)[-3:]) != grouping:
			root = os.path.join(root, grouping)
	return os.path.join(root, volume_id(url))


def local_path(location, base=""):
	"""A project's local location as a path, resolved against the project's own
	directory when it is a relative one."""
	if location.startswith("file://"):
		location = location[len("file://"):]
	path = os.path.expanduser(location)
	if not os.path.isabs(path) and base:
		path = os.path.join(base, path)
	return os.path.normpath(path)


def _tag_value(tags, prefix):
	"""The rest of the first tag starting with `prefix`, or ""."""
	for tag in tags:
		if tag.startswith(prefix):
			return tag[len(prefix):]
	return ""


class VolpkgVolume:
	"""One entry of a project's "volumes" list, told where it is on disk."""

	def __init__(self, entry, base="", root=None):
		if isinstance(entry, str):
			self.location, self.tags = entry, []
		else:
			self.location = entry["location"]
			self.tags = list(entry.get("tags") or [])
		self.sample_id = _tag_value(self.tags, _SAMPLE_TAG)
		self.volume_id = _tag_value(self.tags, _VOLUME_TAG)
		self.preferred = _PREFERRED_TAG in self.tags
		self.base_scale = base_scale(self.location)
		self.remote = is_remote(self.location)
		if self.remote:
			self.url = source_url(self.location)
			self.path = cache_dir(self.url, self.sample_id, root)
		else:
			self.url = ""
			self.path = local_path(self.location, base)
		self.name = os.path.basename(self.path if not self.remote else self.url)

	@property
	def voxel_size_um(self):
		"""The width of one of the entry's voxels, when the entry says.

		Quoted at whatever level its "#vc-base-scale" selector enters the
		pyramid at, so this is `resolution_um` doubled once per level.
		"""
		try:
			size = float(_tag_value(self.tags, _VOXEL_SIZE_TAG))
		except ValueError:
			return None
		return size if size > 0.0 else None

	@property
	def resolution_um(self):
		"""The width of a voxel of the pyramid's own finest level, the one the
		scene's voxel size means, or None when the entry does not say.

		Only the open data entries say. The rest name it in the volume
		directory's own name often enough for `ui.resolution_from_path`.
		"""
		size = self.voxel_size_um
		return size if size is None else size / (1 << self.base_scale)

	@property
	def cached(self):
		"""Whether anything of the volume has been mirrored to disk yet."""
		return os.path.isdir(self.path)

	def __repr__(self):
		return "<VolpkgVolume %s>" % self.name


def volumes(path, root=None):
	"""The volumes a .volpkg.json lists, in the order it lists them."""
	with open(path, encoding="utf-8") as file:
		project = json.load(file)
	base = os.path.dirname(os.path.abspath(path))
	found = []
	for entry in project.get("volumes") or []:
		if isinstance(entry, str):
			if not entry:
				continue
		elif not isinstance(entry, dict) or not entry.get("location"):
			# The same entries VC3D's own reader skips over.
			continue
		found.append(VolpkgVolume(entry, base, root))
	return found
