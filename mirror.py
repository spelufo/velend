"""Read-through, byte-preserving VC3D native Zarr mirrors."""
import asyncio
import os
from pathlib import Path
import shutil
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import weakref

from zarr.storage import LocalStore

from . import volpkg

_NETWORK = threading.BoundedSemaphore(8)
_LOCKS_GUARD = threading.Lock()
_LOCKS = weakref.WeakValueDictionary()
_METADATA = {'.zarray', '.zattrs', '.zgroup', '.zmetadata', 'zarr.json'}
_GIB = 1 << 30

LOG_MIRROR_OPS = True

def _object_lock(path):
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(str(path), threading.Lock())


def _gib(count):
    return '%.1f GiB' % (count / _GIB)


def _content_length(response):
    """What the server says an object weighs, or 0 when it does not say."""
    try:
        return max(int(response.headers.get('Content-Length')), 0)
    except (TypeError, ValueError):
        return 0


def _tree_bytes(root):
    """What the files under `root` take up.

    The total VC3D's budget keeps, near enough. It counts the chunk payloads
    only, leaving out the metadata that keeps a Zarr readable and that it never
    evicts; counting everything overstates it by a few kilobytes per array, and
    stopping early is the safe direction to be wrong in.
    """
    total = 0
    stack = [root]
    while stack:
        try:
            entries = list(os.scandir(stack.pop()))
        except OSError:
            continue  # A directory retired or unreadable mid-walk is not ours.
        for entry in entries:
            try:
                if entry.is_dir(follow_symlinks=False):
                    stack.append(entry.path)
                elif entry.is_file(follow_symlinks=False):
                    total += entry.stat(follow_symlinks=False).st_size
            except OSError:
                continue
    return total


class CacheBudget:
    """How much room a cache root's downloads have left, per VC3D's settings.

    VC3D counts every chunk beneath its cache root against one budget and
    evicts the least recently read to make room, whichever program downloaded
    it. We keep the count and refuse downloads that would break it, but evict
    nothing: our total can only lag VC3D's, and a total that lags stops us
    early rather than deleting something the other program still wanted.
    """

    # Scanning a full cache root walks a few hundred thousand files, so the
    # total is kept for the process. A download that finds it spent redoes it,
    # since eviction elsewhere is the only thing that can free room.
    RESCAN_INTERVAL = 60.0

    def __init__(self, root):
        self.root = root
        self.maximum = None
        self.minimum_free = 0
        self._lock = threading.Lock()
        self._bytes = None  # Until the first scan lands, only the floor holds.
        self._scanning = False
        self._scanned = 0.0

    def configure(self, maximum, minimum_free):
        with self._lock:
            self.maximum = maximum
            self.minimum_free = minimum_free
        if maximum is not None:
            self.scan()

    def check(self, size):
        """Refuse a download of `size` bytes the settings leave no room for."""
        with self._lock:
            total, maximum, minimum_free = (
                self._bytes, self.maximum, self.minimum_free)
        if minimum_free:
            try:
                free = shutil.disk_usage(self.root).free
            except OSError:
                free = None  # Unmeasurable is not a reason to refuse.
            if free is not None and free - size < minimum_free:
                raise OSError(
                    "VC3D's settings keep %s of the disk free and %s is left: "
                    'not downloading into %s'
                    % (_gib(minimum_free), _gib(free), self.root))
        if maximum is not None and total is not None and total + size > maximum:
            # Only VC3D evicts, so a rescan is what picks the room back up.
            self.scan()
            raise OSError(
                "VC3D's cache is at the %s maximum its settings allow "
                '(%s used): not downloading into %s'
                % (_gib(maximum), _gib(total), self.root))

    def record(self, size):
        """Count what a download just wrote."""
        with self._lock:
            if self._bytes is not None:
                self._bytes += size

    def scan(self):
        """Total the root on a thread, unless one is running or just did."""
        with self._lock:
            if self._scanning or (
                    self._scanned
                    and time.monotonic() - self._scanned < self.RESCAN_INTERVAL):
                return None
            self._scanning = True
        thread = threading.Thread(
            target=self._scan, name='velend-cache-budget', daemon=True)
        thread.start()
        return thread

    def _scan(self):
        total = _tree_bytes(self.root)
        with self._lock:
            # Whatever `record` added while this ran is in the walk or is a
            # chunk's worth of drift, which the next scan takes back out.
            self._bytes = total
            self._scanned = time.monotonic()
            self._scanning = False


# One budget per root, since the count is of the root and not of any one volume
# mirrored into it.
_BUDGETS = {}
_BUDGETS_GUARD = threading.Lock()


def budget_for(path):
    """The budget a mirror at `path` downloads within.

    VC3D applies its maximum to what lies under its own cache root and leaves
    volumes outside it unlimited, so a mirror of one's own is held to the free
    space floor alone. A full disk is nobody's idea of a cache either.
    """
    maximum, minimum_free = volpkg.cache_limits()
    # Resolved, since the store's own root is: a cache reached through a
    # symlink is still the cache VC3D counts.
    root = Path(volpkg.remote_cache_root()).resolve()
    if not path.is_relative_to(root):
        root, maximum = path, None
    with _BUDGETS_GUARD:
        budget = _BUDGETS.get(root)
        if budget is None:
            budget = _BUDGETS[root] = CacheBudget(root)
    budget.configure(maximum, minimum_free)
    return budget


class CacheLease:
    """The native shared lease; ownership follows the store and its readers."""
    def __init__(self, root):
        path = root.parent / ('.' + root.name + '.vc_cache.lock')
        root.parent.mkdir(parents=True, exist_ok=True)
        self.fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o666)
        try:
            if os.name == 'nt':
                import ctypes
                import msvcrt
                from ctypes import wintypes
                class Overlapped(ctypes.Structure):
                    _fields_ = [('Internal', ctypes.c_size_t), ('InternalHigh', ctypes.c_size_t),
                                ('Offset', wintypes.DWORD), ('OffsetHigh', wintypes.DWORD),
                                ('hEvent', wintypes.HANDLE)]
                self.overlapped = Overlapped()
                self.kernel = ctypes.WinDLL('kernel32', use_last_error=True)
                self.kernel.LockFileEx.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD,
                                                  wintypes.DWORD, wintypes.DWORD, ctypes.POINTER(Overlapped)]
                if not self.kernel.LockFileEx(msvcrt.get_osfhandle(self.fd), 1, 0, 1, 0,
                                              ctypes.byref(self.overlapped)):
                    raise ctypes.WinError(ctypes.get_last_error())
            else:
                import fcntl
                fcntl.flock(self.fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except OSError as error:
            os.close(self.fd)
            self.fd = None
            raise OSError('Cannot acquire shared VC3D cache lease (cache may be exclusively locked): %s' % path) from error

    def __del__(self):
        if getattr(self, 'fd', None) is not None:
            os.close(self.fd)
            self.fd = None


class MirrorStore(LocalStore):
    def __init__(self, root, source_url, *, online=lambda: True):
        raw = os.fspath(root)
        url = urllib.parse.urlsplit(source_url)
        if url.scheme not in ('http', 'https') or not url.netloc or url.username or url.password or url.query or url.fragment:
            raise ValueError('Source URL must be a public HTTP(S) Zarr root without credentials, query or fragment')
        if '://' in raw or raw.startswith(('//', '\\\\')):
            raise ValueError('Source URL requires a local cache directory')
        root = Path(raw).absolute()
        if not root.name or root.is_symlink():
            raise ValueError('Cache must be a directory, not a root or symlink')
        root = root.resolve()
        self.lease = CacheLease(root)
        if (root / ".vc_delta3d_cache").exists():
            raise ValueError("Delta3D caches are unsupported; select a native Zarr mirror")
        if root.is_dir() and not any((root / name).exists() for name in ('.zgroup', 'zarr.json', '.zmetadata')):
            if any(p.is_dir() and p.name.startswith('level_') for p in root.iterdir()):
                raise ValueError('Legacy internal caches are unsupported; select a native Zarr mirror')
        root.mkdir(parents=True, exist_ok=True)
        super().__init__(root, read_only=True)
        self.source_url = source_url.rstrip('/')
        self.online = online
        self.budget = budget_for(root)

    def with_read_only(self, read_only=False):
        if not read_only:
            raise ValueError('Mirror stores are read-only')
        return self

    def _path(self, key):
        if not isinstance(key, str) or not key or any(p in ('', '.', '..') for p in key.split('/')) or '\\' in key or ':' in key or '\0' in key:
            raise ValueError('Unsafe Zarr key: %r' % key)
        path = self.root / key
        if not path.resolve().is_relative_to(self.root) or path.is_symlink():
            raise ValueError('Zarr key escapes cache: %r' % key)
        return path

    @staticmethod
    def _touch(path):
        """Bump the time VC3D's cache evicts by, the way a read of its own
        would: what we are reading is not what it should throw away first."""
        try:
            os.utime(path)
        except OSError:
            pass

    def _publish(self, path, response=None):
        """Write the response to `path`, and count it. Returns its size."""
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + '.tmp.' + uuid.uuid4().hex)
        try:
            with temporary.open('xb') as output:
                count = 0
                deadline = time.monotonic() + 300
                if response is not None:
                    while block := response.read1(1024 * 1024):
                        if time.monotonic() > deadline:
                            raise TimeoutError("HTTP object download exceeded five minutes")
                        output.write(block)
                        count += len(block)
                    expected = response.headers.get('Content-Length')
                    if expected is not None and count != int(expected):
                        raise OSError('Incomplete HTTP transfer')
            if LOG_MIRROR_OPS:
                print(f"velend: Saving {path}")
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        self.budget.record(count)
        return count

    def get_sync(self, key, *, prototype=None, byte_range=None):
        path = self._path(key)
        marker = self._path(key + '.empty')
        with _object_lock(path):
            if marker.exists():
                return None
            if not path.is_file():
                if not self.online():
                    raise OSError('Network access is disabled in Blender; object is not cached: ' + key)
                with _NETWORK:
                    if not self.online():
                        raise OSError("Network access is disabled in Blender")
                    url = self.source_url + '/' + urllib.parse.quote(key, safe='/')
                    request = urllib.request.Request(url, headers={'Accept-Encoding': 'identity'})
                    try:
                        if LOG_MIRROR_OPS:
                            print("velend: Requesting", url)
                        with urllib.request.urlopen(request, timeout=30) as response:
                            if response.status != 200:
                                print("velend: Downloading ", key, "failed, response status", response.status)
                                raise OSError('Expected complete HTTP object, got %s' % response.status)
                            if path.name not in _METADATA:
                                # What keeps a Zarr readable is exempt: VC3D's
                                # budget will not evict it either, and a few
                                # kilobytes refused would cost us the volume.
                                self.budget.check(_content_length(response))
                            self._publish(path, response)
                    except urllib.error.HTTPError as error:
                        if error.code != 404:
                            raise
                        if path.name not in _METADATA:
                            self._publish(marker)
                            path.unlink(missing_ok=True)
                        return None
                marker.unlink(missing_ok=True)
            else:
                self._touch(path)
            return super().get_sync(key, prototype=prototype, byte_range=byte_range)

    async def get(self, key, prototype=None, byte_range=None):
        return await asyncio.to_thread(self.get_sync, key, prototype=prototype, byte_range=byte_range)

    async def get_partial_values(self, prototype, key_ranges):
        return await asyncio.gather(*(self.get(key, prototype, byte_range) for key, byte_range in key_ranges))

    async def exists(self, key):
        return await self.get(key) is not None

    def exists_sync(self, key):
        return self.get_sync(key) is not None

    async def getsize(self, key):
        value = await self.get(key)
        if value is None:
            raise FileNotFoundError(key)
        return len(value)
