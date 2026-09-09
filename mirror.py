"""Read-through, byte-preserving VC3D native Zarr mirrors."""
import asyncio
import os
from pathlib import Path
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import weakref

from zarr.storage import LocalStore

_NETWORK = threading.BoundedSemaphore(8)
_LOCKS_GUARD = threading.Lock()
_LOCKS = weakref.WeakValueDictionary()
_METADATA = {'.zarray', '.zattrs', '.zgroup', '.zmetadata', 'zarr.json'}


def _object_lock(path):
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(str(path), threading.Lock())


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
    def _publish(path, response=None):
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
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

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
                    request = urllib.request.Request(self.source_url + '/' + urllib.parse.quote(key, safe='/'),
                                                     headers={'Accept-Encoding': 'identity'})
                    try:
                        with urllib.request.urlopen(request, timeout=30) as response:
                            if response.status != 200:
                                raise OSError('Expected complete HTTP object, got %s' % response.status)
                            self._publish(path, response)
                    except urllib.error.HTTPError as error:
                        if error.code != 404:
                            raise
                        if path.name not in _METADATA:
                            self._publish(marker)
                            path.unlink(missing_ok=True)
                        return None
                marker.unlink(missing_ok=True)
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
