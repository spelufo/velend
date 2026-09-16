import asyncio
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import gc
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest import mock

import numpy as np
import zarr
from zarr.abc.store import RangeByteRequest, OffsetByteRequest, SuffixByteRequest
from zarr.core.buffer import default_buffer_prototype

ROOT = Path(__file__).resolve().parents[1]
package = types.ModuleType('velend_test')
package.__path__ = [str(ROOT)]
sys.modules.setdefault('velend_test', package)
mirror = importlib.import_module('velend_test.mirror')
MirrorStore = mirror.MirrorStore
state = importlib.import_module('velend_test.state')


class MirrorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.objects = {}
        self.requests = Counter()
        owner = self
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                owner.requests[self.path] += 1
                if self.path == '/error':
                    self.send_error(503)
                    return
                if self.path == '/short':
                    self.send_response(200)
                    self.send_header('Content-Length', '100')
                    self.end_headers()
                    self.wfile.write(b'short')
                    self.close_connection = True
                    return
                data = owner.objects.get(self.path)
                if data is None:
                    self.send_error(404)
                else:
                    self.send_response(200)
                    self.send_header('Content-Length', str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
            def log_message(self, *args):
                pass
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = 'http://127.0.0.1:%d' % self.server.server_port
        self.cache = self.root / 'cache'
        self.store = MirrorStore(self.cache, self.url)

    def tearDown(self):
        state.close_volume()
        self.store = None
        gc.collect()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.temp.cleanup()

    def test_entrypoints_and_exact_bytes(self):
        data = bytes(range(256)) * 10
        self.objects['/0/1.2.3'] = data
        with ThreadPoolExecutor(12) as pool:
            results = list(pool.map(lambda _: self.store.get_sync('0/1.2.3').to_bytes(), range(24)))
        self.assertTrue(all(result == data for result in results))
        self.assertEqual(self.requests['/0/1.2.3'], 1)
        self.assertEqual((self.cache / '0/1.2.3').read_bytes(), data)
        proto = default_buffer_prototype()
        ranges = [RangeByteRequest(2, 9), OffsetByteRequest(2558), SuffixByteRequest(3)]
        expected = [data[2:9], data[2558:], data[-3:]]
        async def reads():
            self.assertTrue(await self.store.exists('0/1.2.3'))
            self.assertEqual(await self.store.getsize('0/1.2.3'), len(data))
            values = await self.store.get_partial_values(proto, [('0/1.2.3', r) for r in ranges])
            self.assertEqual([v.to_bytes() for v in values], expected)
            batches = [b async for b in self.store.get_ranges('0/1.2.3', ranges, prototype=proto)]
            self.assertEqual({i: v.to_bytes() for b in batches for i, v in b}, dict(enumerate(expected)))
            values = [v async for v in self.store._get_many([('0/1.2.3', proto, None)])]
            self.assertEqual(values[0][1].to_bytes(), data)
        asyncio.run(reads())
        self.assertEqual(dict((i, v.to_bytes()) for i, v in self.store.get_ranges_sync('0/1.2.3', ranges, prototype=proto)), dict(enumerate(expected)))
        self.assertEqual(self.requests['/0/1.2.3'], 1)

    def test_markers_errors_offline_and_metadata(self):
        self.assertIsNone(self.store.get_sync('0/0/0/0'))
        self.assertTrue((self.cache / '0/0/0/0.empty').exists())
        (self.cache / '0/0/0/0').write_bytes(b'ignored')
        self.assertIsNone(self.store.get_sync('0/0/0/0'))
        self.assertEqual(self.requests['/0/0/0/0'], 1)
        for key in ('.zattrs', '0/zarr.json'):
            self.assertIsNone(self.store.get_sync(key))
            self.assertFalse((self.cache / (key + '.empty')).exists())
        for key in ('error', 'short'):
            with self.assertRaises(Exception):
                self.store.get_sync(key)
            self.assertFalse((self.cache / key).exists())
            self.assertFalse((self.cache / (key + '.empty')).exists())
        self.assertFalse(list(self.cache.rglob('*.tmp.*')))
        (self.cache / 'cached').write_bytes(b'local')
        self.store.online = lambda: False
        self.assertEqual(self.store.get_sync('cached').to_bytes(), b'local')
        with self.assertRaisesRegex(OSError, 'disabled'):
            self.store.get_sync('uncached')

    def test_empty_initialization_fill_and_shards(self):
        for version in (2, 3):
            source = self.root / ('source%d' % version)
            group = zarr.open_group(source, mode='w', zarr_format=version)
            group.attrs['multiscales'] = [{'datasets': [{'path': '0'}]}]
            kwargs = {'shards': (4, 4, 4)} if version == 3 else {}
            array = group.create_array('0', shape=(8, 8, 8), chunks=(2, 2, 2), dtype='u1', fill_value=17, **kwargs)
            array[:2, :2, :2] = 42
            for path in source.rglob('*'):
                if path.is_file():
                    self.objects['/' + path.relative_to(source).as_posix()] = path.read_bytes()
            volume = state.open_volume(str(self.root / ('mirror%d' % version)), self.url)
            np.testing.assert_array_equal(volume[0][:], array[:])
            np.testing.assert_array_equal(volume[0][1:3, 1:3, 1:3], array[1:3, 1:3, 1:3])
            self.objects.clear()
        local = state.open_volume(str(source))
        np.testing.assert_array_equal(local[0][:], array[:])

    @unittest.skipIf(sys.platform == 'win32', 'Unix flock interoperability')
    def test_lease_lifetime_and_contention(self):
        lock = self.root / '.cache.vc_cache.lock'
        code = 'import os,fcntl,sys; f=os.open(sys.argv[1],os.O_RDWR); fcntl.flock(f,int(sys.argv[2])|fcntl.LOCK_NB)'
        import fcntl
        def attempt(mode):
            return subprocess.run([sys.executable, '-c', code, str(lock), str(mode)], capture_output=True).returncode
        self.assertEqual(attempt(fcntl.LOCK_SH), 0)
        self.assertNotEqual(attempt(fcntl.LOCK_EX), 0)
        self.store = None
        gc.collect()
        self.assertEqual(attempt(fcntl.LOCK_EX), 0)
        with lock.open('r+') as held:
            fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(OSError, 'lease'):
                MirrorStore(self.cache, self.url)
        self.assertTrue(lock.exists())

    @unittest.skipIf(sys.platform == 'win32', 'Unix flock interoperability')
    def test_outstanding_write_keeps_lease_and_publishes_atomically(self):
        import fcntl
        self.objects['/chunk'] = b'exact bytes'
        entered, release = threading.Event(), threading.Event()
        mirror = importlib.import_module('velend_test.mirror')
        replace = mirror.os.replace
        def publish(source, destination):
            self.assertIn('.tmp.', source.name)
            self.assertFalse(destination.exists())
            self.assertEqual(source.read_bytes(), b'exact bytes')
            entered.set()
            self.assertTrue(release.wait(5))
            replace(source, destination)
        with mock.patch.object(mirror.os, 'replace', publish), ThreadPoolExecutor(1) as pool:
            future = pool.submit(self.store.get_sync, 'chunk')
            try:
                self.assertTrue(entered.wait(5))
                self.store = None
                state.close_volume()
                gc.collect()
                code = 'import os,fcntl,sys; f=os.open(sys.argv[1],os.O_RDWR); fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)'
                result = subprocess.run([sys.executable, '-c', code, str(self.root / '.cache.vc_cache.lock')], capture_output=True)
                self.assertNotEqual(result.returncode, 0)
            finally:
                release.set()
            self.assertEqual(future.result(timeout=5).to_bytes(), b'exact bytes')
        self.assertFalse(list(self.cache.rglob('*.tmp.*')))

    def test_state_keys_include_url_and_old_arrays_keep_lease(self):
        source = self.root / 'local'
        group = zarr.open_group(source, mode='w', zarr_format=2)
        group.attrs['multiscales'] = [{'datasets': [{'path': '0'}]}]
        group.create_array('0', shape=(2, 2, 2), chunks=(2, 2, 2), dtype='u1')
        first = state.get_volume(str(source))
        self.assertIs(state.get_volume(str(source)), first)
        second = state.get_volume(str(source), self.url)
        self.assertIsNot(second, first)
        self.assertIs(state.get_volume(str(source), self.url), second)
        state.close_volume()
        self.assertEqual(second[0].shape, (2, 2, 2))

    def test_unsafe_and_incompatible(self):
        for key in ('../escape', '/absolute', 'a//b', 'a/../b', 'C:/foo', 'a\\b'):
            with self.assertRaises(ValueError):
                self.store.get_sync(key)
        for path in ('https://host/cache', '//host/cache'):
            with self.assertRaises(ValueError):
                MirrorStore(path, self.url)
        legacy = self.root / 'legacy'
        (legacy / 'level_0').mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, 'Legacy'):
            MirrorStore(legacy, self.url)
        self.assertFalse((legacy / '.zgroup').exists())
        (self.cache / '.vc_delta3d_cache').write_text('D3D1\n')
        with self.assertRaisesRegex(ValueError, 'Delta3D'):
            MirrorStore(self.cache, self.url)
        self.assertEqual((self.cache / '.vc_delta3d_cache').read_text(), 'D3D1\n')

    def budget_store(self, maximum, minimum_free=0):
        """A store held to the given limits, its cache inside VC3D's root."""
        for target, value in (('cache_limits', lambda: (maximum, minimum_free)),
                              ('remote_cache_root', lambda: str(self.root))):
            patch = mock.patch.object(mirror.volpkg, target, value)
            patch.start()
            self.addCleanup(patch.stop)
        return MirrorStore(self.cache, self.url)

    def scanned(self, budget):
        """The budget's total, once its background scan has landed."""
        deadline = time.monotonic() + 5
        while budget._bytes is None:
            self.assertLess(time.monotonic(), deadline, 'scan did not finish')
            time.sleep(0.01)
        return budget._bytes

    def test_budget_refuses_downloads_that_would_fill_the_disk(self):
        store = self.budget_store(None, minimum_free=1 << 62)
        self.objects['/0/0'] = b'x' * 100
        with self.assertRaisesRegex(OSError, 'free'):
            store.get_sync('0/0')
        self.assertFalse((self.cache / '0/0').exists())

    def test_budget_refuses_downloads_past_the_maximum(self):
        self.cache.mkdir(parents=True, exist_ok=True)
        (self.cache / 'existing').write_bytes(b'x' * 400)
        store = self.budget_store(500)
        self.assertEqual(self.scanned(store.budget), 400)

        self.objects['/0/0'] = b'x' * 50
        self.assertEqual(store.get_sync('0/0').to_bytes(), b'x' * 50)
        self.assertEqual(store.budget._bytes, 450)
        self.assertEqual(mirror.vc3d_cache_usage(), (450, 500))

        self.objects['/0/1'] = b'x' * 100
        with self.assertRaisesRegex(mirror.CacheLimitError, 'maximum'):
            store.get_sync('0/1')
        self.assertFalse((self.cache / '0/1').exists())
        # What is already cached stays readable when the cache is full.
        self.assertEqual(store.get_sync('0/0').to_bytes(), b'x' * 50)

    def test_budget_leaves_room_for_the_metadata_a_zarr_needs(self):
        store = self.budget_store(0)
        self.assertEqual(self.scanned(store.budget), 0)
        for key in ('.zattrs', '0/zarr.json'):
            self.objects['/' + key] = b'{}'
            self.assertEqual(store.get_sync(key).to_bytes(), b'{}')
        self.objects['/0/0'] = b'x'
        with self.assertRaisesRegex(OSError, 'maximum'):
            store.get_sync('0/0')

    def test_budget_is_only_the_free_space_floor_outside_vc3ds_cache(self):
        patch = mock.patch.object(
            mirror.volpkg, 'remote_cache_root', lambda: str(self.root / 'vc3d'))
        patch.start()
        self.addCleanup(patch.stop)
        patch = mock.patch.object(
            mirror.volpkg, 'cache_limits', lambda: (500, 0))
        patch.start()
        self.addCleanup(patch.stop)
        store = MirrorStore(self.cache, self.url)
        # A mirror of one's own is not part of what VC3D counts, so it keeps
        # the unlimited behaviour VC3D gives such volumes.
        self.assertIsNone(store.budget.maximum)
        self.assertEqual(store.budget.root, self.cache.resolve())

    def test_a_cache_hit_bumps_the_time_vc3d_evicts_by(self):
        self.objects['/0/0'] = b'x' * 10
        self.store.get_sync('0/0')
        path = self.cache / '0/0'
        stale = time.time() - 86400
        os.utime(path, (stale, stale))
        self.assertEqual(self.store.get_sync('0/0').to_bytes(), b'x' * 10)
        self.assertGreater(path.stat().st_mtime, stale + 1)
        self.assertEqual(self.requests['/0/0'], 1)


if __name__ == '__main__':
    unittest.main()
