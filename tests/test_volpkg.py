import importlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
package = types.ModuleType('velend_test')
package.__path__ = [str(ROOT)]
sys.modules.setdefault('velend_test', package)
volpkg = importlib.import_module('velend_test.volpkg')


OPEN_DATA = 'https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com'


class UrlTests(unittest.TestCase):
    def test_https_passes_through(self):
        self.assertEqual(volpkg.source_url(OPEN_DATA + '/a.zarr'), OPEN_DATA + '/a.zarr')

    def test_trailing_slashes_and_selectors_go(self):
        self.assertEqual(
            volpkg.source_url(OPEN_DATA + '/a.zarr/#vc-base-scale=2'),
            OPEN_DATA + '/a.zarr')

    def test_s3_resolves_to_the_bucket_over_https(self):
        self.assertEqual(
            volpkg.source_url('s3://vesuvius-challenge-open-data/PHerc1203/a.zarr'),
            OPEN_DATA + '/PHerc1203/a.zarr')

    def test_s3_names_its_own_region(self):
        self.assertEqual(
            volpkg.source_url('s3+eu-west-1://bucket/a.zarr'),
            'https://bucket.s3.eu-west-1.amazonaws.com/a.zarr')

    def test_remote_locations_are_the_ones_with_a_scheme(self):
        self.assertTrue(volpkg.is_remote('s3://bucket/a.zarr'))
        self.assertTrue(volpkg.is_remote('https://host/a.zarr'))
        self.assertFalse(volpkg.is_remote('/volumes/a.zarr'))

    def test_base_scale_selector(self):
        self.assertEqual(volpkg.base_scale(OPEN_DATA + '/a.zarr'), 0)
        self.assertEqual(volpkg.base_scale(OPEN_DATA + '/a.zarr#vc-base-scale=2'), 2)
        self.assertEqual(volpkg.base_scale(OPEN_DATA + '/a.zarr#nonsense'), 0)


class VolumeIdTests(unittest.TestCase):
    # The ids VC3D wrote into its own settings for these two, which is what
    # says our hash of the URL is still the one it names directories with.
    def test_masked_volume(self):
        self.assertEqual(
            volpkg.volume_id(
                OPEN_DATA + '/PHerc1203/volumes/'
                '20250820131727-9.362um-1.2m-113keV-masked.zarr'),
            '20250820131727-9.362um-1.2m-113keV-masked.zarr-adf63bbdf658dd8f')

    def test_surface_prediction(self):
        self.assertEqual(
            volpkg.volume_id(
                OPEN_DATA + '/PHerc0009B/representations/predictions/surfaces/'
                '20260319104112-surface-20260413222639-surface-m7-L2-th0.2.zarr'),
            '20260319104112-surface-20260413222639-surface-m7-L2-th0.2.zarr'
            '-6fbce782f0c7a189')

    def test_sample_directories_are_path_safe(self):
        self.assertEqual(volpkg.sample_directory('PHerc1203'), 'PHerc1203')
        self.assertEqual(volpkg.sample_directory('../PHerc/1203'), 'PHerc_1203')
        self.assertEqual(volpkg.sample_directory('...'), 'sample')


class CacheDirTests(unittest.TestCase):
    URL = OPEN_DATA + '/PHerc1203/volumes/a.zarr'

    def test_open_data_volumes_are_grouped_by_sample(self):
        self.assertEqual(
            volpkg.cache_dir(self.URL, 'PHerc1203', '/cache'),
            '/cache/open_data/volumes/PHerc1203/' + volpkg.volume_id(self.URL))

    def test_a_root_already_inside_the_sample_is_left_alone(self):
        root = '/cache/open_data/volumes/PHerc1203'
        self.assertEqual(
            volpkg.cache_dir(self.URL, 'PHerc1203', root),
            root + '/' + volpkg.volume_id(self.URL))

    def test_volumes_of_no_sample_sit_under_the_root(self):
        self.assertEqual(
            volpkg.cache_dir(self.URL, '', '/cache'),
            '/cache/' + volpkg.volume_id(self.URL))


class SettingsTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / 'VC3D.ini'
        self.addCleanup(self.directory.cleanup)

    def write(self, text):
        self.path.write_text(text)

    def test_reads_the_viewer_section(self):
        self.write('[general]\nremote_cache_dir=/wrong\n'
                   '[viewer]\nother=1\nremote_cache_dir=/cache\n')
        with mock.patch.object(volpkg, 'settings_path', lambda: str(self.path)):
            self.assertEqual(volpkg.remote_cache_root(), '/cache')

    def test_unquotes_and_unescapes(self):
        self.write('[viewer]\nremote_cache_dir="/cache/a b"\n')
        with mock.patch.object(volpkg, 'settings_path', lambda: str(self.path)):
            self.assertEqual(volpkg.remote_cache_root(), '/cache/a b')

    def test_falls_back_to_the_default_when_unset(self):
        self.write('[viewer]\nother=1\n')
        with mock.patch.object(volpkg, 'settings_path', lambda: str(self.path)), \
                mock.patch('os.path.isdir', return_value=False):
            self.assertTrue(volpkg.remote_cache_root().endswith('/.VC3D/remote_cache'))


class ProjectTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'PHerc1203.volpkg.json'

    def write(self, volumes):
        self.path.write_text(json.dumps({'name': 'PHerc1203', 'volumes': volumes}))
        return volpkg.volumes(str(self.path), root='/cache')

    def test_a_remote_volume_reads_as_a_cache_directory_and_a_url(self):
        url = OPEN_DATA + '/PHerc1203/volumes/20250820131727-9.362um.zarr'
        volume, = self.write([{
            'location': url,
            'tags': [
                'vc-open-data-sample-id:PHerc1203',
                'vc-open-data-volume-id:20250820131727',
                'vc-open-data-voxel-size-um:9.362000',
                'vc-open-data-preferred-source',
            ],
        }])
        self.assertEqual(volume.url, url)
        self.assertEqual(
            volume.path,
            '/cache/open_data/volumes/PHerc1203/' + volpkg.volume_id(url))
        self.assertEqual(volume.name, '20250820131727-9.362um.zarr')
        self.assertEqual(volume.sample_id, 'PHerc1203')
        self.assertAlmostEqual(volume.resolution_um, 9.362)
        self.assertTrue(volume.preferred)

    def test_a_base_scale_entry_quotes_its_voxel_size_at_that_level(self):
        volume, = self.write([{
            'location': OPEN_DATA + '/PHerc1203/volumes/a.zarr#vc-base-scale=2',
            'tags': ['vc-open-data-voxel-size-um:9.612000'],
        }])
        self.assertEqual(volume.base_scale, 2)
        self.assertAlmostEqual(volume.voxel_size_um, 9.612)
        self.assertAlmostEqual(volume.resolution_um, 2.403)
        # The selector is the client's, not the server's.
        self.assertEqual(volume.url, OPEN_DATA + '/PHerc1203/volumes/a.zarr')

    def test_a_local_volume_keeps_its_own_path_and_no_url(self):
        volume, = self.write(['/volumes/segment.zarr'])
        self.assertEqual(volume.path, '/volumes/segment.zarr')
        self.assertEqual(volume.url, '')
        self.assertFalse(volume.remote)
        self.assertIsNone(volume.resolution_um)

    def test_relative_locations_resolve_against_the_project(self):
        volume, = self.write(['volumes/segment.zarr'])
        self.assertEqual(
            volume.path, str(Path(self.directory.name) / 'volumes/segment.zarr'))

    def test_entries_naming_nothing_are_skipped(self):
        self.assertEqual(self.write(['', {}, {'tags': []}, 3]), [])


if __name__ == '__main__':
    unittest.main()
