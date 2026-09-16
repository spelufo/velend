import copy
import importlib
import sys
import types
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
package = types.ModuleType('velend_test')
package.__path__ = [str(ROOT)]
sys.modules.setdefault('velend_test', package)
# `metadata` reaches for the add-on preferences and the extension's cache
# directory, neither of which exists outside Blender. Nothing under test here
# touches either, so a stub is enough to get the module imported.
sys.modules.setdefault('bpy', types.ModuleType('bpy'))
metadata = importlib.import_module('velend_test.metadata')


OPEN_DATA = 'https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com'

# One sample, two volumes, and the matrix taking the first into the second.
MANIFEST = {
    'samples': {
        'PHercTest': {
            'sample': {
                'id': 'PHercTest',
                'properties': {
                    'type': 'scroll',
                    'volume_transforms': [
                        {
                            'from_volume_id': '20231027191953',
                            'transforms': [
                                {
                                    'to_volume_id': '20231117143551',
                                    'matrix': [
                                        [0.5, 0.0, 0.0, 10.0],
                                        [0.0, 0.5, 0.0, 20.0],
                                        [0.0, 0.0, 0.5, 30.0],
                                    ],
                                },
                            ],
                        },
                    ],
                },
            },
            'scans': {},
            'volumes': {
                '20231027191953': {'id': '20231027191953', 'long_id': '20231027191953-3.240um'},
                '20231117143551': {'id': '20231117143551', 'long_id': '20231117143551-7.910um'},
                '20241024131838': {'id': '20241024131838', 'long_id': '20241024131838-8.000um'},
            },
            'segments': {},
        },
    },
}


class VolumeIdTests(unittest.TestCase):
    def test_the_bucket_directory_names_the_volume(self):
        self.assertEqual(
            metadata.volume_id_for(OPEN_DATA + '/samples/P/volumes/20231027191953-3.240um.zarr/'),
            '20231027191953')

    def test_vc3d_cache_directory_names_it_too(self):
        self.assertEqual(
            metadata.volume_id_for('/c/20231027191953-3.240um.zarr-adf63bbdf658dd8f'),
            '20231027191953')

    def test_the_first_location_that_names_one_wins(self):
        self.assertEqual(
            metadata.volume_id_for('', '/c/20231117143551-7.910um.zarr'), '20231117143551')

    def test_a_directory_naming_none_gives_none(self):
        self.assertEqual(metadata.volume_id_for('/scans/my-own-volume.zarr'), '')
        self.assertEqual(metadata.volume_id_for('', None), '')

    def test_a_timestamp_has_to_start_the_name(self):
        self.assertEqual(metadata.volume_id_for('/c/scroll-20231027191953.zarr'), '')


class VolumeTransformTests(unittest.TestCase):
    def setUp(self):
        # `parse` keeps the manifest's own dictionaries, and two of the tests
        # below rewrite a matrix in place.
        parsed = metadata.parse(copy.deepcopy(MANIFEST))
        self.previous = (
            metadata.SAMPLES, metadata.SCANS, metadata.VOLUMES, metadata.SEGMENTS)
        metadata.SAMPLES, metadata.SCANS, metadata.VOLUMES, metadata.SEGMENTS = parsed

    def tearDown(self):
        metadata.SAMPLES, metadata.SCANS, metadata.VOLUMES, metadata.SEGMENTS = self.previous

    def test_scene_volume_source_uses_the_scene_frame_after_a_render_switch(self):
        source = metadata.source_volume_for(
            '20231027191953', '', '/cache/20231117143551-7.910um.zarr'
        )
        self.assertEqual(source, 'PHercTest/volumes/20231027191953-3.240um.zarr')

    def test_uncatalogued_scene_volume_falls_back_to_its_saved_id(self):
        source = metadata.source_volume_for(
            '19000101000000', '', '/cache/20231117143551-7.910um.zarr'
        )
        self.assertEqual(source, '19000101000000')

    def test_the_registered_direction_comes_out_as_stated(self):
        matrix = metadata.volume_transform('20231027191953', '20231117143551')
        np.testing.assert_allclose(matrix, [
            [0.5, 0.0, 0.0, 10.0],
            [0.0, 0.5, 0.0, 20.0],
            [0.0, 0.0, 0.5, 30.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        np.testing.assert_allclose(
            matrix[:3, :3] @ [2.0, 4.0, 6.0] + matrix[:3, 3], [11.0, 22.0, 33.0])

    def test_the_direction_the_manifest_leaves_out_is_the_inverse(self):
        forward = metadata.volume_transform('20231027191953', '20231117143551')
        back = metadata.volume_transform('20231117143551', '20231027191953')
        np.testing.assert_allclose(back @ forward, np.eye(4), atol=1e-12)

    def test_a_volume_is_its_own_frame(self):
        np.testing.assert_allclose(
            metadata.volume_transform('20231027191953', '20231027191953'), np.eye(4))

    def test_unregistered_volumes_are_unrelated(self):
        # Of the same sample, but with no transform stated either way round.
        self.assertIsNone(metadata.volume_transform('20231027191953', '20241024131838'))

    def test_a_volume_outside_the_catalogue_is_unrelated(self):
        self.assertIsNone(metadata.volume_transform('20231027191953', '19000101000000'))

    def test_nothing_relates_a_volume_that_is_not_named(self):
        self.assertIsNone(metadata.volume_transform('', '20231027191953'))
        self.assertIsNone(metadata.volume_transform('20231027191953', ''))
        self.assertIsNone(metadata.volume_transform('', ''))

    def test_a_fourth_row_that_is_not_affine_is_no_matrix(self):
        transforms = metadata.SAMPLES['PHercTest'].volume_transforms['20231027191953']
        transforms[0]['matrix'] = [
            [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0], [0.1, 0.0, 0.0, 1.0]]
        self.assertIsNone(metadata.volume_transform('20231027191953', '20231117143551'))

    def test_a_fourth_row_that_is_affine_is_read_as_the_three_that_matter(self):
        transforms = metadata.SAMPLES['PHercTest'].volume_transforms['20231027191953']
        transforms[0]['matrix'] = [
            [2.0, 0.0, 0.0, 1.0], [0.0, 2.0, 0.0, 2.0],
            [0.0, 0.0, 2.0, 3.0], [0.0, 0.0, 0.0, 1.0]]
        np.testing.assert_allclose(
            metadata.volume_transform('20231027191953', '20231117143551'),
            transforms[0]['matrix'])

    def test_a_matrix_of_the_wrong_shape_is_no_matrix(self):
        transforms = metadata.SAMPLES['PHercTest'].volume_transforms['20231027191953']
        transforms[0]['matrix'] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        self.assertIsNone(metadata.volume_transform('20231027191953', '20231117143551'))

    def test_a_registration_that_cannot_be_run_backwards_gives_none(self):
        transforms = metadata.SAMPLES['PHercTest'].volume_transforms['20231027191953']
        transforms[0]['matrix'] = [
            [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
        self.assertIsNone(metadata.volume_transform('20231117143551', '20231027191953'))


if __name__ == '__main__':
    unittest.main()
