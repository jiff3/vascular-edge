import numpy as np

from vascular_edge.segmentation import fallback_wmh_mask


def test_fallback_segments_connected_bright_cluster():
    flair = np.ones((9, 9, 9), dtype=np.float32)
    flair[3:5, 3:5, 3:5] = 20
    mask = fallback_wmh_mask(flair, z_threshold=2.5, min_voxels=3)
    assert mask.dtype == bool
    assert mask[3:5, 3:5, 3:5].all()
