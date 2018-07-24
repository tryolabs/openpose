"""
As implemented in https://github.com/abewley/sort but with some modifications

For each detected item, it computes the intersection over union (IOU) w.r.t.
each tracked object. (IOU matrix).Then, it applies the Hungarian algorithm
(via linear_assignment) to assign each det. item to the best possible tracked
item (i.e. to the one with max. IOU).

Note: a more recent approach uses a Deep Association Metric instead.
see https://github.com/nwojke/deep_sort
"""

import numpy as np
import scipy

from numba import jit
from sklearn.utils.linear_assignment_ import linear_assignment


# @jit
def pose_distance(detected_pose, predicted_pose):
    non_zero_indices = np.any(detected_pose > 0, axis=1) * np.any(predicted_pose > 0, axis=1)
    dists = predicted_pose[non_zero_indices, :] - detected_pose[non_zero_indices, :]
    abs_dists_per_point = np.sum(abs(dists), axis=1)
    if not abs_dists_per_point.any() or np.sum(non_zero_indices) < 3:
        return 1000  # Random large number
    else:
        # We use the median to protect ourselves from outliers caused by new limbs being detected
        # for a person, which causes the kalman filter to make abrupt predicitons with high error.
        return np.median(abs_dists_per_point)



def associate_detections_to_trackers(detected_objects, predicted_objects, score_threshold):
    """
    Assigns detected objects to tracked objects (both represented as bounding boxes)

    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    # Handle edge cases
    if len(predicted_objects) == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(len(detected_objects)),
            np.empty((0, 5), dtype=int),  # TODO what?
            None,
        )

    # Calculate iou_matrix and emb_dist_matrix
    dist_matrix = np.zeros((len(detected_objects), len(predicted_objects)), dtype=np.float32)
    for d, det_obj in enumerate(detected_objects):
        for t, pred_obj in enumerate(predicted_objects):
            dist_matrix[d, t] = pose_distance(det_obj, pred_obj['pose'])

    # Hungarian method
    # The linear assignment module tries to minimise the total assignment cost.
    matched_indices = linear_assignment(dist_matrix)
    unmatched_detections = []
    for d, _ in enumerate(detected_objects):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, _ in enumerate(predicted_objects):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    # Filter out matched with low score
    matches = []
    for m in matched_indices:
        if dist_matrix[m[0], m[1]] > score_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers), dist_matrix
