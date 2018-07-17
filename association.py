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

MATCH_LIST = []
NORM_EMB_DIST = 0.03

@jit
def pose_distance(detected_pose, predicted_pose):
    # TODO insert code here
    return 1

@jit
def iou(bb_test, bb_gt):
    """
    Computes IUO between two bboxes in the form [x1,y1,x2,y2]
    """
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1]) + (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1]) - wh)  # noqa
    return(o)


@jit
def embedding_distance(detected_embedding, predicted_embedding, top_k_matching=100):
    partition_edge = top_k_matching * -1

    # First reduce each 7x7 embedding to a single scalar
    detected_embedding_reduced = np.amax(detected_embedding, (0, 1))
    predicted_embedding_reduced = np.amax(predicted_embedding, (0, 1))

    # We use only the best matches to calculate distance
    matching_scores = detected_embedding_reduced * predicted_embedding_reduced
    best_k_matches = np.argpartition(matching_scores, partition_edge)[partition_edge:]

    # Now reduce the similarities among all feat maps to a single score
    return scipy.spatial.distance.cosine(
        detected_embedding_reduced[best_k_matches], predicted_embedding_reduced[best_k_matches]
    )


@jit
def normalize_embedding(embedding_dist, norm):
        return max(
            1 - (embedding_dist - NORM_EMB_DIST) / NORM_EMB_DIST, 0
        )


def associate_detections_to_trackers(detected_objects, predicted_objects, score_threshold=0.3):
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
    # In our case we pass -score_matrix as we want to maximise the total score between
    # track predictions and the frame detection.
    matched_indices = linear_assignment(-dist_matrix)

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
        if dist_matrix[m[0], m[1]] < score_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers), dist_matrix
