import numpy as np

from kalman import KalmanPoseTracker
from association import associate_detections_to_trackers


class PoseTracker:

    def __init__(self, tracked_points_num, max_age=30, min_hits=15, score_threshold=0.3):
        self.tracked_points_num = tracked_points_num 
        self.max_age = max_age
        self.min_hits = min_hits
        self.score_threshold = score_threshold
        self.frame_count = 0
        self.trackers = []

    def update(self, detected_objects):
        """
        Updates and returns the internal tracker state.

        Args:
            detected_objects: A list of newly detected objects.

        Returns:
            A list of targets.

        """
        self.frame_count += 1

        # Generate predicted objects from trackers
        predicted_objects = []
        to_del = []
        for idx, tracker in enumerate(self.trackers):
            predicted_objects.append({'pose': tracker.predict()})

            # Store the tracker indices we'll delete in the following section
            if np.any(np.isnan(predicted_objects[idx]['pose'])):
                to_del.append(idx)

        # TODO: Merge with previous if-statement
        # Delete the self.trackers that generated no predictions
        for t in reversed(to_del):
            self.trackers.pop(t)

        # Update/create trackers
        if detected_objects.any():
            # Associate detected objects to predted objects
            # matched: tuples of indices, unmatched_dets & unmatched_trks: indices
            matched, unmatched_dets, unmatched_trks, dist_matrix = associate_detections_to_trackers(
                    detected_objects, predicted_objects, score_threshold=self.score_threshold
                )

            # Update matched trackers with assigned detected objects.
            for t, tracker in enumerate(self.trackers):
                if t not in unmatched_trks:
                    matched_det_idx = matched[np.where(matched[:, 1] == t)[0], 0]
                    trk_matched_bbox = detected_objects[matched_det_idx]
                    # trk_matched_bbox, trk_matched_embedding = [
                    #     (detected_objects[idx]['pose'], detected_objects[idx]['feat'])
                    #     for idx in matched_det_idx
                    # ][0]
                    trk_debug_dict = {
                        'dist': dist_matrix[matched_det_idx, t][0],
                    }
                    tracker.update(trk_matched_bbox, trk_debug_dict)

            # Create and initialize new trackers for unmatched detected objects.
            for i in unmatched_dets:
                self.trackers.append(
                    KalmanPoseTracker(detected_objects[i], self.tracked_points_num)
                )
        else:
            for tracker in reversed(self.trackers):
                tracker.update(None, None)

        # Return matched and unmatched trackers
        # We return both so we can draw them for debugging and such I think
        matched_predictions = []
        unmatched_predictions = []
        i = len(self.trackers)
        for tracker in reversed(self.trackers):
            # TODO fix this mess, use `.update(debug_dict)`
            object_ = {
                'pose': tracker.get_state(),
                'id': tracker.id,
                'age': tracker.age,
                'hit_streak': tracker.hit_streak,
                'time_since_update': tracker.time_since_update,
                'score': tracker.debug_dict.get('score'),
            }

            if (tracker.time_since_update < 1 and
                    (tracker.hit_streak >= self.min_hits or self.frame_count <= self.min_hits)):
                matched_predictions.append(object_)
            else:
                unmatched_predictions.append(object_)

            i -= 1

            # Remove dead tracklet.
            if tracker.time_since_update > self.max_age:
                self.trackers.pop(i)

        return matched_predictions, unmatched_predictions
