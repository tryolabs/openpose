# From Python
import sys
import cv2
import os
import time
import click
from trackers import PoseTracker
sys.path.append('build/python/openpose')
from openpose import *


# Globals
DEBUG = True if len(sys.argv) > 3 and sys.argv[3] else False
BODY_PART_KEYS = ["nose", "neck", "rshoulder", "relbow", "rwrist", "lshoulder", "lelbow", "lwrist",
                  "midhip", "rhip", "rknee", "rankle", "lhip", "lknee", "lankle", "reye", "leye",
                  "rear", "lear", "lbigtoe", "lsmalltoe", "lheel", "rbigtoe", "rsmalltoe", "rheel"]
# Using only these 4 because dissapearing limbs don't play well with kalman filters
BODY_PARTS_TO_TRACK_KEYS = ['nose', 'reye', 'leye', 'midhip']
ATTENTION_TIMERS = {}
INTERFRAME_DISTANCE_THRESHOLD = 250
MIN_TRIANGLE_AREA = 200

def main():
    # Set up network
    dir_path = os.path.dirname(os.path.realpath(__file__))
    params = dict()
    params["logging_level"] = 3
    params["output_resolution"] = "-1x-1"
    params["net_resolution"] = "-1x368"
    # params["output_resolution"] = "320x176"
    # params["net_resolution"] = "320x176"
    params["model_pose"] = "BODY_25"
    params["alpha_pose"] = 0.6
    params["scale_gap"] = 0.3
    params["scale_number"] = 1
    params["render_threshold"] = 0.05
    # If GPU version is built, and multiple GPUs are available, set the ID here
    params["num_gpu_start"] = 0
    params["disable_blending"] = False
    params["default_model_folder"] = dir_path + "/models/"
    openpose = OpenPose(params)

    # Read Input Video
    cap = cv2.VideoCapture(sys.argv[1])
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_counter = 0
    video_progress_bar = click.progressbar(length=total_frames)

    # Setup Output Video
    output_video_path = sys.argv[2] if len(sys.argv) > 2 else None
    if output_video_path:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(sys.argv[2], fourcc, fps, (frame_width, frame_height))  # Width, Height

    # Setup tracker
    tracker = PoseTracker(tracked_points_num=len(BODY_PARTS_TO_TRACK_KEYS), score_threshold = INTERFRAME_DISTANCE_THRESHOLD)
    posterior_time = 0

    # Generate video
    with video_progress_bar as progress_bar:
        start = time.time()
        for _ in progress_bar:
            frame_counter += 1
            # Read frame
            start_frame = time.time()
            ret, frame = cap.read()
            if frame is None:
                break

            # Output keypoints and the image with the human skeleton blended on it
            pre_pred = time.time()
            detected_people, output_image = openpose.forward(frame, True)
            if DEBUG: frame = output_image
            post_pred = time.time()

            # Filter out body parts we do not wish to track
            body_parts_to_track_indexes = [BODY_PART_KEYS.index(key) for key in BODY_PARTS_TO_TRACK_KEYS]
            detected_people = detected_people[:, body_parts_to_track_indexes]

            # Filter out probs from body parts data points
            detected_people = detected_people[:, :, :2]

            # Filter out people for which we haven't detected the body parts we want to track
            detected_people = detected_people[np.any(detected_people > 0, axis=(1, 2)), :, :]

            # Get predictions (it returns unmatched_people only for debugging)
            predicted_people, unmatched_people = tracker.update(detected_people)

            prior_time, posterior_time = posterior_time, time.time()
            delta_t = posterior_time - prior_time

            # Draw each person
            # Predicted
            for predicted_person in predicted_people:
                draw_body_parts(frame, predicted_person, [('nose', 'reye', 'leye')], (0, 255, 0), 3, delta_t, predicted_person.get('debug'))

            if DEBUG:
                # Unmatched
                for unmatched_person in unmatched_people:
                    draw_body_parts(frame, unmatched_person, [('nose', 'reye', 'leye')], (255, 0, 0), 1, delta_t, unmatched_person.get('debug'))

                # Detected
                for detected_person_pose in detected_people:
                    draw_body_parts(frame, {'last_detection': detected_person_pose}, [('nose', 'reye', 'leye')], (255, 255, 255), 1)
        
            # cv2.imshow("output", frame)
            if output_video_path:
                out.write(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        end = time.time()

    # Cleanup
    if output_video_path:
        out.release()
    cap.release()
    cv2.destroyAllWindows()
    print("fps {:2f}".format(frame_counter / (end - start)))


def draw_body_parts(frame, person, triangles, color, width, delta_t=None, debug_=None):
    body_parts = {key: value for key, value in zip(BODY_PARTS_TO_TRACK_KEYS, person['last_detection'].astype(int))}

    thresh_fraction = 1/4
    for mid, right, left in triangles:
        mid = tuple(body_parts[mid])
        right = tuple(body_parts[right])
        left = tuple(body_parts[left])

        # Draw triangle
        if all(mid) and all(right) and all(left):
            eye_x_dist = abs(right[0] - left[0])
            # Check that nose is in mid fraction of face, so they are looking straight ahead
            right_thresh = right[0] + eye_x_dist * thresh_fraction
            left_thresh = left[0] - eye_x_dist * thresh_fraction
            triangle_area = (
                (max(left[0], mid[0], right[0]) - min(left[0], mid[0], right[0])) *
                (max(left[1], mid[1], right[1]) - min(left[1], mid[1], right[1]))
            ) / 2
            if right_thresh < mid[0] < left_thresh and person.get('id') and triangle_area > MIN_TRIANGLE_AREA:
                if person['id'] in ATTENTION_TIMERS.keys():
                    ATTENTION_TIMERS[person['id']] += delta_t
                else:
                    ATTENTION_TIMERS[person['id']] = 0
                text = "{:.2f}".format(ATTENTION_TIMERS[person['id']])
                cv2.putText(frame, text, (mid[0] - 20, mid[1] + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

            # Draw triangle
            cv2.line(frame, mid, right, color, width)
            cv2.line(frame, mid, left, color, width)
            cv2.line(frame, right, left, color, width)
            cv2.circle(frame, mid, 5, color, width)

            # Draw id
            if DEBUG:
                id = person.get('id')
                if id is not None:
                    cv2.putText(frame, str(id), mid, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2, cv2.LINE_AA)

                # Draw person's debug dict info
                if debug_ is not None:
                    dist = debug_.get('dist') if debug_.get('dist') else 0
                    cv2.putText(frame, str(int(dist)), (mid[0] - 10, mid[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)


if __name__ == '__main__':
    main()
