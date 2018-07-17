# From Python
import sys
import cv2
import os
import time
import click
from trackers import PoseTracker
sys.path.append('build/python/openpose')
from openpose import *

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

# Setup Video
cap = cv2.VideoCapture(sys.argv[1])
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output/output.avi',fourcc, fps, (frame_width, frame_height))  # Width, Height
frame_counter = 0
video_progress_bar = click.progressbar(length=total_frames)

# Setup tracker
body_part_keys = [
    "nose", "neck", "rshoulder", "relbow", "rwrist", "lshoulder", "lelbow", "lwrist", "midhip",
    "rhip", "rknee", "rankle", "lhip", "lknee", "lankle", "reye", "leye", "rear", "lear",
    "lbigtoe", "lsmalltoe", "lheel", "rbigtoe", "rsmalltoe", "rheel"
]
body_parts_to_track_keys = ['nose', 'reye', 'leye']
tracker = PoseTracker(tracked_points_num=len(body_parts_to_track_keys))

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
        post_pred = time.time()

        # Filter out body parts we do not wish to track
        body_parts_to_track_indexes = [body_part_keys.index(key) for key in body_parts_to_track_keys]
        detected_people = detected_people[:, body_parts_to_track_indexes]

        # Filter out probs from body parts data points
        detected_people = detected_people[:, :, :2]

        # Get predictions (it returns unmatched_people only for debugging)
        predicted_people, unmatched_people = tracker.update(detected_people)

        # Draw each person
        for predicted_person in predicted_people:
            body_parts = {key: value for key, value in zip(body_parts_to_track_keys, predicted_person['pose'].astype(int))}

            triangles = [
                ('nose', 'reye', 'leye'),
                # ('nose', 'rear', 'lear'),
            ]

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
                    if right_thresh < mid[0] < left_thresh:
                        color = (0, 0, 255)
                    else:
                        color = (0, 255, 0)
                    cv2.line(frame, mid, right, color)
                    cv2.line(frame, mid, left, color)
                    cv2.line(frame, right, left, color)
                    cv2.circle(frame, mid, 5, color)
                    cv2.putText(frame, str(predicted_person['id']), mid, cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

        cv2.imshow("output", frame)
        out.write(frame)
        end_frame = time.time()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    end = time.time()

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()
print("fps {:2f}".format(frame_counter / (end - start)))
