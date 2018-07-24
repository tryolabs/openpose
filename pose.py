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
BODY_PART_KEYS = [
    "nose", "neck", "rshoulder", "relbow", "rwrist", "lshoulder", "lelbow", "lwrist", "midhip",
    "rhip", "rknee", "rankle", "lhip", "lknee", "lankle", "reye", "leye", "rear", "lear",
    "lbigtoe", "lsmalltoe", "lheel", "rbigtoe", "rsmalltoe", "rheel"
]
BODY_PARTS_TO_TRACK_KEYS = ['nose', 'reye', 'leye', 'midhip']  # Using only these 4 because dissapearing limbs don't play well with kalman filters


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
    tracker = PoseTracker(tracked_points_num=len(BODY_PARTS_TO_TRACK_KEYS), score_threshold = 250)

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

            # Draw each person
            # Predicted
            for predicted_person in predicted_people:
                draw_body_parts(frame, predicted_person, [('nose', 'reye', 'leye')], (0, 255, 0), 3, predicted_person.get('debug'))

            if DEBUG:
                # Unmatched
                for unmatched_person in unmatched_people:
                    draw_body_parts(frame, unmatched_person, [('nose', 'reye', 'leye')], (255, 0, 0), 1, unmatched_person.get('debug'))

                # Detected
                for detected_person_pose in detected_people:
                    draw_body_parts(frame, {'pose': detected_person_pose}, [('nose', 'reye', 'leye')], (255, 255, 255), 1)
        
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


def draw_body_parts(frame, person, triangles, color, width, debug_=None):
    body_parts = {key: value for key, value in zip(BODY_PARTS_TO_TRACK_KEYS, person['pose'].astype(int))}

    thresh_fraction = 1/4
    for mid, right, left in triangles:
        mid = tuple(body_parts[mid])
        right = tuple(body_parts[right])
        left = tuple(body_parts[left])

        # Draw triangle
        # NOTE: the `not` is an ulgy ugly hack, sorry
        if all(mid) and all(right) and all(left) and not(sum(mid) < 2 or sum(right) < 2 or sum(left) < 2):
            eye_x_dist = abs(right[0] - left[0])
            # Check that nose is in mid fraction of face, so they are looking straight ahead
            right_thresh = right[0] + eye_x_dist * thresh_fraction
            left_thresh = left[0] - eye_x_dist * thresh_fraction
            if right_thresh < mid[0] < left_thresh:
                pass
            try:
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
            except:
                pass


def get_color(class_label):
    """Rudimentary way to create color palette for plotting clases.

    Accepts integer or strings as class_labels.
    """
    # We get these colors from the luminoth web client
    web_colors_hex = [
        'ff0029', '377eb8', '66a61e', '984ea3', '00d2d5', 'ff7f00', 'af8d00',
        '7f80cd', 'b3e900', 'c42e60', 'a65628', 'f781bf', '8dd3c7', 'bebada',
        'fb8072', '80b1d3', 'fdb462', 'fccde5', 'bc80bd', 'ffed6f', 'c4eaff',
        'cf8c00', '1b9e77', 'd95f02', 'e7298a', 'e6ab02', 'a6761d', '0097ff',
        '00d067', '000000', '252525', '525252', '737373', '969696', 'bdbdbd',
        'f43600', '4ba93b', '5779bb', '927acc', '97ee3f', 'bf3947', '9f5b00',
        'f48758', '8caed6', 'f2b94f', 'eff26e', 'e43872', 'd9b100', '9d7a00',
        '698cff', 'd9d9d9', '00d27e', 'd06800', '009f82', 'c49200', 'cbe8ff',
        'fecddf', 'c27eb6', '8cd2ce', 'c4b8d9', 'f883b0', 'a49100', 'f48800',
        '27d0df', 'a04a9b',
    ]
    hex_color = web_colors_hex[hash(class_label) % len(web_colors_hex)]
    return hex_to_rgb(hex_color)


def hex_to_rgb(x):
    return [int(x[i:i + 2], 16) for i in (0, 2, 4)]


if __name__ == '__main__':
    main()
