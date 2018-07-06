# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
import time
import click
sys.path.append('build/python/openpose')
from openpose import *


def generate_mappings(Ws, Hs, Wd, Hd, R1, R2, Cx, Cy):
    """Generates mapping between fisheye and panoramic view.

    This will only be generated once per processed video."""
    # TODO: Change to fixed point??
    x_map = np.zeros((Hd, Wd), np.float32)
    y_map = np.zeros((Hd, Wd), np.float32)
    for y in range(Hd):
        for x in range(Wd):
            r = R2 - (float(y) / float(Hd)) * (R2 - R1) + R1
            theta = (float(x) / float(Wd)) * 2.0 * np.pi
            xS = Cx + r * np.sin(theta)
            yS = Cy + r * np.cos(theta)
            x_map.itemset((y, x), int(xS))
            y_map.itemset((y, x), int(yS))
    return x_map, y_map

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
# Ensure you point to the correct path where models are located
params["default_model_folder"] = dir_path + "/models/"
# Construct OpenPose object allocates GPU memory

# params["disable_multi_thread"] = True  # TODO Try this (due to https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/demo_overview.md )
openpose = OpenPose(params)
cap = cv2.VideoCapture(sys.argv[1])
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output/output.avi',fourcc, fps, (6114, 923))  # Width, Height
frame_counter = 0
video_progress_bar = click.progressbar(
    length=total_frames,
)

if sys.argv[2] == 'fish':
    # Input frame size
    Hs, Ws, = frame_height, frame_width 
    # Donut's properties. Our donut seems to be centered, so I eyeball its position here
    Cx = Ws / 2
    Cy = Hs / 2
    R1 = 50
    R2 = int(round((Hs / 2) * .95))

    # Output frame size
    Wd = int(round(2 * np.pi * R2))
    Hd = R2 - R1

    x_map, y_map = generate_mappings(Ws, Hs, Wd, Hd, R1, R2, Cx, Cy)

with video_progress_bar as progress_bar:
    start = time.time()
    for _ in progress_bar:
        frame_counter += 1
        # Read frame
        start_frame = time.time()
        ret, frame = cap.read()
        if frame is None:
            break
        # frame = cv2.imread("examples/media/COCO_val2014_000000000192.jpg")

        # Output keypoints and the image with the human skeleton blended on it
        pano_frame = cv2.remap(frame, x_map, y_map, cv2.INTER_LINEAR)
        pre_pred = time.time()
        keypoints, output_image = openpose.forward(pano_frame, True)
        post_pred = time.time()

        # Draw each person
        for person in keypoints:
            part_list = [
                "nose", "neck", "rshoulder", "relbow", "rwrist", "lshoulder", "lelbow", "lwrist", "midhip",
                "rhip", "rknee", "rankle", "lhip", "lknee", "lankle", "reye", "leye", "rear", "lear",
                "lbigtoe", "lsmalltoe", "lheel", "rbigtoe", "rsmalltoe", "rheel"
            ]
            body_parts = {key: tuple(coords) for (key, coords) in zip(part_list, person[:, :2])}

            triangles = [
                ('nose', 'reye', 'leye'),
                ('nose', 'rear', 'lear'),
                # ('neck', 'rshoulder', 'lshoulder'),
                # ('midhip', 'rhip', 'lhip')
            ]

            for mid, right, left in triangles:
                mid = body_parts[mid]
                right = body_parts[right]
                left = body_parts[left]

                # Draw triangle
                if all(mid) and all(right) and all(left):
                    eye_x_dist = abs(right[0] - left[0])
                    # Check that nose is in mid third of face, so they are looking straight ahead
                    right_thresh = right[0] + eye_x_dist / 3
                    left_thresh = left[0] - eye_x_dist / 3
                    if right_thresh < mid[0] < left_thresh:
                        color = (0, 0, 255)
                    else:
                        color = (0, 255, 0)
                    cv2.line(frame, mid, right, color)
                    cv2.line(frame, mid, left, color)
                    cv2.line(frame, right, left, color)
                    cv2.circle(frame, mid, 5, color)

        # cv2.imshow("output", frame)
        out.write(output_image)
        end_frame = time.time()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Profiling
        # print(end - start, post_pred - pre_pred)
    end = time.time()

cap.release()
out.release()
cv2.destroyAllWindows()
print("fps {:2f}".format(frame_counter / (end - start)))
