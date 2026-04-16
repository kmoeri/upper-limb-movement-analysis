# src/tracking.py

# required installation:
# install mediapipe library with 'pip install mediapipe'

# libraries
import os
import cv2 as cv
import numpy as np
import mediapipe as mp
from mediapipe import solutions
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

# Initialize mediapipe marker modules
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_holistic = mp.solutions.holistic

# mediapipe drawing utility
mp_drawing = mp.solutions.drawing_utils

# upper limb pose connections
UPPER_BODY_CONNECTIONS = list(mp.solutions.pose.POSE_CONNECTIONS)
LOWER_LIMB_INDICES = set(range(25, 33))
UPPER_BODY_CONNECTIONS_FILTERED = set()
for connection in UPPER_BODY_CONNECTIONS:
    start_idx, end_idx = connection
    if start_idx not in LOWER_LIMB_INDICES and end_idx not in LOWER_LIMB_INDICES:
        UPPER_BODY_CONNECTIONS_FILTERED.add(connection)


# ============================================================================= #
#                        COMBINED POSE AND HAND TRACKER                         #
# ============================================================================= #
# Set 'normalize' to 'true' for world coordinates and to 'false' for normalized values
def combined_landmark_extractor(video_path: str, res_path: str,
                                model_paths: tuple, n_hands: int = 2,
                                min_hand_detect_conf: float = 0.5, min_pose_detect_conf: float = 0.5,
                                min_track_conf: float = 0.8,
                                normalize: bool = True, visualize: bool = False,
                                save_video: bool = True) -> tuple[dict, dict]:
    """
    Tracks hand and pose landmarks in video files using the current 'hand_landmarker.task' and
    'pose_landmarker_heavy.task' models from Google's MediaPipe Solutions.

    Args:
        video_path (str): Absolute path to video file.
        res_path (str): Absolute path of directory to save pose landmark results.
        model_paths (str): Absolute path to hands and pose tracking model (.task)
        n_hands (int, optional): Number of hands to track. Defaults to 2.
        min_hand_detect_conf (float, optional): Minimum detection confidence. Defaults to 0.5.
        min_pose_detect_conf (float, optional): Minimum detection confidence. Defaults to 0.5.
        min_track_conf (float, optional): Minimum tracking confidence. Defaults to 0.8.
        normalize (bool, optional): Whether to use normalized or real world coordinates. Defaults to True.
        visualize (bool, optional): Whether to stream the tracked landmarks on runtime. Defaults to False.
        save_video (bool, optional): Whether to save the video. Defaults to True.

    Returns:
        marker_data_dict (dict): Tracked hand and pose landmark data.
    """

    # helper function to draw pose landmark labels on frame
    def draw_pose(rgb_image, detection_result):
        annotated_img = np.copy(rgb_image)
        if not detection_result.pose_landmarks:
            return annotated_img

        for pose_landmarks in detection_result.pose_landmarks:
            filtered_proto = landmark_pb2.NormalizedLandmarkList()
            for i, landmark in enumerate(pose_landmarks):
                # Filter out lower limbs (25-32)
                if i not in LOWER_LIMB_INDICES:
                    filtered_proto.landmark.extend([
                        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
                    ])

            solutions.drawing_utils.draw_landmarks(
                annotated_img,
                filtered_proto,
                UPPER_BODY_CONNECTIONS_FILTERED,
                solutions.drawing_styles.get_default_pose_landmarks_style()
            )
        return annotated_img

    # helper function to draw hands landmark labels on frame
    def draw_hands(rgb_image, hand_landmarks_lst, handedness_lst, corrected_labels):
        annotated_img = np.copy(rgb_image)
        if not hand_landmarks_lst:
            return annotated_img

        for idx in range(len(hand_landmarks_lst)):
            hand_landmarks = hand_landmarks_lst[idx]
            label = corrected_labels[idx]  # Use our corrected label!

            proto = landmark_pb2.NormalizedLandmarkList()
            proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in hand_landmarks
            ])

            solutions.drawing_utils.draw_landmarks(
                annotated_img, proto, solutions.hands.HAND_CONNECTIONS,
                solutions.drawing_styles.get_default_hand_landmarks_style(),
                solutions.drawing_styles.get_default_hand_connections_style()
            )

            # Draw the corrected label text
            h, w, _ = annotated_img.shape
            x_coords = [lm.x for lm in hand_landmarks]
            y_coords = [lm.y for lm in hand_landmarks]
            cv.putText(annotated_img, label, (int(min(x_coords) * w), int(min(y_coords) * h) - 10),
                       cv.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2, cv.LINE_AA)
        return annotated_img

    # setup the models
    pose_data_dict = {}
    hands_data_dict = {}

    pose_model_path, hands_model_path = model_paths

    # run on CPU
    base_opts_pose = python.BaseOptions(model_asset_path=pose_model_path, delegate=python.BaseOptions.Delegate.CPU)
    base_opts_hand = python.BaseOptions(model_asset_path=hands_model_path, delegate=python.BaseOptions.Delegate.CPU)
    # run on GPU
    #base_opts_pose = python.BaseOptions(model_asset_path=pose_model_path, delegate=python.BaseOptions.Delegate.GPU)
    #base_opts_hand = python.BaseOptions(model_asset_path=hands_model_path, delegate=python.BaseOptions.Delegate.GPU)

    # select the pose landmark tracking model options
    pose_options = vision.PoseLandmarkerOptions(
        base_options=base_opts_pose,
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        min_pose_detection_confidence=min_pose_detect_conf,
        min_tracking_confidence=min_track_conf,
        output_segmentation_masks=False
    )

    # select the hands landmark tracking model options
    hand_options = vision.HandLandmarkerOptions(
        base_options=base_opts_hand,
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=n_hands,
        min_hand_detection_confidence=min_hand_detect_conf,
        min_tracking_confidence=min_track_conf
    )

    # create the hand and pose landmark objects
    with vision.PoseLandmarker.create_from_options(pose_options) as pose_detector, \
            vision.HandLandmarker.create_from_options(hand_options) as hand_detector:

        # create a video capture object and pass a valid video file path
        cap = cv.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open {video_path}")
            return {}, {}

        # variables required to define the number of leading zeros for frame numbering
        orig_w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        f_tot = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        f_digits = len(str(f_tot))
        fps = cap.get(cv.CAP_PROP_FPS)
        fnum = 0

        # initialize video writer for detected landmarks overlay on video
        out_path = os.path.join(res_path, os.path.splitext(os.path.basename(video_path))[0] + '_tracked.mp4')
        if save_video:
            out = cv.VideoWriter(out_path, cv.VideoWriter.fourcc(*'mp4v'), fps, (orig_w, orig_h))

        print(f"Extracting Pose & Hands for {os.path.basename(video_path)}...")

        # iterate over each frame of the video clip using opencv
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break

            # frame processing
            timestamp_ms = int((fnum / fps) * 1000)

            # converts the current frame to a mediapipe image object
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv.cvtColor(image, cv.COLOR_BGR2RGB))

            # perform hand and pose landmark detection on current frame
            pose_res = pose_detector.detect_for_video(mp_image, timestamp_ms)
            hand_res = hand_detector.detect_for_video(mp_image, timestamp_ms)

            # lists to store output for this frame
            p_marker_lst = [(None, None, None)] * 33
            h_marker_lst = [(None, None, None)] * 42

            # A) process pose
            pose_norm_lms = pose_res.pose_landmarks  # always use normalized landmarks for distance reference
            pose_out_lms = pose_res.pose_landmarks if normalize else pose_res.pose_world_landmarks

            left_wrist_pose, right_wrist_pose = None, None

            if pose_norm_lms and len(pose_norm_lms) > 0:
                # get the reference wrists from the pose model (indices 15 = Left, 16 = Right)
                left_wrist_pose = np.array([pose_norm_lms[0][15].x, pose_norm_lms[0][15].y])
                right_wrist_pose = np.array([pose_norm_lms[0][16].x, pose_norm_lms[0][16].y])

            if pose_out_lms and len(pose_out_lms) > 0:
                for i, lm in enumerate(pose_out_lms[0]):
                    if i < 33:
                        p_marker_lst[i] = (lm.x, lm.y, lm.z)

            # B) process hands - includes swap prevention
            hand_norm_lms = hand_res.hand_landmarks  # always use normalized for distance check
            hand_out_lms = hand_res.hand_landmarks if normalize else hand_res.hand_world_landmarks
            handedness = hand_res.handedness

            assigned_this_frame = {'Left': False, 'Right': False}
            corrected_labels_for_drawing = []

            if hand_norm_lms:
                for idx, norm_lm in enumerate(hand_norm_lms):
                    mp_label = handedness[idx][0].category_name

                    # hand wrist is index 0
                    hand_wrist = np.array([norm_lm[0].x, norm_lm[0].y])
                    final_label = mp_label

                    # 1) swap prevention logic: compare detected wrists of hands model to those of the pose model
                    if left_wrist_pose is not None and right_wrist_pose is not None:
                        dist_L = np.linalg.norm(hand_wrist - left_wrist_pose)
                        dist_R = np.linalg.norm(hand_wrist - right_wrist_pose)

                        # override MediaPipe's detection based on physical proximity to the shoulder/arm
                        if dist_L < dist_R:
                            final_label = 'Left'
                        else:
                            final_label = 'Right'

                    # 2) fallback: prevent double assignment in the same frame
                    if assigned_this_frame[final_label]:
                        final_label = 'Right' if final_label == 'Left' else 'Left'

                    assigned_this_frame[final_label] = True
                    corrected_labels_for_drawing.append(final_label)

                    # extract coordinates to the correct base index (0 for 'Left', 21 for 'Right')
                    base_idx = 0 if final_label == 'Left' else 21
                    out_lm = hand_out_lms[idx]
                    for i, lm in enumerate(out_lm):
                        h_marker_lst[base_idx + i] = (lm.x, lm.y, lm.z)

            # draw the landmarks onto the image/frame
            annotated_image = mp_image.numpy_view()
            annotated_image = draw_pose(annotated_image, pose_res)
            annotated_image = draw_hands(annotated_image, hand_norm_lms, handedness, corrected_labels_for_drawing)

            # live visualization of the tracked landmarks
            if visualize:
                cv.imshow("Combined Tracking", cv.cvtColor(annotated_image, cv.COLOR_RGB2BGR))
                if cv.waitKey(1) & 0xFF == ord("q"):
                    break

            # save a video with the tracked landmarks on top of the original recording
            if save_video:
                out.write(cv.cvtColor(annotated_image, cv.COLOR_RGB2BGR))

            # update dictionary: stores the landmark data of each frame in a dict with the frame number as key
            key = str(fnum).zfill(f_digits)
            pose_data_dict[key] = p_marker_lst
            hands_data_dict[key] = h_marker_lst
            fnum += 1

        cap.release()
        cv.destroyAllWindows()

        return pose_data_dict, hands_data_dict

