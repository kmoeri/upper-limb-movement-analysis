# libraries
import os
import cv2 as cv
import numpy as np
import pandas as pd

from mediapipe.framework.formats import landmark_pb2
from mediapipe.python import solutions

# modules
import utils

# order corresponding to the MediaPipe convention
HAND_POINTS_LEFT = [
    'wrist1',                               # Index 0
    'cmc11', 'mcp11', 'ip11', 'ftip11',     # Thumb (Indices 1-4)
    'mcp12', 'pip12', 'dip12', 'ftip12',    # Index Finger (Indices 5-8)
    'mcp13', 'pip13', 'dip13', 'ftip13',    # Middle Finger (Indices 9-12)
    'mcp14', 'pip14', 'dip14', 'ftip14',    # Ring Finger (Indices 13-16)
    'mcp15', 'pip15', 'dip15', 'ftip15',    # Little Finger (Indices 17-20)
]
HAND_POINTS_RIGHT = [
    'wrist2',                               # Index 0
    'cmc21', 'mcp21', 'ip21', 'ftip21',     # Thumb (Indices 1-4)
    'mcp22', 'pip22', 'dip22', 'ftip22',    # Index Finger (Indices 5-8)
    'mcp23', 'pip23', 'dip23', 'ftip23',    # Middle Finger (Indices 9-12)
    'mcp24', 'pip24', 'dip24', 'ftip24',    # Ring Finger (Indices 13-16)
    'mcp25', 'pip25', 'dip25', 'ftip25',    # Little Finger (Indices 17-20)
]

# all 42 points
LANDMARK_NAMES_ALL = HAND_POINTS_LEFT + HAND_POINTS_RIGHT

# Pose landmarks
POSE_ARM_POINTS_LEFT = ['shoulder_right', 'shoulder_left', 'elbow_left', 'wrist_left']
POSE_ARM_POINTS_RIGHT = ['shoulder_left', 'shoulder_right', 'elbow_right', 'wrist_right']


def _extract_single_hand_data(row: pd.Series, hand_label: str) -> dict[str, list[tuple[float, float, float]]]:
    """
    Maps DataFrame row data for a single specified hand to the drawing structure.

    Args:
        row (pd.Series): A single row from the preprocessed DataFrame.
        hand_label (str): The hand to extract ('Left' or 'Right').

    Returns:
        dict: {'Left': [...], 'Right': [...]} where only the specified hand contains 21 (x, y, z) tuples,
        and the other hand is empty.
    """

    frame_landmarks: dict[str, list[tuple[float, float, float]]] = {'Left': [], 'Right': []}

    if hand_label == 'Left':
        points_list = HAND_POINTS_LEFT
        prefix = 'wrist1'                   # check for existence of a key point
        target_label = 'Left'
    elif hand_label == 'Right':
        points_list = HAND_POINTS_RIGHT
        prefix = 'wrist2'                   # check for existence of a key point
        target_label = 'Right'
    else:
        return frame_landmarks              # return empty if label is invalid

    # check if the hand's primary point exists in the DataFrame row
    if f'{prefix}_x' not in row:
        return frame_landmarks

    landmarks = []
    for base_name in points_list:
        # assumption: if the wrist is present, all other points are present (or filled with NaNs in preprocessing)
        try:
            landmarks.append((row[f'{base_name}_x'], row[f'{base_name}_y'], row[f'{base_name}_z']))
        except KeyError:
            # if a point is unexpectedly missing, fall back to empty list
            landmarks = []
            break

    frame_landmarks[target_label] = landmarks
    return frame_landmarks


def _extract_single_arm_data(row: pd.Series, arm_label: str) -> dict[str, list[tuple[float, float, float]]]:
    """
    Maps DataFrame row data for a single specified arm (shoulder, elbow, wrist)
    from the pose-processed file to the drawing structure.

    Args:
        row (pd.Series): A single row from the pose-processed DataFrame.
        arm_label (str): The arm to extract ('Left' or 'Right').

    Returns:
        dict: {'Left': [...], 'Right': [...]} where only the specified arm
              contains 3 (x, y, z) tuples (shoulder, elbow, wrist).
    """

    frame_landmarks: dict[str, list[tuple[float, float, float]]] = {'Left': [], 'Right': []}

    if arm_label == 'Left':
        points_list = POSE_ARM_POINTS_LEFT
        target_label = 'Left'
    elif arm_label == 'Right':
        points_list = POSE_ARM_POINTS_RIGHT
        target_label = 'Right'
    else:
        return frame_landmarks

    landmarks = []
    # check for the key points (e.g., shoulder)
    if f'{points_list[0]}_x' not in row:
        return frame_landmarks

    for base_name in points_list:
        try:
            # structure is: base_name_x, base_name_y, base_name_z
            landmarks.append((row[f'{base_name}_x'], row[f'{base_name}_y'], row[f'{base_name}_z']))
        except KeyError:
            # if any required point is missing, treat the arm data as missing
            landmarks = []
            break

    frame_landmarks[target_label] = landmarks
    return frame_landmarks


def draw_combined_landmarks(rgb_image: np.ndarray, hand_data: dict, pose_data: dict, focus_label: str) -> np.ndarray:
    """
    Draws filtered Hand landmarks and Pose arm/shoulder landmarks on an image.

    Args:
        rgb_image (np.ndarray): The current video frame (BGR format from OpenCV).
        hand_data (dict): Dictionary with 21 (x, y, z) tuples for hands. Keys: 'Left' and 'Right'.
        pose_data (dict): Dictionary with up to 3 (x, y, z) tuples for arms/shoulders. Keys: 'Left' and 'Right'.
        focus_label (str): The hand/arm currently being focused on ('Left' or 'Right').

    Returns:
        annotated_img: Image array with skeletal overlay.
    """
    annotated_img = np.copy(rgb_image)
    annotated_img = cv.cvtColor(annotated_img, cv.COLOR_BGR2RGB)

    # 1) draw Hand landmarks
    for hand_label, landmarks_list in hand_data.items():
        if not landmarks_list:
            continue

        # only draw the hand if it is the focused hand
        if hand_label != focus_label:
            continue

        # convert list of tuples into the MediaPipe NormalizedLandmarkList structure
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=lm[0], y=lm[1], z=lm[2]) for lm in landmarks_list
        ])

        solutions.drawing_utils.draw_landmarks(
            annotated_img,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style()
        )

        # add handedness label (using wrist from hand model for text placement)
        height, width, _ = annotated_img.shape
        wrist_x = landmarks_list[0][0]
        wrist_y = landmarks_list[0][1]

        text_x = int(wrist_x * width)
        text_y = int(wrist_y * height) - 10

        cv.putText(annotated_img, hand_label, (text_x, text_y),
                   cv.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 0), 2, cv.LINE_AA)

    # 2) draw Pose arm/shoulder landmarks and connections
    pose_landmarks_list = pose_data.get(focus_label, [])

    if pose_landmarks_list:
        # map the list of (x, y, z) tuples to a MediaPipe proto list
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=lm[0], y=lm[1], z=lm[2]) for lm in pose_landmarks_list
        ])

        # define connections for the arm: shoulder1 -> shoulder2 -> elbow -> wrist
        ARM_CONNECTIONS = [(0, 1), (1, 2), (2, 3)]

        # use a simple, custom style for the arm lines and dots
        landmark_style = solutions.drawing_styles.DrawingSpec(
            color=(0, 255, 255),  # Yellow dots
            thickness=2,
            circle_radius=4
        )
        connection_style = solutions.drawing_styles.DrawingSpec(
            color=(255, 0, 0),  # Blue lines
            thickness=4
        )

        solutions.drawing_utils.draw_landmarks(
            annotated_img,
            pose_landmarks_proto,
            ARM_CONNECTIONS,
            landmark_drawing_spec=landmark_style,
            connection_drawing_spec=connection_style
        )

    # convert back to BGR for OpenCV writer
    return cv.cvtColor(annotated_img, cv.COLOR_RGB2BGR)


def render_focused_hand_and_arm(video_path: str, hand_df: pd.DataFrame, pose_df: pd.DataFrame, out_fpath: str) -> None:
    """
    Generates a video overlaying landmarks for only the specified focus hand (from hand_df)
    and the corresponding arm/shoulder (from pose_df).

    Args:
        video_path (str): Original video file path.
        hand_df (pd.DataFrame): DataFrame with hand landmark columns ('hands_processed.csv').
        pose_df (pd.DataFrame): DataFrame with filtered pose arm/shoulder columns ('pose_processed.csv').
        out_fpath (str): Directory to save the output video.

    Returns:
        None
    """

    # infer focused side for hand and pose data
    focused_side_hand: str = utils.infer_focus_side(hand_df, 'Hand')
    focused_side_pose: str = utils.infer_focus_side(pose_df, 'Pose')

    # handle missing value for focused hand/pose side
    if focused_side_hand is None or focused_side_pose is None:
        print(f'Error: Could not automatically infer focused side from the underlying data of {os.path.basename(video_path)}. '
              f'Hand side: {focused_side_hand}, Pose side: {focused_side_pose}')
        return

    # handle inconsistency
    if focused_side_hand != focused_side_pose:
        print(f'Error: Inconsistent focused side detected. '
              f'Hand side: {focused_side_hand}, Pose side: {focused_side_pose}')
        return

    # create video capture object
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return

    # get video information
    original_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv.CAP_PROP_FPS)

    # create video writer object
    out = cv.VideoWriter(out_fpath, cv.VideoWriter.fourcc(*'mp4v'), fps, (original_width, original_height))

    print(f"Rendering combined hand and arm overlay to: {out_fpath}...")

    # ensure both dataframes have the same indices
    max_frames = min(len(hand_df), len(pose_df))

    # iterate over each frame
    fnum = 0
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        if fnum < max_frames:
            hand_row = hand_df.iloc[fnum]
            pose_row = pose_df.iloc[fnum]

            # extract hand data (21 landmarks)
            hand_landmarks = _extract_single_hand_data(hand_row, focused_side_hand)

            # extract pose arm data (4 points: shoulders, elbow, wrist)
            pose_landmarks = _extract_single_arm_data(pose_row, focused_side_hand)

            # draw combined landmarks
            annotated_image = draw_combined_landmarks(
                image,
                hand_data=hand_landmarks,
                pose_data=pose_landmarks,
                focus_label=focused_side_hand
            )

            # write the frame
            out.write(annotated_image)

        fnum += 1

    cap.release()
    out.release()
    cv.destroyAllWindows()
    print("Rendering complete.")
