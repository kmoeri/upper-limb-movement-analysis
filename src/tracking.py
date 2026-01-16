# install mediapipe library -> pip install mediapipe

# Libraries
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


#################################################################################
###                                1) POSE TRACKER                            ###
#################################################################################
# --- 1.1) WORLD COORDINATES (Current Pose Tracker Model (Oct 2025)) ---------- #
def pose_landmark_extractor_WC(video_path: str, res_path: str,
                               model_path: str, visualize: bool = False,
                               save_video: bool = True) -> dict:
    """
    Tracks world coordinate pose landmarks in video files using the current pose_landmarker_heavy.task model
    (model card date: April, 16, 2021) by Google's MediaPipe Solutions and stores the corresponding coordinates.

    Args:
        video_path (str): Absolute path to video file.
        res_path (str): Absolute path of directory to save pose landmark results.
        model_path (str): Absolute path to pose tracking model (.task)
        visualize (int, optional): Whether to stream the tracked landmarks on runtime. Defaults to False.
        save_video (bool, optional): Whether to save the video. Defaults to True.

    Returns:
        marker_data_dict (dict): Tracked pose landmark data.
    """

    # draw landmarks on image to create video with skeletal overlay
    def draw_pose_landmarks_on_image(rgb_image, detection_result):
        """
        Draws pose landmarks on an image.

        Args:
            rgb_image (mp.Image): Mediapipe Image object.
            detection_result (PoseLandmarkerResult): Mediapipe Pose Landmark results.

        Returns:
            annotated_img (ndarray): Array with pose landmarks annotated on image.
        """

        pose_landmarks_lst = detection_result.pose_landmarks
        annotated_img = np.copy(rgb_image)

        for idx in range(len(pose_landmarks_lst)):
            pose_landmarks = pose_landmarks_lst[idx]

            # filter the landmarks for upper limb (knees, ankle, heel, and index are removed)
            filtered_pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            for i, landmark in enumerate(pose_landmarks):
                if i not in LOWER_LIMB_INDICES:
                    filtered_pose_landmarks_proto.landmark.extend([
                        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
                    ])

            # plot all landmarks
            #pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            #pose_landmarks_proto.landmark.extend([
            #    landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
            #])

            solutions.drawing_utils.draw_landmarks(
                annotated_img,
                filtered_pose_landmarks_proto,
                #solutions.pose.POSE_CONNECTIONS,
                UPPER_BODY_CONNECTIONS_FILTERED,
                solutions.drawing_styles.get_default_pose_landmarks_style()
            )

        return annotated_img

    # dictionary holding the frame number as key and the hand marker coordinates as values
    marker_data_dict: dict = {}

    # select the options for the landmark object
    base_options = python.BaseOptions(model_asset_path=model_path, delegate=python.BaseOptions.Delegate.GPU)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.8,
        output_segmentation_masks=False
    )

    # create the pose landmark object
    detector = vision.PoseLandmarker.create_from_options(options)

    # create a video capture object and pass a valid video file path
    cap: cv.VideoCapture = cv.VideoCapture(video_path)

    # variables required to define the number of leading zeros for frame numbering
    original_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    f_tot = int(cap.get(cv.CAP_PROP_FRAME_COUNT))  # total number of frames in video
    f_digits = len(str(f_tot))  # total digits used in frame number
    fps = cap.get(cv.CAP_PROP_FPS)
    fnum: int = 0  # frame number

    # initialize video writer for detected landmarks overlay on video
    out_path = os.path.join(res_path, os.path.splitext(os.path.basename(video_path))[0] + '_pose_tracked.mp4')
    out = cv.VideoWriter(out_path, cv.VideoWriter.fourcc(*'mp4v'), 90.0, (original_width, original_height))

    print("Extracting pose landmarks...")
    # iterate over each frame of the video clip
    while cap.isOpened():  # returns true if video capture constructor succeeded initialization
        success, image = cap.read()  # grabs, decodes and returns next video frame
        if not success:  # returns false at the end of the video or if no frames were found
            print("End of video reached.")
            break

        # frame processing
        frame_timestamp_ms = int((fnum / fps) * 1000)

        # converts the current frame to a mediapipe image object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv.cvtColor(image, cv.COLOR_BGR2RGB))

        # perform pose landmark detection on current frame
        detection_result = detector.detect_for_video(mp_image, frame_timestamp_ms)

        # defines a list of tuples with None for x, y, and z coordinate for each landmark, when hand is not detected
        marker_lst: list = [(None, None, None)] * 33

        # overlay landmarks on the image
        annotated_image = draw_pose_landmarks_on_image(mp_image.numpy_view(), detection_result)

        if visualize:
            # visualize the detected landmarks
            cv.imshow("Pose Tracking", cv.cvtColor(annotated_image, cv.COLOR_RGB2BGR))
            if cv.waitKey(1) & 0xFF == ord("q"):
                break

        #pose_landmarks_lst = detection_result.pose_landmarks
        pose_landmarks_lst = detection_result.pose_world_landmarks

        # loop through detected hand landmarks
        for pose_landmarks in pose_landmarks_lst:
            # assumes that there is only one person being tracked
            for i, landmark in enumerate(pose_landmarks):
                if i < 33:
                    marker_lst[i] = (landmark.x, landmark.y, landmark.z)

        base_key: str = str('0' * f_digits)
        key: str = base_key[:-len(str(fnum))] + str(fnum)
        marker_data_dict[key] = marker_lst
        fnum += 1

        if save_video:
            # Write video to result path
            out.write(cv.cvtColor(annotated_image, cv.COLOR_RGB2BGR))

    cap.release()
    cv.destroyAllWindows()

    return marker_data_dict


# --- 1.2) NORMALIZED LANDMARKS (Current Pose Tracker Model (Oct 2025)) ------- #
def pose_landmark_extractor(video_path: str, res_path: str, model_path: str,
                            min_pose_detect_conf: float = 0.5,
                            min_pose_track_conf: float = 0.8, normalize: bool = True,
                            visualize: bool = False,  save_video: bool = True) -> dict:
    """
    Tracks normalized pose landmarks in video files using the current pose_landmarker_heavy.task model
    (model card date: April, 16, 2021) by Google's MediaPipe Solutions and stores the corresponding coordinates.

    Args:
        video_path (str): Absolute path to video file.
        res_path (str): Absolute path of directory to save pose landmark results.
        model_path (str): Absolute path to pose tracking model (.task)
        min_pose_detect_conf (float, optional): Minimum pose detection confidence threshold. Defaults to 0.5.
        min_pose_track_conf (float, optional): Minimum pose tracking confidence threshold. Defaults to 0.8.
        normalize (bool, optional): Whether to use normalized or real world coordinates. Defaults to True.
        visualize (int, optional): Whether to stream the tracked landmarks on runtime. Defaults to False.
        save_video (bool, optional): Whether to save the video. Defaults to True.

    Returns:
        marker_data_dict (dict): Tracked pose landmark data.
    """

    # draw landmarks on image to create video with skeletal overlay
    def draw_pose_landmarks_on_image(rgb_image, detection_result):
        """
        Draws pose landmarks on an image.

        Args:
            rgb_image (mp.Image): MediaPipe Image object.
            detection_result (PoseLandmarkerResult): Mediapipe Pose Landmark results.

        Returns:
            annotated_img (ndarray): Array with pose landmarks annotated on image.
        """

        pose_landmarks_lst = detection_result.pose_landmarks
        annotated_img = np.copy(rgb_image)

        for idx in range(len(pose_landmarks_lst)):
            pose_landmarks = pose_landmarks_lst[idx]

            # filter the landmarks for upper limb (knees, ankle, heel, and index are removed)
            filtered_pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            for i, landmark in enumerate(pose_landmarks):
                if i not in LOWER_LIMB_INDICES:
                    filtered_pose_landmarks_proto.landmark.extend([
                        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
                    ])

            # plot all landmarks
            #pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            #pose_landmarks_proto.landmark.extend([
            #    landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
            #])

            solutions.drawing_utils.draw_landmarks(
                annotated_img,
                filtered_pose_landmarks_proto,
                #solutions.pose.POSE_CONNECTIONS,
                UPPER_BODY_CONNECTIONS_FILTERED,
                solutions.drawing_styles.get_default_pose_landmarks_style()
            )

        return annotated_img

    # dictionary holding the frame number as key and the hand marker coordinates as values
    marker_data_dict: dict = {}

    # select the options for the landmark object
    base_options = python.BaseOptions(model_asset_path=model_path, delegate=python.BaseOptions.Delegate.GPU)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        min_pose_detection_confidence=min_pose_detect_conf,
        min_tracking_confidence=min_pose_track_conf,
        output_segmentation_masks=False
    )

    # create the pose landmark object
    detector = vision.PoseLandmarker.create_from_options(options)

    # create a video capture object and pass a valid video file path
    cap: cv.VideoCapture = cv.VideoCapture(video_path)

    # variables required to define the number of leading zeros for frame numbering
    original_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    f_tot = int(cap.get(cv.CAP_PROP_FRAME_COUNT))  # total number of frames in video
    f_digits = len(str(f_tot))  # total digits used in frame number
    fps = cap.get(cv.CAP_PROP_FPS)
    fnum: int = 0  # frame number

    # initialize video writer for detected landmarks overlay on video
    out_path = os.path.join(res_path, os.path.splitext(os.path.basename(video_path))[0] + '_pose_tracked.mp4')
    out = cv.VideoWriter(out_path, cv.VideoWriter.fourcc(*'mp4v'), 90.0, (original_width, original_height))

    print("Extracting pose landmarks...")
    # iterate over each frame of the video clip
    while cap.isOpened():  # returns true if video capture constructor succeeded initialization
        success, image = cap.read()  # grabs, decodes and returns next video frame
        if not success:  # returns false at the end of the video or if no frames were found
            print("End of video reached.")
            break

        # frame processing
        frame_timestamp_ms = int((fnum / fps) * 1000)

        # converts the current frame to a mediapipe image object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv.cvtColor(image, cv.COLOR_BGR2RGB))

        # perform pose landmark detection on current frame
        detection_result = detector.detect_for_video(mp_image, frame_timestamp_ms)

        # defines a list of tuples with None for x, y, and z coordinate for each landmark, when hand is not detected
        marker_lst: list = [(None, None, None)] * 33

        # overlay landmarks on the image
        annotated_image = draw_pose_landmarks_on_image(mp_image.numpy_view(), detection_result)

        # visualize during runtime
        if visualize:
            cv.imshow("Pose Tracking", cv.cvtColor(annotated_image, cv.COLOR_RGB2BGR))
            if cv.waitKey(1) & 0xFF == ord("q"):
                break

        # if true, extract normalized landmarks
        if normalize:
            pose_landmark_lst = detection_result.pose_landmarks
        else:
            pose_landmark_lst = detection_result.pose_world_landmarks

        # iterate over each detected pose landmark
        for pose_landmarks in pose_landmark_lst:
            # assumes that there is only one person being tracked
            for i, landmark in enumerate(pose_landmarks):
                if i < 33:
                    marker_lst[i] = (landmark.x, landmark.y, landmark.z)

        # store the landmark data of each frame in a dict with the frame number as key)
        base_key: str = str('0' * f_digits)
        key: str = base_key[:-len(str(fnum))] + str(fnum)
        marker_data_dict[key] = marker_lst
        fnum += 1

        # if true, save the tracking overlaid on the original video as separate 'mp4' file
        if save_video:
            # Write video to result path
            out.write(cv.cvtColor(annotated_image, cv.COLOR_RGB2BGR))

    cap.release()
    cv.destroyAllWindows()

    return marker_data_dict


#################################################################################
###                                2) HAND TRACKER                            ###
#################################################################################
# --- 2.1) WORLD COORDINATES (Current Hand Tracker Model (Oct 2025)) ---------- #
def hand_landmark_extractor_WC(video_path: str, res_path: str,
                               model_path: str, visualize: bool = True,
                               save_video: bool = True) -> dict:
    """
    Tracks world coordinate hand landmarks in video files using the current hand_landmarker.task model
    (model card date: February, 2021) by Google's MediaPipe Solutions and stores the corresponding coordinates.

    Args:
        video_path (str): Absolute path to video file.
        res_path (str): Absolute path of directory to save pose landmark results.
        model_path (str): Absolute path to hands tracking model (.task)
        visualize (int, optional): Whether to stream the tracked landmarks on runtime. Defaults to False.
        save_video (bool, optional): Whether to save the video. Defaults to True.

    Returns:
        marker_data_dict (dict): Tracked hands landmark data.
    """

    # draw landmarks on image to create video with skeletal overlay
    def draw_hand_landmarks_on_image(rgb_image, detection_result):
        """
        Draws pose landmarks on an image.

        Args:
            rgb_image (mp.Image): Mediapipe Image object.
            detection_result (PoseLandmarkerResult): Mediapipe Pose Landmark results.

        Returns:
            annotated_img (ndarray): Array with pose landmarks annotated on image.
        """

        hand_landmark_lst = detection_result.hand_landmarks
        handedness_lst = detection_result.handedness
        hand_world_landmarks_lst = detection_result.hand_world_landmarks

        annotated_img = np.copy(rgb_image)

        # loop through detected hand landmarks
        for idx in range(len(hand_landmark_lst)):
            hand_landmarks = hand_landmark_lst[idx]
            handedness = handedness_lst[idx]

            # draw hand landmarks
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
            ])

            solutions.drawing_utils.draw_landmarks(
                annotated_img,
                hand_landmarks_proto,
                solutions.hands.HAND_CONNECTIONS,
                solutions.drawing_styles.get_default_hand_landmarks_style(),
                solutions.drawing_styles.get_default_hand_connections_style()
            )

            height, width, _ = annotated_img.shape
            x_coordinates = [landmark.x for landmark in hand_landmarks]
            y_coordinates = [landmark.y for landmark in hand_landmarks]
            text_x = int(min(x_coordinates) * width)
            text_y = int(min(y_coordinates) * height) - 10  # Margin: 10 px

            cv.putText(annotated_img, f'{handedness[0].category_name}', (text_x, text_y),
                       cv.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2, cv.LINE_AA)

        return annotated_img

    # dictionary holding the frame number as key and the hand marker coordinates as values
    marker_data_dict: dict = {}

    # select the options for the landmark object
    base_options = python.BaseOptions(model_asset_path=model_path, delegate=python.BaseOptions.Delegate.GPU)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.8
    )

    # create the hand landmark object
    with vision.HandLandmarker.create_from_options(options) as detector:

        # create a video capture object and pass a valid video file path
        cap: cv.VideoCapture = cv.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Error: Could not open video file at {video_path}")
            return marker_data_dict

        # variables required to define the number of leading zeros for frame numbering
        original_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        f_tot = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        f_digits = len(str(f_tot))
        fps = cap.get(cv.CAP_PROP_FPS)
        fnum: int = 0

        # Target height for HD quality
        target_height = 1544

        # initialize video writer for detected landmarks overlay on video
        out_path = os.path.join(res_path, os.path.splitext(os.path.basename(video_path))[0] + '_hands_tracked.mp4')
        out = cv.VideoWriter(out_path, cv.VideoWriter.fourcc(*'mp4v'), 90.0, (original_width, original_height))

        print("Extracting hand landmarks...")

        # iterate over each frame of the video clip
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("End of video reached.")
                break

            # frame processing
            frame_timestamp_ms = int((fnum / fps) * 1000)

            # calculate target width maintaining the original aspect ratio
            #target_width = int(round(original_width * (target_height / original_height)))
            #image = cv.resize(image, (target_width, target_height), interpolation=cv.INTER_LINEAR)

            # converts the current frame to a mediapipe image object
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv.cvtColor(image, cv.COLOR_BGR2RGB))

            # perform hand landmark detection on current frame
            # NOTE: passing the timestamp is CRITICAL for VIDEO mode smoothing
            detection_result = detector.detect_for_video(mp_image, frame_timestamp_ms)

            # Initialize marker list for 42 landmarks (21 Left, 21 Right)
            marker_lst: list = [(None, None, None)] * 42

            # overlay landmarks on the image
            annotated_image = draw_hand_landmarks_on_image(mp_image.numpy_view(), detection_result)
            if visualize:
                # visualize the detected landmarks
                cv.imshow("Hand Tracking", cv.cvtColor(annotated_image, cv.COLOR_RGB2BGR))
                if cv.waitKey(1) & 0xFF == ord("q"):
                    break

            #hand_landmark_lst = detection_result.hand_landmarks
            hand_landmark_lst = detection_result.hand_world_landmarks
            handedness_lst = detection_result.handedness

            for idx in range(len(hand_landmark_lst)):
                hand_landmarks = hand_landmark_lst[idx]
                handedness = handedness_lst[idx]

                # Determine the base index: 0-20 for Left, 21-41 for Right
                base_index = 0
                if handedness[0].category_name == 'Right':
                    base_index = 21

                for i, landmark in enumerate(hand_landmarks):
                    marker_lst[base_index + i] = (landmark.x, landmark.y, landmark.z)

            # Store the data
            base_key: str = str('0' * f_digits)
            key: str = base_key[:-len(str(fnum))] + str(fnum)
            marker_data_dict[key] = marker_lst
            fnum += 1

            if save_video:
                # Write video to result path
                out.write(cv.cvtColor(annotated_image, cv.COLOR_RGB2BGR))

        cap.release()
        cv.destroyAllWindows()

        return marker_data_dict


# --- 2.2) NORMALIZED LANDMARKS (Current Hand Tracker Model (Oct 2025)) ------- #
# TODO: fix hand swapping by assigning the hands detected in the current frame to
#       the closest match t-1 (previous frame) by calculating the euclidean distance
def hand_landmark_extractor(video_path: str, res_path: str, model_path: str,
                            n_hands: int = 2, min_hand_detect_conf: float = 0.5,
                            min_hand_track_conf: float = 0.8, normalize: bool = True,
                            visualize: bool = True, save_video: bool = True) -> dict:
    """
    Tracks hand landmarks in video files using the current hand_landmarker.task model
    (model card date: February, 2021) by Google's MediaPipe Solutions and stores the corresponding coordinates.

    Args:
        video_path (str): Absolute path to video file.
        res_path (str): Absolute path of directory to save pose landmark results.
        model_path (str): Absolute path to hands tracking model (.task)
        n_hands (int, optional): Number of hands to track. Defaults to 2.
        min_hand_detect_conf (float, optional): Minimum detection confidence. Defaults to 0.5.
        min_hand_track_conf (float, optional): Minimum tracking confidence. Defaults to 0.8.
        normalize (bool, optional): Whether to use normalized or real world coordinates. Defaults to True.
        visualize (bool, optional): Whether to stream the tracked landmarks on runtime. Defaults to False.
        save_video (bool, optional): Whether to save the video. Defaults to True.

    Returns:
        marker_data_dict (dict): Tracked hands landmark data.
    """

    # draw landmarks on image to create video with skeletal overlay
    def draw_hand_landmarks_on_image(rgb_image, detection_result):
        """
        Draws pose landmarks on an image.

        Args:
            rgb_image (mp.Image): Mediapipe Image object.
            detection_result (PoseLandmarkerResult): Mediapipe Pose Landmark results.

        Returns:
            annotated_img (ndarray): Array with pose landmarks annotated on image.
        """

        hand_landmark_lst = detection_result.hand_landmarks
        handedness_lst = detection_result.handedness

        annotated_img = np.copy(rgb_image)

        # loop through detected hand landmarks
        for idx in range(len(hand_landmark_lst)):
            hand_landmarks = hand_landmark_lst[idx]
            handedness = handedness_lst[idx]

            # draw hand landmarks
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
            ])

            solutions.drawing_utils.draw_landmarks(
                annotated_img,
                hand_landmarks_proto,
                solutions.hands.HAND_CONNECTIONS,
                solutions.drawing_styles.get_default_hand_landmarks_style(),
                solutions.drawing_styles.get_default_hand_connections_style()
            )

            height, width, _ = annotated_img.shape
            x_coordinates = [landmark.x for landmark in hand_landmarks]
            y_coordinates = [landmark.y for landmark in hand_landmarks]
            text_x = int(min(x_coordinates) * width)
            text_y = int(min(y_coordinates) * height) - 10  # Margin: 10 px

            cv.putText(annotated_img, f'{handedness[0].category_name}', (text_x, text_y),
                       cv.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2, cv.LINE_AA)

        return annotated_img

    # dictionary holding the frame number as key and the hand marker coordinates as values
    marker_data_dict: dict = {}

    # select the options for the landmark object
    base_options = python.BaseOptions(model_asset_path=model_path, delegate=python.BaseOptions.Delegate.GPU)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=n_hands,
        min_hand_detection_confidence=min_hand_detect_conf,
        min_tracking_confidence=min_hand_track_conf
    )

    # create the hand landmark object
    with vision.HandLandmarker.create_from_options(options) as detector:

        # create a video capture object and pass a valid video file path
        cap: cv.VideoCapture = cv.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Error: Could not open video file at {video_path}")
            return marker_data_dict

        # variables required to define the number of leading zeros for frame numbering
        original_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        f_tot = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        f_digits = len(str(f_tot))
        fps = cap.get(cv.CAP_PROP_FPS)
        fnum: int = 0

        # initialize video writer for detected landmarks overlay on video
        out_path = os.path.join(res_path, os.path.splitext(os.path.basename(video_path))[0] + '_hands_tracked.mp4')
        out = cv.VideoWriter(out_path, cv.VideoWriter.fourcc(*'mp4v'), 90.0, (original_width, original_height))

        print("Extracting hand landmarks...")

        # iterate over each frame of the video clip
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("End of video reached.")
                break

            # frame processing
            frame_timestamp_ms = int((fnum / fps) * 1000)

            # converts the current frame to a mediapipe image object
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv.cvtColor(image, cv.COLOR_BGR2RGB))

            # perform hand landmark detection on current frame
            # NOTE: passing the timestamp is essential for VIDEO mode smoothing
            detection_result = detector.detect_for_video(mp_image, frame_timestamp_ms)

            # initialize marker list for 42 landmarks (21 Left, 21 Right)
            marker_lst: list = [(None, None, None)] * 42

            # overlay landmarks on the image
            annotated_image = draw_hand_landmarks_on_image(mp_image.numpy_view(), detection_result)

            # visualize during runtime
            if visualize:
                cv.imshow("Hand Tracking", cv.cvtColor(annotated_image, cv.COLOR_RGB2BGR))
                if cv.waitKey(1) & 0xFF == ord("q"):
                    break

            # if true, extract normalized landmarks
            if normalize:
                hand_landmark_lst = detection_result.hand_landmarks
            else:
                hand_landmark_lst = detection_result.hand_world_landmarks

            # information on handedness corresponding to the lists of hand landmarks: left or right
            handedness_lst = detection_result.handedness

            # iterate over both sides (two lists with 21 landmarks each)
            for idx in range(len(hand_landmark_lst)):
                hand_landmarks = hand_landmark_lst[idx]
                handedness = handedness_lst[idx]

                # determine the base index: 0-20 for left, 21-41 for right
                base_index = 0
                if handedness[0].category_name == 'Right':
                    base_index = 21

                # iterate over each landmark of the current hand
                for i, landmark in enumerate(hand_landmarks):
                    marker_lst[base_index + i] = (landmark.x, landmark.y, landmark.z)

            # store the landmark data of each frame in a dict with the frame number as key)
            base_key: str = str('0' * f_digits)
            key: str = base_key[:-len(str(fnum))] + str(fnum)
            marker_data_dict[key] = marker_lst
            fnum += 1

            # if true, save the tracking overlaid on the original video as separate 'mp4' file
            if save_video:
                # write video to result path
                out.write(cv.cvtColor(annotated_image, cv.COLOR_RGB2BGR))

        cap.release()
        cv.destroyAllWindows()

        return marker_data_dict


# --- 2.3) NORMALIZED LANDMARKS (Legacy Hand Tracker) ------------------------- #
def legacy_hand_landmark_extractor_normalized(video_fpath: str, res_fpath: str) -> dict:

    # dictionary holding the frame number as key and the hand marker coordinates as values
    marker_data_dict: dict = {}

    # open video file
    cap: cv.VideoCapture = cv.VideoCapture(video_fpath)

    # write video file
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    out_path = os.path.join(res_fpath, os.path.basename(video_fpath))
    out = cv.VideoWriter(out_path, cv.VideoWriter.fourcc(*'mp4v'), 90.0, (frame_width, frame_height))

    f_tot = int(cap.get(cv.CAP_PROP_FRAME_COUNT))   # total number of frames in video
    f_digits = len(str(f_tot))                      # total digits used in frame number
    fnum: int = 0                                   # frame number
    n_landmarks: list = []

    # Create hands object
    hands = mp_hands.Hands(
        static_image_mode=False,            # True: batch of unrelated images; False: video stream
        max_num_hands=2,                    # Detect up to 2 hands; any integer > 0
        model_complexity=1,                 # Higher model complexity -> more accurate tracking
        min_detection_confidence=0.5,
        min_tracking_confidence=0.95
    )

    print("Extracting hand landmarks...")
    # iterate over each frame of video clip
    while cap.isOpened():                       # returns true if video capture constructor succeeded initialization
        success, image = cap.read()             # grabs, decodes and returns next video frame
        if not success:                         # returns false at the end of the video or if no frames were found
            print("End of video reached.")
            break

        # marking the image as not writeable to pass by reference.
        image.flags.writeable = False

        # opencv loads images (image frames) in BGR color space -> convert to RGB as mediapipe works with RGB images
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        # extract the face mesh from the image
        results = hands.process(image)

        # 21 landmarks for each hand
        marker_lst: list = [(None, None, None)] * 42

        # check whether any hand is detected
        if results.multi_hand_landmarks:

            # iterate through detected hands and store the corresponding landmarks
            for hand_landmarks, hand_handedness in zip(results.multi_hand_landmarks, results.multi_handedness):

                # Get the handedness label ('Left' or 'Right')
                hand_label = hand_handedness.classification[0].label

                # Determine the starting index in marker_lst based on handedness
                if hand_label == 'Left':
                    start_idx: int = 0
                elif hand_label == 'Right':
                    start_idx: int = 21
                else:
                    # Should not happen with typical MediaPipe output
                    continue

                for i, landmark in enumerate(hand_landmarks.landmark):
                    #print(f"Landmark {start_idx + i}: x={landmark.x:.3f}, y={landmark.y:.3f}, z={landmark.z:.3f}")
                    marker_lst[start_idx + i] = (landmark.x, landmark.y, landmark.z)

                # Optionally, draw landmarks on the frame
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # the dictionary keys are strings that correspond to the frame number filled with leading zeros
            base_key: str = str('0' * f_digits)
            key: str = base_key[:-len(str(fnum))] + str(fnum)

            # store landmark in dictionary
            marker_data_dict[key] = marker_lst

            fnum += 1

        # write frame to video file
        #if not os.path.exists(out_path):
        out.write(cv.cvtColor(image, cv.COLOR_RGB2BGR))

        # Show frame with drawings (for debugging)
        cv.imshow("Hand Tracking", cv.cvtColor(image, cv.COLOR_RGB2BGR))
        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv.destroyAllWindows()

    return marker_data_dict


#################################################################################
###                              3) HOLISTIC TRACKER                          ###
#################################################################################
# --- 3) NORMALIZED LANDMARKS (Legacy Hand Tracker) --------------------------- #
def legacy_holistic_landmark_extractor(video_fpath: str, res_fpath: str,
                                       drawing: bool = True) -> tuple:

    def draw_holistic_landmarks(img, res):
        # 1) Draw face landmarks
        mp_drawing.draw_landmarks(
            img,
            res.face_landmarks,
            mp_holistic.FACEMESH_TESSELATION,
            mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
        )

        # 3a) Draw left hand landmarks
        mp_drawing.draw_landmarks(
            img,
            res.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
        )

        # 3b) Draw right hand landmarks
        mp_drawing.draw_landmarks(
            img,
            res.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
        )

        # 4) Draw pose landmarks
        mp_drawing.draw_landmarks(
            img,
            res.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 80, 66), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
        )

    # dictionary holding the frame number as key and the marker coordinates as values
    pose_marker_data_dict: dict = {}
    hand_marker_data_dict: dict = {}
    face_marker_data_dict: dict = {}

    # open video file
    cap: cv.VideoCapture = cv.VideoCapture(video_fpath)

    # write video file
    frame_width: int = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height: int = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    video_name: str = os.path.splitext(os.path.basename(video_fpath))[0] + '_holistic_overlay.mp4'
    out_path: str = os.path.join(res_fpath, video_name)
    fps = float(cap.get(cv.CAP_PROP_FPS))
    out: cv.VideoWriter = cv.VideoWriter(out_path, cv.VideoWriter.fourcc(*'mp4v'), fps, (frame_width, frame_height))

    if not out.isOpened():
        print("FATAL ERROR: cv.VideoWriter failed to initialize. Check the codec (FOURCC) and file path.")

    f_tot: int = int(cap.get(cv.CAP_PROP_FRAME_COUNT))  # total number of frames in video
    f_digits: int = len(str(f_tot))         # total digits used in frame number
    fnum: int = 0                           # frame number

    # Create holistic object
    holistic = mp_holistic.Holistic(
        static_image_mode=False,            # True: batch of unrelated images; False: video stream
        model_complexity=2,                 # Higher model complexity -> more accurate tracking
        smooth_landmarks=True,
        enable_segmentation=True,
        refine_face_landmarks=True,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.9
    )

    print("Extracting and overlay holistic landmarks...")
    # iterate over each frame of video clip
    while cap.isOpened():                   # returns true if video capture constructor succeeded initialization
        success, image = cap.read()         # grabs, decodes and returns next video frame
        if not success:                     # returns false at the end of the video or if no frames were found
            print("End of video reached.")
            break

        # marking the image as not writeable to pass by reference.
        image.flags.writeable = False

        # convert image to RGB (required by MediaPipe)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        # extract landmarks
        results = holistic.process(image)

        # mark the image as writable and convert it back to BGR for drawing with opencv
        image.flags.writeable = True
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

        if drawing:
            draw_holistic_landmarks(image, results)

        # 1) Store the pose coordinates (x,y,z) of the current frame in a list
        pose_marker_lst: list = [(None, None, None)] * 33
        if results.pose_landmarks:
            for i, landmark in enumerate(results.pose_landmarks.landmark):
                # print(f"Landmark {i}: x={landmark.x:.3f}, y={landmark.y:.3f}, z={landmark.z:.3f}")
                pose_marker_lst[i] = (landmark.x, landmark.y, landmark.z)

        # 2) Store the hand coordinates (x,y,z) of the current frame in a list
        hand_marker_lst: list = [(None, None, None)] * 42
        if results.left_hand_landmarks:
            for i, landmark in enumerate(results.left_hand_landmarks.landmark):
                hand_marker_lst[i] = (landmark.x, landmark.y, landmark.z)
        if results.right_hand_landmarks:
            for i, landmark in enumerate(results.right_hand_landmarks.landmark):
                hand_marker_lst[21 + i] = (landmark.x, landmark.y, landmark.z)

        face_marker_lst: list = [(None, None, None)] * 478
        if results.face_landmarks:
            for i, landmark in enumerate(results.face_landmarks.landmark):
                face_marker_lst[i] = (landmark.x, landmark.y, landmark.z)

        # the dictionary keys are strings that correspond to the frame number filled with leading zeros
        base_key: str = str('0' * f_digits)
        key: str = base_key[:-len(str(fnum))] + str(fnum)

        # store landmark in dictionary
        pose_marker_data_dict[key] = pose_marker_lst
        hand_marker_data_dict[key] = hand_marker_lst
        face_marker_data_dict[key] = face_marker_lst

        fnum += 1

        # write frame to video file
        out.write(image)

        # Show frame with drawings
        cv.imshow("Holistic Tracking", image)
        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()           # release the VideoCapture object
    out.release()           # release the VideoWriter object
    cv.destroyAllWindows()

    print(f'Saved holistic tracking data to: {out_path}')

    return pose_marker_data_dict, hand_marker_data_dict, face_marker_data_dict
