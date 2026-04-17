# src/core.py

# libraries
import os
import pickle
import numpy as np
import pandas as pd
from dataclasses import dataclass, field


# container for recordings
@dataclass
class Exercise:
    """
    The Exercise class is a metadata container for exercise information.
    Kinematic data is stored as Parquet and loaded as DataFrames or NumPy tensors.
    """
    # exercise information
    visit_id: str                                   # 'T1', 'T2', 'T3'
    exercise_id: str                                # 'FingerTapping', 'HandOpening', etc.
    side_condition: str                             # 'Healthy' or 'Affected'
    side_focus: str                                 # 'L' or 'R'

    # metadata
    cam_id: str                                     # e.g., 'camZ'

    # data storage
    raw_landmark_data_path: str = ''                # e.g., 'data/02_tracking_data/...P001_T1_WT-01_camZ.parquet'
    clean_landmark_data_path: str = ''              # e.g., 'data/02_tracking_data/...P001_T1_WT-01_camZ_clean.parquet'

    metrics: dict = field(default_factory=dict)     # stores results

    def load_dataframe(self, stage: str = 'raw') -> pd.DataFrame:
        file_path: str = self.raw_landmark_data_path if stage == 'raw' else self.clean_landmark_data_path

        if not file_path or not os.path.exists(file_path):
            raise FileNotFoundError(f'Data for stage "{stage}" not found at {file_path}.')

        return pd.read_parquet(file_path)

    def save_dataframe(self, df: pd.DataFrame, stage: str = 'clean') -> None:
        file_path: str = self.raw_landmark_data_path if stage == 'raw' else self.clean_landmark_data_path

        if not file_path:
            raise ValueError(f'Save aborted: Data for stage "{stage}" has not been set.')

        df.to_parquet(file_path, engine='pyarrow')

    def get_coord_tensor(self, stage: str = 'clean') -> np.ndarray:
        df = self.load_dataframe(stage)

        coord_cols: list[str] = [col for col in df.columns if col.endswith(('_x', '_y', '_z'))]

        n_frames: int = len(df)
        n_joints: int = len(coord_cols) // 3

        return df[coord_cols].to_numpy().reshape(n_frames, n_joints, 3)

    def get_rot_tensor(self, stage: str = 'clean') -> np.ndarray:
        df = self.load_dataframe(stage)

        rot_cols: list[str] = [col for col in df.columns if col.endswith('_rot')]

        n_frames: int = len(df)
        n_joints: int = len(rot_cols)

        # flatten 9 element array and reshape to 3x3 matrices
        stacked_rots = np.vstack(df[rot_cols].to_numpy().flatten())
        return stacked_rots.reshape(n_frames, n_joints, 3, 3)


class Participant:
    """
    The Participant class represents one visit of a participant.
    - participant ID
    - visit ID
    - affected side of participant (R or L)
    """
    def __init__(self, pid: str, visit_id: str, affected_side: str):
        self.pid = pid                                  # participant identifier ('P001', 'P002', ...)
        self.visit_id = visit_id                        # visit identifier ('T1', 'T2', 'T3')
        self.affected_side = affected_side              # 'Healthy' or 'Affected'

        # additional attributes
        self.left_hand_size: float = 0.0                # median hand size 'left' across all exercises
        self.right_hand_size: float = 0.0               # median hand size 'right' across all exercises

        # storage: e.g., "FingerTapping_Affected"
        self.exercises: dict[str, Exercise] = {}

    def add_exercise(self, exercise: Exercise):
        if exercise.visit_id != self.visit_id:
            raise ValueError(f'Mismatch: trying to add session {exercise.visit_id} to visit {self.visit_id}.')

        # e.g., key: "FingerTapping_Affected"
        key = f'{exercise.exercise_id}_{exercise.side_condition}'
        self.exercises[key] = exercise

    def get_paired_exercises(self, exercise_id: str) -> tuple:
        healthy = self.exercises.get(f'{exercise_id}_Healthy')
        affected = self.exercises.get(f'{exercise_id}_Affected')
        return healthy, affected

    def save(self, dest_dir: str):
        # e.g.: "P001_T1.pkl"
        filepath = os.path.join(dest_dir, f'{self.pid}_{self.visit_id}.pickle')
        with open(filepath, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load(filepath: str):
        with open(filepath, 'rb') as file:
            return pickle.load(file)
