# src/core.py

# libraries
import os
import pickle
from dataclasses import dataclass, field


# container for recordings
@dataclass
class Exercise:
    """
    The Exercise class is a container for holding exercise information and storing data (tracking and metrics).
    """
    # exercise info
    visit_id: str                                               # 'T1', 'T2', 'T3'
    exercise_id: str                                            # 'FingerTapping', 'HandOpening', etc.
    side_condition: str                                         # 'Healthy' or 'Affected'
    side_focus: str                                             # 'L' or 'R'

    # metadata
    cam_id: str                                                 # e.g., 'camZ'

    # data storage
    raw_pose_landmarks: dict                                    # tracked pose landmarks raw
    raw_hand_landmarks: dict                                    # tracked hand landmarks raw
    clean_pose_landmarks: dict = field(default_factory=dict)    # preprocessed pose data (world: meters, normalized: px)
    clean_hand_landmarks: dict = field(default_factory=dict)    # preprocessed hand data (world: meters, normalized: px)

    metrics: dict = field(default_factory=dict)                 # stores results

    # additional attributes
    left_hand_size: float = 0.0                                 # median hand size 'left' across all exercises
    right_hand_size: float = 0.0                                # median hand size 'right' across all exercises


class Participant:
    """
    The Participant class represents one visit of a participant.
    - participant ID
    - visit ID
    - affected side of participant (R or L)
    """
    def __init__(self, pid: str, visit_id: str, affected_side: str):
        self.pid = pid
        self.visit_id = visit_id
        self.affected_side = affected_side

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
