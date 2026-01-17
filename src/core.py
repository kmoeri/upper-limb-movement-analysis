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
    visit_id: str                                       # 'T1', 'T2', 'T3'
    exercise_id: str                                    # 'FingerTapping', 'HandOpening', etc.
    side_focus: str                                     # 'L' or 'R'
    side_condition: str                                 # 'Healthy' or 'Affected'

    # data storage
    raw_landmarks: dict                                 # normalized data from MediaPipe (0.0 - 1.0)
    px_landmarks: dict                                  # aspect-ratio corrected data (pixels)
    metrics: dict = field(default_factory=dict)         # stores results


class Participant:
    """
    The Participant class represents one visit of a participant.
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
        filepath = os.path.join(dest_dir, f'{self.pid}_{self.visit_id}.pkl')
        with open(filepath, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load(filepath: str):
        with open(filepath, 'rb') as file:
            return pickle.load(file)
