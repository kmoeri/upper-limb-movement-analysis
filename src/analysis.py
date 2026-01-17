
# modules
import src.utils as utils
from src.config import config
from src.core import Participant


class KinematicAnalyser:
    def __init__(self):
        self.utils = utils.ToolBox(config['camera_param']['width'], config['camera_param']['height'],
                                   config['camera_param']['fps'])

    def process_participant(self, participant: Participant):
        pass

    def _analyze_finger_tapping(self, exercise):
        pass

    def _analyze_finger_alternation(self, exercise):
        pass

    def _analyze_hand_opening(self, exercise):
        pass

    def _analyze_pronation_supination(self, exercise):
        pass

    def _analyze_table_tapping(self, exercise):
        pass
