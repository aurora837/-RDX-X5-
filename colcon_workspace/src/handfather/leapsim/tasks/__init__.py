

from .leap_hand_rot_pack import LeapHandRot
from .leap_hand_grasp import LeapHandGrasp

# Mappings from strings to environments
isaacgym_task_map = {
    "LeapHandGrasp": LeapHandGrasp,
    "LeapHandRot": LeapHandRot,
}
