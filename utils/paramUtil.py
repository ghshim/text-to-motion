import numpy as np

# Define a kinematic tree for the skeletal struture
# kit_kinematic_chain = [[0, 11, 12, 13, 14, 15], [0, 16, 17, 18, 19, 20], [0, 1, 2, 3, 4], [3, 5, 6, 7], [3, 8, 9, 10]]

# kit_raw_offsets = np.array(
#     [
#         [0, 0, 0],
#         [0, 1, 0],
#         [0, 1, 0],
#         [0, 1, 0],
#         [0, 1, 0],
#         [1, 0, 0],
#         [0, -1, 0],
#         [0, -1, 0],
#         [-1, 0, 0],
#         [0, -1, 0],
#         [0, -1, 0],
#         [1, 0, 0],
#         [0, -1, 0],
#         [0, -1, 0],
#         [0, 0, 1],
#         [0, 0, 1],
#         [-1, 0, 0],
#         [0, -1, 0],
#         [0, -1, 0],
#         [0, 0, 1],
#         [0, 0, 1]
#     ]
# )

# Lower legs
l_idx1, l_idx2 = 5, 8
# Right/Left foot
fid_r, fid_l = [8, 11], [7, 10]
# Face direction, r_hip, l_hip, sdr_r, sdr_l
face_joint_indx = [2, 1, 17, 16]
# l_hip, r_hip
r_hip, l_hip = 2, 1

t2m_raw_offsets = np.array([[0,0,0],
                           [1,0,0],
                           [-1,0,0],
                           [0,1,0],
                           [0,-1,0],
                           [0,-1,0],
                           [0,1,0],
                           [0,-1,0],
                           [0,-1,0],
                           [0,1,0],
                           [0,0,1],
                           [0,0,1],
                           [0,1,0],
                           [1,0,0],
                           [-1,0,0],
                           [0,0,1],
                           [0,-1,0],
                           [0,-1,0],
                           [0,-1,0],
                           [0,-1,0],
                           [0,-1,0],
                           [0,-1,0]])

t2m_kinematic_chain = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]
t2m_left_hand_chain = [[20, 22, 23, 24], [20, 34, 35, 36], [20, 25, 26, 27], [20, 31, 32, 33], [20, 28, 29, 30]]
t2m_right_hand_chain = [[21, 43, 44, 45], [21, 46, 47, 48], [21, 40, 41, 42], [21, 37, 38, 39], [21, 49, 50, 51]]

# kit_tgt_skel_id = '03950'
t2m_tgt_skel_id = '000021'

import numpy as np

JOINT_NAMES = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
]

JOINT_HIERARHCY = {
    0: [1, 2, 3],
    1: [4],
    4: [7],
    7: [10],
    2: [5],
    5: [8],
    8: [11],
    3: [6],
    6: [9],
    9: [12, 13, 14],
    12: [15],
    13: [16],
    16: [18],
    18: [20],
    14: [17],
    17: [19],
    19: [21]
}
EPSILON = np.finfo(float).eps

# from https://github.com/facebookresearch/fairmotion/blob/main/fairmotion/utils/constants.py
EYE_T = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    float,
)

NUM_JOINTS = 22

SKELETON = [
    (0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6), 
    (4, 7), (5, 8), (6, 9), (7, 10), (8, 11), (9, 12), 
    (9, 13), (9, 14), (12, 15), (13, 16), (14, 17), 
    (16, 18), (17, 19), (18, 20), (19, 21)
]