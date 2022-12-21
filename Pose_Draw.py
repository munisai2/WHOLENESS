import cv2
import numpy as np


# Dictionary that maps from joint names to keypoint indices.
KEYPOINT_DICT = {
    'nose': 0,
    'right_eye': 1,
    'left_eye': 2,
    'right_ear': 3,
    'left_ear': 4,
    'right_shoulder': 5,
    'left_shoulder': 6,
    'right_elbow': 7,
    'left_elbow': 8,
    'right_wrist': 9,
    'left_wrist': 10,
    'right_hip': 11,
    'left_hip': 12,
    'right_knee': 13,
    'left_knee': 14,
    'right_ankle': 15,
    'left_ankle': 16
}

# Maps bones to a matplotlib color name.
KEYPOINT_EDGE_INDS_TO_COLOR = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}

def draw_keypoints(frame,keypoints,confidence_threshold=0.3):
  y,x,c= frame.shape
  shaped = np.squeeze(np.multiply(keypoints,[y,x,1]))

  for kp in shaped:
    ky,kx,kp_conf =kp
    if kp_conf>confidence_threshold:
      cv2.circle(frame,(int(kx),int(ky)),4,(0,0,0),-1)
      print(int(kx),int(ky))


def draw_connections(frame,keypoints,edges,confidence_threshold=0.3):
    slope=0
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))


    for edge,color in edges.items():
        p1,p2=edge

        y1,x1,c1 = shaped[p1]
        y2,x2,c2 =shaped[p2]


        if c1>confidence_threshold and c2>confidence_threshold:
            slope= (y2-y1)/(x2-x1)
            print(slope)
            cv2.line(frame,(int(x1),int(y1)),(int(x2),int(y2)),(214, 118, 49),2)
    return slope







def calculate(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180 / np.pi)
    if angle > 180:
        angle = 360 - angle
    return round(angle,2)