import tensorflow as tf
import numpy as np
import cv2
import virtual_assistance as va
from matplotlib import pyplot as plt

import Pose_Draw
import Pose_Draw as pd

# calculate= Pose_Draw.calculate

interpreter = tf.lite.Interpreter(model_path="thundermodel.tflite")
interpreter.allocate_tensors()
cap = cv2.VideoCapture(0)

stage = None
counter = 0

while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    # reshape image
    img = frame.copy()
    img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 256, 256)
    input_image = tf.cast(img, dtype=tf.float32)

    # setup input and output
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # make predictions
    interpreter.set_tensor(input_details[0]['index'], np.array(input_image))

    # Invoke inference.
    interpreter.invoke()
    # Get the model prediction.
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
    print(keypoints_with_scores)

    # elbow
    # left side
    left_shoulder = keypoints_with_scores[0][0][6][0:2]
    left_elbow = keypoints_with_scores[0][0][8][0:2]
    left_wrist = keypoints_with_scores[0][0][10][0:2]

    left_hip = keypoints_with_scores[0][0][12][0:2]
    left_knee = keypoints_with_scores[0][0][14][0:2]
    left_ankle = keypoints_with_scores[0][0][16][0:2]

    # right side
    right_shoulder = keypoints_with_scores[0][0][5][0:2]
    right_elbow = keypoints_with_scores[0][0][7][0:2]
    right_wrist = keypoints_with_scores   [0][0][9][0:2]

    right_hip = keypoints_with_scores[0][0][11][0:2]
    right_knee = keypoints_with_scores[0][0][13][0:2]
    right_ankle = keypoints_with_scores[0][0][15][0:2]

    edges = pd.KEYPOINT_EDGE_INDS_TO_COLOR
    slope = pd.draw_connections(frame, keypoints_with_scores, edges, 0.3)

    pd.draw_keypoints(frame, keypoints_with_scores, 0.4)
    y, x, c = frame.shape

    # print((np.multiply(left_shoulder, [y,x])).astype(int),(np.multiply(left_elbow, [y,x])).astype(int),(np.multiply(left_wrist, [y,x])).astype(int))

    # angle calculation
    # elbow
    angle_left_elbow = pd.calculate(left_shoulder, left_elbow, left_wrist)
    angle_right_elbow = pd.calculate(right_shoulder, right_elbow, right_wrist)
    # shoulder
    angle_left_shoulder = pd.calculate(left_elbow, left_shoulder, left_hip)
    angle_right_shoulder = pd.calculate(right_elbow, right_shoulder, right_hip)
    # hip
    angle_left_hip = pd.calculate(left_shoulder, left_hip, left_knee)
    angle_right_hip = pd.calculate(right_shoulder, right_hip, right_knee)
    # knee
    angle_left_knee = pd.calculate(left_hip, left_knee, left_ankle)
    angle_right_knee = pd.calculate(right_hip, right_knee, right_ankle)

    # angle datainto the screeen
    y, x, c = frame.shape
    # elbows
    cv2.putText(frame, str(angle_left_elbow), tuple(reversed(np.multiply(left_elbow, [y, x]).astype(int))),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (29, 5, 245), 1, cv2.LINE_AA)
    # cv2.putText(frame, str(angle_right_elbow), tuple(reversed(np.multiply(right_elbow, [y, x]).astype(int))),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (29, 5, 245), 1, cv2.LINE_AA)
    # shoulders
    cv2.putText(frame, str(angle_left_shoulder), tuple(reversed(np.multiply(left_shoulder, [y, x]).astype(int))),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (29, 5, 245), 1, cv2.LINE_AA)
    # cv2.putText(frame, str(angle_right_shoulder), tuple(reversed(np.multiply(right_shoulder, [y, x]).astype(int))),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (29, 5, 245), 1, cv2.LINE_AA)
    # hips
    cv2.putText(frame, str(angle_left_hip), tuple(reversed(np.multiply(left_hip, [y, x]).astype(int))),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (29, 5, 245), 1, cv2.LINE_AA)
    # cv2.putText(frame, str(angle_right_hip), tuple(reversed(np.multiply(right_hip, [y, x]).astype(int))),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (29, 5, 245), 1, cv2.LINE_AA)
    # knees
    cv2.putText(frame, str(angle_left_knee), tuple(reversed(np.multiply(left_knee, [y, x]).astype(int))),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (29, 5, 245), 1, cv2.LINE_AA)
    # cv2.putText(frame, str(angle_right_knee), tuple(reversed(np.multiply(right_knee, [y, x]).astype(int))),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (29, 5, 245), 1, cv2.LINE_AA)

    if (angle_left_elbow >150 and angle_right_elbow > 150) and (angle_left_shoulder>70 and angle_right_shoulder > 70):
        if (angle_left_hip > 150 and angle_right_hip > 150) and (angle_left_knee > 150 and angle_right_knee > 150):
            stage = "Down"
            va.talk(stage)

        if (angle_left_hip < 70 and angle_right_hip < 70) and (angle_left_knee < 70 and angle_right_knee < 70):
            stage = "Up"
            va.talk(stage)
            counter+=1

        if counter==10:

            va.talk("only 5 left")

        if counter==15:

            va.talk("Nice Job, you have reached your goal")
    else:

        va.talk("Keep your arms straight")

    cv2.rectangle(frame, (0, 0), (225, 73), (0, 0, 0), -1)

    # rep data
    cv2.putText(frame, 'REPS', (15, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, str(counter),
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, .75, (229, 172, 81), 2, cv2.LINE_AA)

    # stage data

    cv2.putText(frame, 'STAGE', (90, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, stage,
                (80, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (229, 172, 81), 2, cv2.LINE_AA)

    cv2.imshow("pose estimation", frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
