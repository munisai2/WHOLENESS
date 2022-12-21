import cv2
import mediapipe as mp
import numpy as np
import virtual_assistance as va

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def calculate(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180 / np.pi)
    if angle > 180:
        angle = 360 - angle
    return angle


cap = cv2.VideoCapture(0)

# curl counter variable
counter = 0
stage = None
positive = ['ok', 'yes', 'yeah', 'sure', 'obviously', 'easily', 'continue', 'yess']
negative = ['stop', 'no', 'enough', 'wait', "can't do any more", 'na']
# setuping mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    va.talk(3)
    va.talk(2)
    va.talk(1)
    va.talk('start')
    while cap.isOpened():
        ret, frame = cap.read()
        # recolouring image to rgb
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.flip(image, 1)
        image.flags.writeable = False

        result = pose.process(image)

        # recoloring back to bgr
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


        # print(result)
        # print(result.pose_landmarks)
        # print(mp_pose.POSE_CONNECTIONS)

        # extract landmarks
        try:
            landmarks = result.pose_landmarks.landmark
            # print(len(landmarks))

            # for ln in mp_pose.PoseLandmark:
            #     print(ln)
            print(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value])

            # left points

            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]

            # right points
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]

            # calculating angle
            angle_left_shoulder = calculate(left_hip, left_shoulder, left_elbow)
            angle_left_elbow = calculate(left_shoulder, left_elbow, left_wrist)
            angle_left_hip = calculate(left_shoulder, left_hip, left_knee)

            angle_right_shoulder = calculate(right_hip, right_shoulder, right_elbow)
            angle_right_elbow = calculate(right_shoulder, right_elbow, right_wrist)
            angle_right_hip = calculate(right_shoulder, right_hip, right_knee)

            # asigning angle with respect to cam size

            tuple(np.multiply(left_elbow, [640, 480]).astype(int))
            tuple(np.multiply(left_shoulder, [640, 480]).astype(int))
            tuple(np.multiply(left_hip, [640, 480]).astype(int))

            tuple(np.multiply(right_hip, [640, 480]).astype(int))
            tuple(np.multiply(right_elbow, [640, 480]).astype(int))
            tuple(np.multiply(right_shoulder, [640, 480]).astype(int))

            # visualizing angle

            # cv2.putText(image,str(angle_left_elbow),
            #             tuple(np.multiply(left_elbow ,[640,480]).astype(int)),
            #             cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2,cv2.LINE_AA)

            cv2.putText(image, str(angle_left_shoulder),
                        tuple(np.multiply(left_shoulder, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.putText(image, str(angle_left_hip),
                        tuple(np.multiply(left_hip, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.putText(image, str(angle_right_hip),
                        tuple(np.multiply(right_hip, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

            # cv2.putText(image,str(angle_right_elbow),
            #             tuple(np.multiply(right_elbow ,[640,480]).astype(int)),
            #             cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2,cv2.LINE_AA)

            cv2.putText(image, str(angle_right_shoulder),
                        tuple(np.multiply(right_shoulder, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

            if angle_left_hip > 150 and angle_right_hip > 150:

                if angle_left_elbow > 160 and angle_right_elbow > 160:
                    stage = "Down"
                    va.talk(stage)
                if angle_left_elbow < 95 and angle_right_elbow < 95 and stage == "Down":
                    stage = 'up'
                    va.talk(stage)
                    counter += 1
                    print(counter)

                if counter == 10:
                    va.talk("come on you can do it, just 5 more")

                # if counter >10:
                # va.talk(counter)

                if counter == 15:
                    va.talk("nice job, you have reached your goal")
                    va.talk("do you want to continue")
                    cmd = va.take_command()
                    print(cmd)
                if cmd in positive:
                    if counter >= 15 and counter <= 20:
                        va.talk(counter)
                        va.talk("great efforts")

                    if counter >= 20:
                        va.talk("great job. you had burn 8 calories")
                        break
                if cmd in negative:
                    calories = counter * 0.4
                    va.talk('you burn ', {calories}, "calories")
                    break
            elif angle_left_elbow < 80 and angle_right_elbow < 80:
                va.talk("come on! you can, lets finish this task")

            elif angle_left_hip < 150 and angle_right_hip < 150:
                va.talk("lower your hip")


        except:
            pass

        # setup status box (bgr)
        cv2.rectangle(image, (0, 0), (225, 73), (120, 112, 260), -1)

        # rep data
        cv2.putText(image, 'REPS', (15, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, str(counter),
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # stage data

        cv2.putText(image, 'STAGE', (90, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, stage,
                    (80, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # render detections (joints and edges)
        mp_drawing.draw_landmarks(image, result.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=1),
                                  mp_drawing.DrawingSpec(color=(214, 118, 49), thickness=2, circle_radius=2))

        cv2.imshow('wholeness', image)
        # cv2.imshow('wholeness', cv2.flip(image, 1))

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

