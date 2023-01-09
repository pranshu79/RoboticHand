import cv2 as cv
import mediapipe as mp
import math
import tensorflow as tf

def print_pred(frame, pred, bools):
    val = pred.argmax(axis=1)[0]
    prob = pred[0][val]
    print(val, prob)
    if bools==[0,0,0,0,0] or bools==[1,0,0,0,0] and val==11 and prob>0.99:
        bools[0]=1
        frame = cv.putText(frame, "START", (50,70), cv.FONT_HERSHEY_SIMPLEX, 2, (200,200,200), 3, cv.LINE_AA)
    elif bools[0]==1 and val==2 or val==16 and prob>0.9:
        bools[1]=1
        frame = cv.putText(frame, "I", (50,70), cv.FONT_HERSHEY_SIMPLEX, 2, (150,0,0), 3, cv.LINE_AA)
    elif bools[0]==1 and val==14 and prob>0.9:
        bools[2]=1
        frame = cv.putText(frame, "LIKE", (50,70), cv.FONT_HERSHEY_SIMPLEX, 2, (150,0,0), 3, cv.LINE_AA)
    elif bools[0]==1 and val==3 and prob>0.9:
        bools[3]=1
        frame = cv.putText(frame, "YOU", (50,70), cv.FONT_HERSHEY_SIMPLEX, 2, (150,0,0), 3, cv.LINE_AA)
    elif bools==[1,1,1,1,0] and val==11 and prob>0.99:
        bools[-1]=1

cap = cv.VideoCapture(0)
hands = mp.solutions.hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5)
model = tf.keras.models.load_model('model.h5')
bools = [0,0,0,0,0]
while True:
    success, frame = cap.read()
    results = hands.process(frame)
    datapoint = []
    if results.multi_hand_landmarks:
        for hand_landmark in results.multi_hand_landmarks:
            idx_to_coordinates = {}
            for idx, landmark in enumerate(hand_landmark.landmark):
                datapoint.append(landmark.x)
                datapoint.append(landmark.y)
                x = min(math.floor(landmark.x * frame.shape[1]), frame.shape[1] - 1)
                y = min(math.floor(landmark.y * frame.shape[0]), frame.shape[0] - 1)
                idx_to_coordinates[idx] = x, y
                cv.circle(frame, (x,y), 4, (0,0,250), -1)
            for connection in mp.solutions.hands.HAND_CONNECTIONS:
                start_idx = connection[0]
                end_idx = connection[1]
                if start_idx in idx_to_coordinates and end_idx in idx_to_coordinates:
                    cv.line(frame, idx_to_coordinates[start_idx], idx_to_coordinates[end_idx], (0,250,0), 2)
    frame = cv.flip(frame, 1)
    if bools[-1]==1:
        frame = cv.putText(frame, "END", (50,70), cv.FONT_HERSHEY_SIMPLEX, 2, (200,200,200), 3, cv.LINE_AA)
    elif datapoint:
        datapoint=datapoint[:21]
        pred = model.predict([datapoint])
        print_pred(frame, pred, bools)
    cv.imshow('window', frame)
    c = cv.waitKey(1)
    if c==ord('e'):
        bools=[0,0,0,0,0]
    elif c==ord('q'):
        break

cap.release()
cv.destroyAllWindows()