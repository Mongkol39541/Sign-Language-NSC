import cv2
import mediapipe as mp
import numpy as np
import time

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             )
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             )
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             )
    
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, lh, rh])

def play_video_lstm(name_video, model, actions):
  sequence = []
  process_time = []
  num = 0
  acu = []
  class_name = []
  with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    cap = cv2.VideoCapture("static/videos/{0}".format(name_video))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vdo_writer = cv2.VideoWriter('static/videos/{0}'.format("LSTM" + name_video), fourcc, 30.0, (800, 480))
    while True:
      ret, frame = cap.read()
      if ret:
        frame = cv2.resize(frame, (800, 480))
        image, results = mediapipe_detection(frame, holistic)
        draw_styled_landmarks(image, results)
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]
        if len(sequence) == 30:
          process_time = []
          num += 1
          start_time = time.time()
          res = model.predict(np.expand_dims(sequence, axis=0))[0]
          end_time = time.time()
          elapsed_time = end_time - start_time
          process_time.append(elapsed_time)
          name = actions[np.argmax(res)]
          cal = res[np.argmax(res)] * 100
          acu.append(cal)
          if cal <= 25:
            name = "Do not know"
          cv2.putText(image, name, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 2)
          cv2.putText(image, str('%.2f' %(cal)) + " %", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
          class_name.append(name)
          vdo_writer.write(image)
        else:
          vdo_writer.write(frame)
      if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
        break
  vdo_writer.release()
  cap.release()
  cv2.destroyAllWindows()
  class_name = np.array(class_name)
  unique_values, counts = np.unique(class_name, return_counts=True)
  most_common_index = np.argmax(counts)
  return '%.4f' %(sum(process_time) / num), '%.4f' %(sum(acu) / num), unique_values[most_common_index]