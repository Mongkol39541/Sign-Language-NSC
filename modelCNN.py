import cv2
import numpy as np
import time

def play_video_cnn(name_video, model, actions):
    cap = cv2.VideoCapture("static/videos/{0}".format(name_video))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vdo_writer = cv2.VideoWriter('static/videos/{0}'.format("CNN" + name_video), fourcc, 30.0, (800, 480))
    process_time = []
    num = 0
    acu = []
    class_name = []
    while True:
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (800, 480))
            num += 1
            start_time = time.time()
            results = model(frame)
            end_time = time.time()
            elapsed_time = end_time - start_time
            process_time.append(elapsed_time)
            boxes = results.xyxy[0][:, :4].cpu().numpy()
            classes = results.pred[0][:, -1].cpu().numpy()
            scores = results.pred[0][:, 4].cpu().numpy()
            try:
                boxe = boxes[0]
                cv2.rectangle(frame, (int(boxe[0]), int(boxe[1])), (int(boxe[2]), int(boxe[3])), (0, 255, 0), 2)
                cv2.putText(frame, actions[int(classes[0])], (int(boxe[0]) + 50, int(boxe[1]) + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, str('%.2f' %(scores[0] * 100)) + " %", (int(boxe[0]) + 50, int(boxe[1]) + 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                class_name.append(actions[int(classes[0])])
                acu.append(scores[0] * 100)
            except:
                pass
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