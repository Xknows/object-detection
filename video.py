import cv2

# copy class names into a array
classNames = []
classFile = "coco.names"
with open(classFile, "rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

configPath = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightPath = "frozen_inference_graph.pb"


# feed neural network
net = cv2.dnn.DetectionModel(weightPath, configPath)
net.setInputSize(320, 320)
# these parameters of neural network are adjustable and should play around with them
net.setInputScale(1/255)
net.setInputMean(100)
# convert RGB to BGR = true
net.setInputSwapRB(True)

# read from camera(webcam in this case)
cap = cv2.VideoCapture(0)
print("running")
# threshold
min_confidence = 0.45

while True:
    # return frames
    _, image = cap.read()
    # if the detection was successful, the output will be as follows;
    classIds, confs, bbox = net.detect(image, confThreshold=min_confidence)

    if len(classIds) == True:
        # rectangle and text parameters
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            cv2.rectangle(image, box, color=(0, 0, 255), thickness=2)
            cv2.putText(image, classNames[classId-1].upper(), (box[0]+10, box[1]-10),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(image, str(round(confidence*100, 2)), (box[0]+10, box[1]+30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Output", image)
    # for quit press Q
    if cv2.waitKey(1) & 0XFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()
