"""Run an object detection model on a live feed."""
from pathlib import Path

import cv2

DATA_DIR = Path(__file__).parent / "Object_Detection_Files"

with open(DATA_DIR / "coco.names", "rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

configPath = DATA_DIR / "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = DATA_DIR / "frozen_inference_graph.pb"

# This is some set up values to get good results
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


# This is to set up what the drawn box size/colour is and the font/size/colour
# of the name tag and confidence label
def getObjects(img, thres, nms, draw=True, objects=[]):
    """Detect objects in an image and draw them."""
    classIds, confs, bbox = net.detect(
        img, confThreshold=thres, nmsThreshold=nms
    )
    # Below has been commented out, if you want to print each sighting of an
    # object to the console you can uncomment below print(classIds,bbox)
    if len(objects) == 0:
        objects = classNames

    if len(classIds) == 0:
        return img, []

    objectInfo = []
    for classId, confidence, box in zip(
        classIds.flatten(), confs.flatten(), bbox
    ):
        className = classNames[classId - 1]
        if className not in objects:
            continue

        objectInfo.append([box, className])
        if not draw:
            continue

        cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
        cv2.putText(
            img,
            classNames[classId - 1].upper(),
            (box[0] + 10, box[1] + 30),
            cv2.FONT_HERSHEY_COMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            img,
            str(round(confidence * 100, 2)),
            (box[0] + 200, box[1] + 30),
            cv2.FONT_HERSHEY_COMPLEX,
            1,
            (0, 255, 0),
            2,
        )

    return img, objectInfo


# Below determines the size of the live feed window that will be displayed on
# the Raspberry Pi OS
if __name__ == "__main__":
    cap = cv2.VideoCapture(4)
    cap.set(3, 640)
    cap.set(4, 480)
    # cap.set(10,70)

    window_name = "projector"
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(
        window_name,
        cv2.WND_PROP_FULLSCREEN,
        cv2.WINDOW_FULLSCREEN,
    )

    # Below is the never ending loop that determines what will happen when an
    # object is identified.
    while True:
        success, img = cap.read()

        # Below provides a huge amount of controll. the 0.45 number is the
        # threshold number, the 0.2 number is the nms number)
        result, objectInfo = getObjects(img, 0.45, 0.2)

        cv2.imshow(window_name, img)
        cv2.waitKey(1)

    cv2.destroyAllWindows()
