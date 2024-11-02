from utils import *
from tello_controller import *


class TelloFaceTracker:
    def __init__(self, w=360, h=240, fbRange=[6200, 6800], pid=[0.4, 0.4, 0]):
        self.w = w
        self.h = h
        self.fbRange = fbRange
        self.pid = pid
        self.pError = 0
        self.startCounter = 0
        self.mydrone = self.initialize()

    def initialize(self):
        mytello = TelloController()
        mytello.takeoff()  # Asigură-te că drona decolează la inițializare
        mytello.streamon()
        return mytello

    def telloGetFrame(self):
        img = self.mydrone.get_frame()  # Get the frame directly
        if img is None:
            print("Nu am primit niciun cadru video.")
            return None
        else:
            img = cv2.resize(img, (self.w, self.h))
        return img

    def findFace(self, img):
        detector = cv2.FaceDetectorYN.create("./Configuration/face_detection_yunet_2023mar.onnx", "", [320, 320], 0.6, 0.3, 5000, 0, 0)
        img_W = int(img.shape[1])
        img_H = int(img.shape[0])
        detector.setInputSize((img_W, img_H))

        detections = detector.detect(img)

        myFaceList = []
        myFaceListArea = []

        if detections[1] is not None and len(detections[1]) > 0:
            for detection in detections[1]:
                x, y, w, h = detection[:4].astype(int)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cx = x + w // 2
                cy = y + h // 2
                area = w * h
                cv2.circle(img, [cx, cy], 5, (0, 255, 0), cv2.FILLED)
                myFaceListArea.append(area)
                myFaceList.append([cx, cy])

        if len(myFaceListArea) != 0:
            i = myFaceListArea.index(max(myFaceListArea))
            return img, [myFaceList[i], myFaceListArea[i]]
        else:
            return img, [[0, 0], 0]

    def trackFace(self, info):
        fb = 0
        ud = 0  # Variabila pentru mișcarea verticală
        area = info[1]
        cx, cy = info[0]

        # Dacă nu există față, menține drona stabilă
        if area == 0:
            self.mydrone.send_rc_control(0, 0, 0, 0)  # Trimite comanda de control pentru a rămâne stabilă
            return 0  # Nu actualiza pError dacă nu există față

        error = cx - self.w // 2
        speed = self.pid[0] * error + self.pid[1] * (error - self.pError)
        speed = int(np.clip(speed, -100, 100))

        if self.fbRange[0] < area < self.fbRange[1]:
            fb = 0
        elif area > self.fbRange[1]:
            fb = -20  # Dronă îndepărtează
        elif area < self.fbRange[0] and area != 0:
            fb = 20  # Dronă se apropie

        # Logica pentru mișcarea verticală
        if cy < self.h // 3:  # Dacă fața este în partea superioară a imaginii
            ud = 20  # Dronă urcă
        elif cy > self.h * 2 // 3:  # Dacă fața este în partea inferioară a imaginii
            ud = -20  # Dronă coboară
        else:
            ud = 0  # Nicio mișcare verticală

        print(f"Control - Speed: {speed}, FB: {fb}, UD: {ud}, Error: {error}")  # Print pentru debugging
        self.mydrone.send_rc_control(0, fb, ud, speed)  # Trimite comanda de control pentru dronă
        return error

    def run(self):
        while True:
            img = self.telloGetFrame()
            if img is None:
                continue

            img, info = self.findFace(img)
            print("Center", info[0], "Area", info[1])  # Print pentru debugging

            self.pError = self.trackFace(info)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (640, 380))

            cv2.imshow('Image', img)

            if cv2.waitKey(1) & 0xFF == 27:
                self.mydrone.land()
                break

        self.mydrone.streamoff()
        cv2.destroyAllWindows()
