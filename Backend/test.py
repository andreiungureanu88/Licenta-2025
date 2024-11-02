from utils import *


# Initialize Firebase
cred = credentials.Certificate('tellodrone-baebd-firebase-adminsdk-v09bs-1b7c673b01.json')
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://tellodrone-baebd-default-rtdb.europe-west1.firebasedatabase.app/"
})

class TelloController:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(TelloController, cls).__new__(cls)
                    cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self.tello = Tello()
        self.tello.connect()
        self.tello.streamon()
        print(f"Connected to Tello. Battery: {self.tello.get_battery()}%")

    def get_frame(self):
        return self.tello.get_frame_read().frame

    def send_rc_control(self, left_right_velocity, for_back_velocity, up_down_velocity, yaw_velocity):
        self.tello.send_rc_control(left_right_velocity, for_back_velocity, up_down_velocity, yaw_velocity)

    def takeoff(self):
        self.tello.takeoff()

    def land(self):
        self.tello.land()

    def streamoff(self):
        self.tello.streamoff()

    def streamon(self):
        self.tello.streamon()



class FaceDatabase:
    def __init__(self, folder='Images'):
        self.folder = folder
        self.detector = cv2.FaceDetectorYN.create(
            "face_detection_yunet_2023mar.onnx", "", [320, 320], 0.6, 0.3, 5000, 0, 0)
        self.face_net = FaceNet()
        self.database = {}
        self.ref = db.reference('People')

    @staticmethod
    def generate_unique_id():
        now = datetime.datetime.now()
        random_number = random.randint(1000, 9999)
        return now.strftime("%Y%m%d%H%M%S") + str(random_number)

    def build_database(self):
        for filename in listdir(self.folder):
            path = os.path.join(self.folder, filename)
            image = cv2.imread(path)

            img_width = int(image.shape[1])
            img_height = int(image.shape[0])
            self.detector.setInputSize((img_width, img_height))

            detections = self.detector.detect(image)

            if detections[1] is not None and len(detections[1]) > 0:
                x, y, w, h = detections[1][0][:4].astype(int)
            else:
                x, y, w, h = 1, 1, 10, 10

            x1, y1 = abs(x), abs(y)
            x2, y2 = x1 + w, y1 + h

            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            rgb_image = Image.fromarray(rgb_image)
            rgb_array = np.asarray(rgb_image)

            face_region = rgb_array[y1:y2, x1:x2]
            face_image = Image.fromarray(face_region)
            face_image = face_image.resize((160, 160))
            face_array = np.asarray(face_image)
            face_array = expand_dims(face_array, axis=0)

            with tf.device('/GPU:0'):
                face_embedding = self.face_net.embeddings(face_array)[0]

            person_id = self.generate_unique_id()
            name = os.path.splitext(filename)[0]
            date_added = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            self.database[person_id] = {
                "name": name,
                "date_added": date_added,
                "last_recognition": date_added,
                "total_recognition": 1,
                "signature": face_embedding.tolist()
            }

            self.ref.child(person_id).set(self.database[person_id])

class FaceRecognizer:
    def __init__(self):
        self.detector = cv2.FaceDetectorYN.create(
            "face_detection_yunet_2023mar.onnx", "", [320, 320], 0.6, 0.3, 5000, 0, 0)
        self.face_net = FaceNet()
        self.tello = TelloController()
        self.database = self.load_database_from_firebase()
        self.unknown_faces_queue = queue.Queue()

        # Start the consumer thread
        self.consumer_thread = threading.Thread(target=self.consume_unknown_faces)
        self.consumer_thread.daemon = True
        self.consumer_thread.start()

    def load_database_from_firebase(self):
        ref = db.reference('People')
        data = ref.get()
        return {key: value for key, value in data.items()} if data else {}

    def is_image_clear(self, image, sharpness_threshold=6):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
        sharpness = laplacian.var()

        min_val, max_val = gray_image.min(), gray_image.max()
        contrast = max_val - min_val

        print(f"Sharpness: {sharpness}")
        return sharpness >= sharpness_threshold

    def is_face_in_database(self, face_embedding):
        min_distance = float('inf')
        for key, value in self.database.items():
            stored_signature = np.array(value['signature'])
            dist = np.linalg.norm(stored_signature - face_embedding)
            if dist < min_distance:
                min_distance = dist

        return min_distance < 1

    def consume_unknown_faces(self):
        while True:
            face_image, face_embedding = self.unknown_faces_queue.get()

            if not self.is_face_in_database(face_embedding):
                self.add_new_face(face_image, face_embedding)

            self.unknown_faces_queue.task_done()

    def add_new_face(self, face_image, face_embedding):

        if not self.is_image_clear(np.array(face_image)):
            print("The image is not clear. The window will not be opened.")
            return

        root = tk.Tk()
        root.title("Add New Face")

        def on_submit():
            name = entry.get()
            if name:
                person_id = FaceDatabase.generate_unique_id()
                date_added = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                new_entry = {
                    "name": name,
                    "date_added": date_added,
                    "last_recognition": date_added,
                    "total_recognition": 1,
                    "signature": face_embedding.tolist()
                }

                self.database[person_id] = new_entry
                db.reference('People').child(person_id).set(new_entry)

                print(f"Added new person: {name}")

            root.destroy()

        def on_cancel():
            root.destroy()

        # Convert face_image to RGB before creating PhotoImage
        face_image = cv2.cvtColor(np.array(face_image), cv2.COLOR_BGR2RGB)
        face_image = Image.fromarray(face_image)  # Create an Image object from the RGB array
        img = ImageTk.PhotoImage(image=face_image)  # Create a PhotoImage object

        panel = tk.Label(root, image=img)
        panel.pack(side="top", fill="both", expand="yes")

        entry = tk.Entry(root)
        entry.pack(side="top", fill="both", expand="yes")
        entry.focus()

        btn_ok = tk.Button(root, text="OK", command=on_submit)
        btn_ok.pack(side="left", fill="both", expand="yes")

        btn_cancel = tk.Button(root, text="Cancel", command=on_cancel)
        btn_cancel.pack(side="right", fill="both", expand="yes")

        root.mainloop()

    def recognize_faces(self):
        while True:
            frame = self.tello.get_frame()
            if frame is None:
                break

            # Convert frame from BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            img_width = int(frame_rgb.shape[1])
            img_height = int(frame_rgb.shape[0])
            self.detector.setInputSize((img_width, img_height))
            detections = self.detector.detect(frame_rgb)

            if detections[1] is not None and len(detections[1]) > 0:
                for detection in detections[1]:
                    x, y, w, h = detection[:4].astype(int)

                    x1, y1 = abs(x), abs(y)
                    x2, y2 = x1 + w, y1 + h

                    face_region = frame_rgb[y1:y2, x1:x2]
                    face_image = Image.fromarray(face_region)
                    face_image = face_image.resize((160, 160))
                    face_array = np.asarray(face_image)
                    face_array = expand_dims(face_array, axis=0)

                    with tf.device('/GPU:0'):
                        face_embedding = self.face_net.embeddings(face_array)[0]

                    min_distance = float('inf')
                    identity = 'UNKNOWN'

                    for key, value in self.database.items():
                        stored_signature = np.array(value['signature'])
                        dist = np.linalg.norm(stored_signature - face_embedding)
                        if dist < min_distance:
                            min_distance = dist
                            identity = value['name']

                    if min_distance < 1:
                        cv2.putText(frame, identity, (x1, y1 - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    else:
                        cv2.putText(frame, "UNKNOWN", (x1, y1 - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2, cv2.LINE_AA)
                        self.unknown_faces_queue.put((face_image, face_embedding))

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.imshow('Face Recognition', frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

        self.tello.streamoff()
        cv2.destroyAllWindows()


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
        detector = cv2.FaceDetectorYN.create("face_detection_yunet_2023mar.onnx", "", [320, 320], 0.6, 0.3, 5000, 0, 0)
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



def main_menu():

    while True:
        print("\n--- Tello Drone Control Menu ---")
        print("1. Face Recognition")
        print("2. Face Tracking")
        print("3. Land")
        print("4. Exit")
        choice = input("Select an option (1-4): ")

        if choice == '1':
            print("Starting face recognition...")
            face_recognizer = FaceRecognizer()
            face_recognizer.recognize_faces()
        elif choice == '2':
            print("Starting face tracking...")
            tracker = TelloFaceTracker()
            tracker.run()
        elif choice == '3':
            tello=TelloController()
            tello.land()  # Land the drone
            print("Drone is landing...")
        elif choice == '4':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please select a valid option.")

if __name__ == '__main__':
    main_menu()


