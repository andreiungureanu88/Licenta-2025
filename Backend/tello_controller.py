import threading
from djitellopy import Tello


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

