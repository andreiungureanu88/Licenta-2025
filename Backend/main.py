from utils import *
from tello_controller import TelloController
from face_recognizer import FaceRecognizer
from tello_face_tracker import TelloFaceTracker


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
            tello = TelloController()
            tello.land()  # Land the drone
            print("Drone is landing...")
        elif choice == '4':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please select a valid option.")

if __name__ == '__main__':
    main_menu()
