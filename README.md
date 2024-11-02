# Licenta-2025

### Changes and Notes:
1. **Singleton Controller Class (`TelloController`)**:
    - Ensures a single instance of the Tello drone connection is created and shared.
    - Manages the Tello drone operations like connecting, streaming, takeoff, and landing.

2. **Face Database Class (`FaceDatabase`)**:
    - Handles the initialization and building of the face database from stored images.
    - Generates unique IDs and stores the database in Firebase.

3. **Face Recognizer Class (`FaceRecognizer`)**:
    - Loads the face database from Firebase.
    - Consumes unknown faces and prompts user input via a GUI for adding new faces.
    - Recognizes faces in real-time and updates the database.

4. **Tello Face Tracker Class (`TelloFaceTracker`)**:
    - Uses PID control for tracking faces and sending RC commands to the Tello drone.

5. **Main Function**:
    - Initializes the face database and starts the face recognition and tracking process.

This structure ensures that the Tello drone connection is managed efficiently and integrates face detection, recognition, and tracking in a cohesive manner.