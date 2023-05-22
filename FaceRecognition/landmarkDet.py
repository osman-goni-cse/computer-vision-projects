import cv2
import mediapipe as mp

# Initialize the face mesh detector
face_mesh_detector = mp.solutions.face_mesh.FaceMesh()

# Start the video capture
cap = cv2.VideoCapture(0)

while True:
    # Capture the current frame
    ret, frame = cap.read()

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces in the frame
    faces = face_mesh_detector.process(rgb_frame)

    # Draw the face mesh on the frame
    for face in faces.multi_face_landmarks:
        for landmark in face.landmark:
            x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)


    # Display the frame
    cv2.imshow("Frame", frame)

    # Wait for a key press
    key = cv2.waitKey(1) & 0xFF

    # If the key `q` is pressed, stop the loop
    if key == ord("q"):
        break

# Release the video capture
cap.release()

# Close all windows
cv2.destroyAllWindows()
