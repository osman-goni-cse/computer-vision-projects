import face_recognition
import cv2
import numpy as np
import os

video_capture = cv2.VideoCapture(0)

# Load a sample picture and learn how to recognize it.
# obama_image = face_recognition.load_image_file("Obama.jpg")
# obama_face_encoding = face_recognition.face_encodings(obama_image)[0]
#
# # Load a second sample picture and learn how to recognize it.
# biden_image = face_recognition.load_image_file("biden.jpg")
# biden_face_encoding = face_recognition.face_encodings(biden_image)[0]
#
# # Load a third sample picture and learn how to recognize it.
# osman_image = face_recognition.load_image_file("Osman Goni.jpg")
# osman_face_encoding = face_recognition.face_encodings(osman_image)[0]
#
# # Load a fourth sample picture and learn how to recognize it.
# raju_image = face_recognition.load_image_file("Raju.jpg")
# raju_face_encoding = face_recognition.face_encodings(raju_image)[0]

folder_path = 'images'
known_face_encodings = []
known_face_names = []

with os.scandir(folder_path) as folder:
    for entry in folder:
        if entry.is_file() and entry.name.endswith(('.jpg', '.jpeg', '.png')):
            file_name_without_ext = os.path.splitext(entry.name)[0]
            # print(file_name_without_ext)
            load_image = face_recognition.load_image_file(folder_path + "/" +  entry.name)
            image_face_encoding = face_recognition.face_encodings(load_image)[0]
            known_face_encodings.append(image_face_encoding)
            known_face_names.append(file_name_without_ext)

# Create arrays of known face encodings and their names
# known_face_encodings = [
#     obama_face_encoding,
#     biden_face_encoding,
#     osman_face_encoding,
#     raju_face_encoding
# ]
# known_face_names = [
#     "Barack Obama",
#     "Joe Biden",
#     "Osman Goni",
#     "Raju"
# ]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Only process every other frame of video to save time
    if process_this_frame:
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        # rgb_small_frame = small_frame[:, :, ::-1]
        rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        # face_encodings = face_recognition.face_encodings(np.ascontiguousarray(rgb_small_frame[:, :, ::-1]), face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            print("Matches " , matches)

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            print(face_distances)
            best_match_index = np.argmin(face_distances)
            # print(np.argmin(face_distances))
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()