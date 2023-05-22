import cv2
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

video_capture = cv2.VideoCapture(0)

people_list = []
last_count = 0
people_count = 0

while True:
    _, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray, 1.3, 5)

    detections = faceCascade.detectMultiScale(gray, 1.15, 5)

    for i in range(len(detections)):
        face_i = detections[i]
        x, y, w, h = face_i

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 222, 0), 1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        # people_list.insert(len(people_list)+1,i)

        # cv2.putText(frame, "id: "+str ( people_list[i]), (x, y), font, 2, (255, 255, 255), 2, cv2.LINE_AA)
        if len(detections) > last_count:
            people_count += len(detections) - last_count

        last_count = len(detections)
        # cv2.putText(frame, "Number of People: " + str(last_count))

    # Display the resulting frame
    cv2.putText(frame, f"Number of people: {people_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


video_capture.release()
cv2.destroyAllWindows()