import cv2

# Load YOLO object detector
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Initialize tracker
tracker = cv2.MultiTracker_create()

# Capture video from webcam
cap = cv2.VideoCapture(0)

# Process each frame of video
while True:
    # Read frame from video capture
    ret, frame = cap.read()

    # Run object detection on frame
    blob = cv2.dnn.blobFromImage(frame, scalefactor=0.00392, size=(416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Parse detection results
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] == "person":
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                w = int(detection[2] * frame.shape[1])
                h = int(detection[3] * frame.shape[0])
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    # Update tracker with detected person boxes
    ok, updated_boxes = tracker.update(frame)
    for box in updated_boxes:
        x, y, w, h = [int(c) for c in box]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Initialize tracker for new person detections
    for i, box in enumerate(boxes):
        if i not in tracker.getIndexOfDetectedObjects():
            tracker.add(cv2.TrackerKCF_create(), frame, tuple(box))

    # Display frame with tracked person boxes
    cv2.imshow("Frame", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
