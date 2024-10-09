import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Load the YOLO model (cfg and weights)
model_cfg = "C:\\Users\\KK COMPUTERS\\Desktop\\Detection\\yolov3.cfg"
model_weights = "C:\\Users\\KK COMPUTERS\\Desktop\\Detection\\yolov3.weights"

# Load YOLO
net = cv2.dnn.readNetFromDarknet(model_cfg, model_weights)

# Load the COCO class labels YOLO was trained on
with open("C:\\Users\\KK COMPUTERS\\Desktop\\Detection\\coco.names", "r") as f:
    CLASSES = [line.strip() for line in f.readlines()]

# Get the output layer names from YOLO
layer_names = net.getLayerNames()
try:
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
except TypeError:
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Streamlit UI
st.title("Live Vehicle Detection App")
st.write("Start the camera feed and detect vehicles such as cars, buses, motorbikes, etc.")

# Button to start the live feed
if st.button('Start Camera'):
    # Access the webcam
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            st.write("Failed to grab frame from camera.")
            break

        # Resize frame for faster processing (optional)
        frame = cv2.resize(frame, (640, 480))

        # Get image dimensions
        (h, w) = frame.shape[:2]

        # Create a blob from the image and perform a forward pass of YOLO
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        layer_outputs = net.forward(output_layers)

        # Initialize lists to hold detection data
        boxes = []
        confidences = []
        classIDs = []
        detected_vehicles = []

        # Loop over each output layer's detections
        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                if confidence > 0.5:
                    box = detection[0:4] * np.array([w, h, w, h])
                    (centerX, centerY, width, height) = box.astype("int")
                    startX = int(centerX - (width / 2))
                    startY = int(centerY - (height / 2))

                    boxes.append([startX, startY, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        # Perform non-maxima suppression to suppress weak overlapping boxes
        indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)

        if len(indices) > 0:
            for i in indices.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                # Only draw bounding boxes around vehicles
                if CLASSES[classIDs[i]] in ["car", "bus", "motorbike", "truck", "bicycle", "minitruck"]:
                    color = (0, 255, 0)  # Green for vehicle detection
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)
                    text = "{}: {:.4f}".format(CLASSES[classIDs[i]], confidences[i])
                    cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    # Add detected vehicle to the list
                    detected_vehicles.append(CLASSES[classIDs[i]])

        # Display the frame with detected vehicles in Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result_image = Image.fromarray(frame_rgb)
        st.image(result_image, caption="Detected Vehicles", use_column_width=True)

        # Display the list and count of detected vehicles
        vehicle_count = len(detected_vehicles)
        st.write(f"Number of vehicles detected: {vehicle_count}")

        # List the detected vehicles
        if vehicle_count > 0:
            st.write("Detected vehicle types:")
            st.write(detected_vehicles)
        else:
            st.write("No vehicles detected.")

        # Break if Enter key (ASCII 13) is pressed
        if cv2.waitKey(1) & 0xFF == 13:  # Enter key's ASCII code is 13
            break

    # Release the webcam and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
