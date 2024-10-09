import cv2 
import numpy as np
import streamlit as st
from PIL import Image
import time

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
st.title("Vehicle Detection App with Traffic Management")
st.write("Upload images representing different roads, and the app will detect vehicles and manage traffic lights with real-time countdown.")

# Initialize a list to store the results of each image
vehicle_data = []

# Process images one at a time, up to 4 separate uploads (representing 4-way intersection)
for i in range(4):
    st.write(f"Upload Image for Road {i+1}")

    # Image uploader widget
    uploaded_file = st.file_uploader(f"Choose Image for Road {i+1}...", type="jpg", key=f"uploader_{i}")

    if uploaded_file is not None:
        # Convert the file to an OpenCV image
        image = np.array(Image.open(uploaded_file).convert('RGB'))

        # Get image dimensions
        (h, w) = image.shape[:2]

        # Create a blob from the image and perform a forward pass of YOLO
        blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        layer_outputs = net.forward(output_layers)

        # Initialize lists to hold detection data
        boxes = []
        confidences = []
        classIDs = []

        # List to store detected vehicle types
        detected_vehicles = []

        # Loop over each output layer's detections
        for output in layer_outputs:
            for detection in output:
                # Extract the confidence and classID
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                # Filter out weak predictions by ensuring the confidence is greater than a threshold
                if confidence > 0.5:
                    # Scale the bounding box coordinates back relative to the size of the image
                    box = detection[0:4] * np.array([w, h, w, h])
                    (centerX, centerY, width, height) = box.astype("int")

                    # Use the center to derive the top and left corner of the bounding box
                    startX = int(centerX - (width / 2))
                    startY = int(centerY - (height / 2))

                    # Update boxes, confidences, and classIDs lists
                    boxes.append([startX, startY, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        # Perform non-maxima suppression to suppress weak overlapping boxes
        indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)

        # Loop over the indices we are keeping after suppression
        if len(indices) > 0:
            for idx in indices.flatten():
                # Extract the bounding box coordinates
                (x, y) = (boxes[idx][0], boxes[idx][1])
                (w, h) = (boxes[idx][2], boxes[idx][3])

                # Only proceed if the detected object is a vehicle (car, bus, motorbike, etc.)
                if CLASSES[classIDs[idx]] in ["car", "bus", "motorbike", "truck", "bicycle", "minitruck"]:
                    # Draw a bounding box rectangle and label on the image
                    color = (0, 255, 0)  # Green for vehicle detection
                    cv2.rectangle(image, (x, y), (x + w, y + h), color, 1)
                    text = "{}: {:.4f}".format(CLASSES[classIDs[idx]], confidences[idx])
                    cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    # Add detected vehicle to the list
                    detected_vehicles.append(CLASSES[classIDs[idx]])

        # Display the list and count of detected vehicles
        vehicle_count = len(detected_vehicles)

        # Store the vehicle count and types for each image
        vehicle_data.append({
            "road": i+1,
            "count": vehicle_count,
            "vehicles": detected_vehicles
        })

# After processing all roads, decide the traffic light status
if vehicle_data:
    for data in vehicle_data:
        st.write()

    # Sort the roads based on vehicle count in descending order
    sorted_vehicle_data = sorted(vehicle_data, key=lambda x: x['count'], reverse=True)

# Traffic light logic: Green for each road based on vehicle count
    st.write("Traffic Light Status:")
    
    # Simulate the traffic light system, each road gets the green signal in sequence, but display in one frame
    total_green_time = 30  # Total time for the green light phase
    roads = len(sorted_vehicle_data)  # Number of roads

    # Placeholder for traffic light statuses
    traffic_placeholder = st.empty()

    # Countdown simulation and display all roads in one frame
    for current_green_road in range(roads):
        green_time = total_green_time

        # Countdown for each road with green light
        for seconds in range(green_time, 0, -1):
            # Display traffic light status for all roads in one frame
            status_display = []
            for i, data in enumerate(sorted_vehicle_data):
                if i == current_green_road:
                    status_display.append(f"\n🚦 Road {data['road']}: **🟢Green** for {seconds} seconds remaining"
                        f" ({data['count']} vehicles detected.)")
                else:
                    status_display.append(f"\n🚦 Road {data['road']}:🔴 Red"
                                          f" ({data['count']} vehicles detected.)")

            # Update the placeholder with the traffic light status in one frame
            traffic_placeholder.markdown("\n".join(status_display))
            time.sleep(1)

    # After the green phase is over, mark all roads as red
    final_display = [f"\n🚦 Road {data['road']}: 🔴Red" for data in sorted_vehicle_data]
    traffic_placeholder.markdown("\n".join(final_display))