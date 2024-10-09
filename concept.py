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
st.title("Vehicle Detection App")
st.write("Upload an image, and the app will detect vehicles such as cars, buses, motorbikes, etc.")

# Initialize a list to store the results of each image
vehicle_data = []

# Process images one at a time, up to 4 separate uploads
for i in range(4):
    st.write(f"Upload Image {i+1}")

    # Image uploader widget
    uploaded_file = st.file_uploader(f"Choose Image {i+1}...", type="jpg", key=f"uploader_{i}")

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

        # Convert the image back to PIL format for display in Streamlit
        result_image = Image.fromarray(image)

        # Display the result for the current image
        #st.image(result_image, caption=f"Detected Vehicles in Image {i + 1}", use_column_width=True)

        # Display the list and count of detected vehicles
        vehicle_count = len(detected_vehicles)
        #st.write(f"Number of vehicles detected in Image {i+1}: {vehicle_count}")

        # List the detected vehicles
        if vehicle_count > 0:
            st.write("vehicle Detected sucessfully:")
            #st.write(detected_vehicles)
        else:
            st.write("No vehicles detected.")

        # Store the vehicle count and types for each image
        vehicle_data.append({
            "image": i+1,
            "count": vehicle_count,
            "vehicles": detected_vehicles
        })

# Optionally, you can display a summary after all images are uploaded
if vehicle_data:
    st.write("Summary of detected vehicles across all images:")
    for data in vehicle_data:
        st.write(f"Image {data['image']}: {data['count']} vehicles detected.")
        #st.write(f"Vehicle types: {data['vehicles']}")
