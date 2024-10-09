import cv2
import numpy as np
import streamlit as st
import time

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load class names (COCO dataset labels)
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Streamlit page settings
st.title("Real-Time Vehicle Detection with Traffic Light Priority")
st.write("Vehicle detection on roads using YOLO and OpenCV, with priority given to roads with more large vehicles.")

# Initialize variables for detection
total_frames = 4  # Number of frames to process
vehicle_count = 0
total_detections = 0
correct_detections = 0
frame_count = 0  # To track which frame we are processing

# Open the camera
cap = cv2.VideoCapture(0)  # 0 is for the primary webcam

# Check if the camera opened successfully
if not cap.isOpened():
    st.error("Failed to open webcam")
else:
    road_vehicle_counts = []  # List to store vehicle counts for each road (frame)

    while frame_count < total_frames:
        # Capture a single frame
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture video frame")
            break

        height, width, channels = frame.shape

        # Prepare the frame for YOLO model
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        # Initialize lists for detected objects
        class_ids = []
        confidences = []
        boxes = []
        large_vehicle_count = 0  # To count large vehicles like bus or truck

        # Process detections
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.5 and classes[class_id] in ["car", "bus", "truck","motorbike","bicycle"]:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

                    # Count large vehicles for prioritization
                    if classes[class_id] in ["bus", "truck"]:
                        large_vehicle_count += 1

        # Non-Maximum Suppression to remove overlapping boxes
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        # Draw bounding boxes and labels on detected vehicles
        font = cv2.FONT_HERSHEY_PLAIN
        frame_vehicle_count = 0  # Count vehicles in the current frame

        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                color = (0, 255, 0)  # Green for vehicle detection
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 10), font, 1, color, 2)

                # Update detection counts
                total_detections += 1
                correct_detections += 1  # Assuming all detected vehicles are correct for this example
                vehicle_count += 1
                frame_vehicle_count += 1  # Increment vehicle count for current frame

        # Add vehicle count and large vehicle count of the frame to the list of road vehicle counts
        road_vehicle_counts.append({'road': frame_count + 1, 'count': frame_vehicle_count, 'large_vehicle_count': large_vehicle_count})

        # Convert the frame to RGB format for Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display the frame in Streamlit
        st.image(frame_rgb, channels="RGB", use_column_width=True)

        # Display the vehicle count for the current frame below the image
        st.write(f"Vehicles detected in frame {frame_count + 1}: {frame_vehicle_count} (Large vehicles: {large_vehicle_count})")

        # Increment frame count to move to the next frame
        frame_count += 1

    # Simulate the traffic light system, prioritizing based on large vehicle count
    total_green_time = 30  # Total time for the green light phase
    # Sort roads by the number of large vehicles (descending)
    sorted_roads = sorted(road_vehicle_counts, key=lambda x: x['large_vehicle_count'], reverse=True)

    # Placeholder for traffic light statuses
    traffic_placeholder = st.empty()

    # Countdown simulation and display all roads in one frame
    for current_road in sorted_roads:
        green_time = total_green_time
        road_name = f"Road {current_road['road']}"
        # Countdown for each road with green light
        for seconds in range(green_time, 0, -1):
            # Display traffic light status for all roads
            status_display = []
            for road in sorted_roads:
                if road == current_road:
                    status_display.append(f"\n 🚦 {road_name}: **🟢 Green** for {seconds} seconds remaining"
                                          f" ({road['count']} vehicles detected, {road['large_vehicle_count']} large vehicles).")
                else:
                    status_display.append(f"\n🚦 Road {road['road']}: 🔴 Red"
                                          f" ({road['count']} vehicles detected, {road['large_vehicle_count']} large vehicles).")
            # Update the placeholder with the traffic light status
            traffic_placeholder.markdown("\n".join(status_display))
            time.sleep(1)

    # After the green phase is over, mark all roads as red
    final_display = [f"\n🚦 Road {road['road']}: 🔴 Red" for road in sorted_roads]
    traffic_placeholder.markdown("\n".join(final_display))

    # Release the camera after all frames are processed
    cap.release()
    cv2.destroyAllWindows()

# Optionally save the final image of the last detected frame
if vehicle_count > 0:
    output_image_path = "detected_vehicles.png"
else:
    st.write("No vehicles detected.")
