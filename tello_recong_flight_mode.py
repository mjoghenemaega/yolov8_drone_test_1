import os
import cv2
import time
import pandas as pd
import numpy as np
from ultralytics import YOLO
from djitellopy import Tello
import keypressModule as kp

# Initialize keypress module
kp.init()

# Connect to Tello
tello = Tello()
tello.connect()
tello.streamon()
print(f"Battery Level: {tello.get_battery()}%")

# Load the YOLO model
model = YOLO('yolov8s.pt')

# Load class labels from coco.txt file
with open("coco.txt", "r") as my_file:
    class_list = my_file.read().split("\n")
print(class_list)

# Define the output folder path and create it if it doesn't exist
output_folder = "./pictures"
os.makedirs(output_folder, exist_ok=True)

# Initialize flags and counters
person_detected = False
image_count = 0  # Counter for saved images

def getKeyboardInput():
    lr, fb, ud, yv = 0, 0, 0, 0
    speed = 50

    # Drone movement controls
    if kp.getKey("LEFT"): lr = -speed 
    elif kp.getKey("RIGHT"): lr = speed   
    if kp.getKey("UP"): fb = speed 
    elif kp.getKey("DOWN"): fb = -speed
    if kp.getKey("w"): ud = speed 
    elif kp.getKey("s"): ud = -speed
    if kp.getKey("a"): yv = speed 
    elif kp.getKey("d"): yv = -speed

    # Takeoff and landing
    if kp.getKey("t"): tello.takeoff()
    if kp.getKey("q"): tello.land()

    # Manual capture with 'z' key
    if kp.getKey("z"):
        image_filename = os.path.join("Resources/Images", f"{time.time()}.jpg")
        cv2.imwrite(image_filename, img)
        print(f"Image saved: {image_filename}")
        time.sleep(0.3)

    return [lr, fb, ud, yv]

while True:
    # Get keyboard input for drone control
    vals = getKeyboardInput()
    tello.send_rc_control(vals[0], vals[1], vals[2], vals[3])

    # Get frame from Tello
    img = tello.get_frame_read().frame
    if img is None:
        break

    # Resize the frame
    img = cv2.resize(img, (600, 500))

    # Run YOLO model on the frame
    results = model.predict(img)
    detections = results[0].boxes.data

    # Process detection results
    px = pd.DataFrame(detections).astype("float")
    new_person_detected = False

    for index, row in px.iterrows():
        x1, y1, x2, y2 = int(row[0]), int(row[1]), int(row[2]), int(row[3])
        d = int(row[5])
        c = class_list[d]

        if c == "person":
            new_person_detected = True

            # Save the frame if a new person is detected
            if not person_detected:
                image_count += 1
                image_filename = os.path.join(output_folder, f"person_detected_{image_count}.jpg")
                cv2.imwrite(image_filename, img)
                print(f"New person detected, saving image as {image_filename}")

            # Draw bounding box and label
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, str(c), (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)

    # Update the person_detected flag
    person_detected = new_person_detected

    # Display the frame
    cv2.imshow("Tello Feed", img)
    if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to exit
        break

# Release resources
tello.streamoff()
cv2.destroyAllWindows()
