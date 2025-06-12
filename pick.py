import cvzone
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import math
import random

# Initialize camera
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Set video width
cap.set(4, 720)   # Set video height

# Initialize the hand detector with higher confidence level
detector = HandDetector(detectionCon=0.9, maxHands=1)

# Load the overlay image (e.g., an emoji or icon)
overlay_image = cv2.imread("C://Users//abdeslam//Desktop//object detection//picking objects//queen.png", cv2.IMREAD_UNCHANGED)  # Use an image with alpha channel

# Load the full screen overlay image
full_screen_overlay = cv2.imread("C://Users//abdeslam//Desktop//object detection//picking objects//bo.png", cv2.IMREAD_UNCHANGED)

# Check if the overlay image is loaded properly
if overlay_image is None:
    print("Error: Overlay image not found or unable to load.")
    exit()

# Resize overlay image to a fixed width and maintain aspect ratio
overlay_width = 100  # Width to resize the overlay image to
overlay_height = int(overlay_image.shape[0] * (overlay_width / overlay_image.shape[1]))  # Keep aspect ratio
overlay_resized = cv2.resize(overlay_image, (overlay_width, overlay_height))

# Resize full-screen image to cover the entire camera frame
full_screen_resized = cv2.resize(full_screen_overlay, (1280, 720))  # Match the camera frame size

# Initialize drag mode flag and initial position of the image
is_dragging = False
image_x_offset, image_y_offset = 200, 200  # Initial position of the image

while True:
    success, img = cap.read()
    
    if not success:
        print("Failed to capture image from camera.")
        break
    
    img = cv2.flip(img, 1)  # Flip image horizontally for mirror effect
    
    # Find the hands and update the image with hand landmarks
    hands, img = detector.findHands(img)
    
    # Overlay the full screen image first (this will be at the bottom layer)
    for c in range(0, 3):  # Iterate through the channels (RGB)
        img[:, :, c] = full_screen_resized[:, :, c] * (full_screen_resized[:, :, 3] / 255.0) + \
                       img[:, :, c] * (1.0 - full_screen_resized[:, :, 3] / 255.0)
    
    if hands:
        hand = hands[0]
        
        # Draw a circle on the thumb tip (landmark 4)
        thumb_tip = hand['lmList'][4]
        cv2.circle(img, (thumb_tip[0], thumb_tip[1]), 10, (255, 75, 75), -1)  # Red circle
        
        # Draw a circle on the index finger tip (landmark 8)
        index_tip = hand['lmList'][8]
        cv2.circle(img, (index_tip[0], index_tip[1]), 10, (75, 255, 60), -1)  # Green circle
        
        
        # Draw a line between the thumb and index finger in orange
        cv2.line(img, (thumb_tip[0], thumb_tip[1]), (index_tip[0], index_tip[1]), (0, 165, 255), 3)  # Orange line
        
        # Calculate the midpoint of the line between the thumb and index finger
        midpoint_x = int((thumb_tip[0] + index_tip[0]) / 2)
        midpoint_y = int((thumb_tip[1] + index_tip[1]) / 2)
        
        # Draw a circle at the midpoint (50x50 pixels)
        cv2.circle(img, (midpoint_x, midpoint_y), 25, (0, 165, 255), -1)  # Orange circle
        
        # Check if both thumb and index tips are inside the circle
        distance_thumb_mid = math.hypot(thumb_tip[0] - midpoint_x, thumb_tip[1] - midpoint_y)
        distance_index_mid = math.hypot(index_tip[0] - midpoint_x, index_tip[1] - midpoint_y)
        
        # Check if the circle is within the bounds of the overlay image
        is_circle_over_image = (midpoint_x >= image_x_offset and midpoint_x <= image_x_offset + overlay_resized.shape[1] and
                                midpoint_y >= image_y_offset and midpoint_y <= image_y_offset + overlay_resized.shape[0])
        
        if distance_thumb_mid <= 25 and distance_index_mid <= 25 and is_circle_over_image:  # Inside the circle and on the image
            is_dragging = True
        else:
            if is_dragging:  # If we were dragging, and now we're outside the circle
                # Update image position to the circle's position
                image_x_offset, image_y_offset = midpoint_x - overlay_resized.shape[1] // 2, midpoint_y - overlay_resized.shape[0] // 2
            is_dragging = False
        
    # Overlay the smaller image on the original image (account for transparency)
    for c in range(0, 3):  # Iterate through the channels (RGB)
        img[image_y_offset:image_y_offset+overlay_resized.shape[0], image_x_offset:image_x_offset+overlay_resized.shape[1], c] = \
            overlay_resized[:, :, c] * (overlay_resized[:, :, 3] / 255.0) + \
            img[image_y_offset:image_y_offset+overlay_resized.shape[0], image_x_offset:image_x_offset+overlay_resized.shape[1], c] * (1.0 - overlay_resized[:, :, 3] / 255.0)

    # Display the image
    cv2.imshow("Hand Detection", img)
    
    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
