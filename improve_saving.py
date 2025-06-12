import cvzone
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import math
import random

# Initialize camera
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Initialize the hand detector
detector = HandDetector(detectionCon=0.9, maxHands=1)

# Load images
overlay_image = cv2.imread("C://Users//abdeslam//Desktop//object detection//picking objects//queen.png", cv2.IMREAD_UNCHANGED)
full_screen_overlay = cv2.imread("C://Users//abdeslam//Desktop//object detection//picking objects//bo.png", cv2.IMREAD_UNCHANGED)

if overlay_image is None:
    print("Error: Overlay image not found.")
    exit()

# Resize images
overlay_width = 100
overlay_height = int(overlay_image.shape[0] * (overlay_width / overlay_image.shape[1]))
overlay_resized = cv2.resize(overlay_image, (overlay_width, overlay_height))
full_screen_resized = cv2.resize(full_screen_overlay, (1280, 720))

# Video Writer setup
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter("C://Users//abdeslam//Desktop//object detection//chess.mp4", fourcc, 30.0, (1280, 720))

# Dragging setup
is_dragging = False
image_x_offset, image_y_offset = 200, 200

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image.")
        break

    img = cv2.flip(img, 1)
    hands, img = detector.findHands(img)

    # Full screen overlay
    for c in range(3):
        alpha = full_screen_resized[:, :, 3] / 255.0
        img[:, :, c] = full_screen_resized[:, :, c] * alpha + img[:, :, c] * (1 - alpha)

    if hands:
        hand = hands[0]
        lmList = hand['lmList']
        thumb_tip = lmList[4][:2]
        index_tip = lmList[8][:2]

        # Draw on hand
        cv2.circle(img, tuple(thumb_tip), 10, (255, 75, 75), -1)
        cv2.circle(img, tuple(index_tip), 10, (75, 255, 60), -1)
        cv2.line(img, tuple(thumb_tip), tuple(index_tip), (0, 165, 255), 3)

        midpoint_x = (thumb_tip[0] + index_tip[0]) // 2
        midpoint_y = (thumb_tip[1] + index_tip[1]) // 2
        midpoint = (midpoint_x, midpoint_y)

        cv2.circle(img, midpoint, 25, (0, 165, 255), -1)

        dist_thumb = math.hypot(thumb_tip[0] - midpoint_x, thumb_tip[1] - midpoint_y)
        dist_index = math.hypot(index_tip[0] - midpoint_x, index_tip[1] - midpoint_y)

        is_circle_over_image = (
            image_x_offset <= midpoint_x <= image_x_offset + overlay_resized.shape[1] and
            image_y_offset <= midpoint_y <= image_y_offset + overlay_resized.shape[0]
        )

        if dist_thumb <= 25 and dist_index <= 25 and is_circle_over_image:
            is_dragging = True
        else:
            if is_dragging:
                image_x_offset = midpoint_x - overlay_resized.shape[1] // 2
                image_y_offset = midpoint_y - overlay_resized.shape[0] // 2
            is_dragging = False

    # Overlay draggable image
    for c in range(3):
        alpha_overlay = overlay_resized[:, :, 3] / 255.0
        img[image_y_offset:image_y_offset + overlay_resized.shape[0],
            image_x_offset:image_x_offset + overlay_resized.shape[1], c] = (
            overlay_resized[:, :, c] * alpha_overlay +
            img[image_y_offset:image_y_offset + overlay_resized.shape[0],
                image_x_offset:image_x_offset + overlay_resized.shape[1], c] * (1 - alpha_overlay)
        )

    # Write frame to video file
    video_writer.write(cv2.cvtColor(img, cv2.COLOR_RGBA2BGR) if img.shape[2] == 4 else img)

    # Show frame
    cv2.imshow("Hand Detection", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
video_writer.release()
cv2.destroyAllWindows()
