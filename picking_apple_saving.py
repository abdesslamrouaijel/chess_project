import cvzone
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import math
import random

# Initialize camera
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Width
cap.set(4, 720)   # Height

# Define codec and output video writer (save as .mp4)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video = cv2.VideoWriter("C://Users//abdeslam//Desktop//object detection//output_hand_tracking.mp4", fourcc, 30.0, (1280, 720))

# Hand detector
detector = HandDetector(detectionCon=0.9, maxHands=1)

# Load overlay image (with alpha channel)
overlay_image = cv2.imread("C://Users//abdeslam//Desktop//object detection//picking objects//apple1.png", cv2.IMREAD_UNCHANGED)
if overlay_image is None:
    print("Error: Overlay image not found.")
    exit()

# Resize overlay image
overlay_width = 100
overlay_height = int(overlay_image.shape[0] * (overlay_width / overlay_image.shape[1]))
overlay_resized = cv2.resize(overlay_image, (overlay_width, overlay_height))

# Drag logic
is_dragging = False
image_x_offset, image_y_offset = 200, 200

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)  # Mirror
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        lm = hand['lmList']
        thumb_tip = lm[4]
        index_tip = lm[8]

        cv2.circle(img, (thumb_tip[0], thumb_tip[1]), 10, (255, 75, 75), -1)
        cv2.circle(img, (index_tip[0], index_tip[1]), 10, (75, 255, 60), -1)

        cv2.line(img, (thumb_tip[0], thumb_tip[1]), (index_tip[0], index_tip[1]), (0, 165, 255), 3)

        midpoint_x = (thumb_tip[0] + index_tip[0]) // 2
        midpoint_y = (thumb_tip[1] + index_tip[1]) // 2
        cv2.circle(img, (midpoint_x, midpoint_y), 25, (0, 165, 255), -1)

        dist_thumb = math.hypot(thumb_tip[0] - midpoint_x, thumb_tip[1] - midpoint_y)
        dist_index = math.hypot(index_tip[0] - midpoint_x, index_tip[1] - midpoint_y)

        is_over_image = (image_x_offset <= midpoint_x <= image_x_offset + overlay_resized.shape[1] and
                         image_y_offset <= midpoint_y <= image_y_offset + overlay_resized.shape[0])

        if dist_thumb <= 25 and dist_index <= 25 and is_over_image:
            is_dragging = True
        else:
            if is_dragging:
                image_x_offset = midpoint_x - overlay_resized.shape[1] // 2
                image_y_offset = midpoint_y - overlay_resized.shape[0] // 2
            is_dragging = False

    # Overlay image with transparency
    try:
        for c in range(3):
            img[image_y_offset:image_y_offset + overlay_resized.shape[0],
                image_x_offset:image_x_offset + overlay_resized.shape[1], c] = \
                overlay_resized[:, :, c] * (overlay_resized[:, :, 3] / 255.0) + \
                img[image_y_offset:image_y_offset + overlay_resized.shape[0],
                    image_x_offset:image_x_offset + overlay_resized.shape[1], c] * (1.0 - overlay_resized[:, :, 3] / 255.0)
    except:
        pass  # In case the image goes out of frame

    # Show and save
    cv2.imshow("Hand Tracking", img)
    output_video.write(img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
output_video.release()
cv2.destroyAllWindows()
