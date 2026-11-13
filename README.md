# Hand-tracking image-overlay experiments

A collection of webcam prototypes using hand landmarks to interact with image overlays. The repository name mentions chess, but the code is a set of visual experiments rather than a complete chess game.

## Files

- `pick.py`, `improve.py`, and `improve_saving.py` explore overlays using chess-related images; the saving variant records video.
- `picking_apple.py` and `picking_apple_saving.py` explore an apple overlay; the saving variant records video.
- `apple1.png`, `queen.png`, `bo.png`, and `board.png` are image assets.

## Requirements and limitations

The scripts use Python, OpenCV, cvzone hand tracking, and NumPy, and require a working webcam. Several scripts contain absolute Windows paths for image inputs or video output, so they are not portable as-is. The correct asset and output locations depend on each script. No dependency manifest or runtime validation is provided.

Existing names and paths are preserved to avoid changing prototype behavior.
