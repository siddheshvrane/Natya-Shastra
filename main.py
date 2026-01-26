import cv2
import mediapipe as mp
import numpy as np
import sys
from visualizer import Visualizer

# MediaPipe Tasks API
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

def main(video_path=None):
    # Initialize MediaPipe Pose Landmarker
    # We use VIDEO mode for tracking continuity
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path='pose_landmarker_full.task'),
        running_mode=VisionRunningMode.VIDEO)
    
    detector = PoseLandmarker.create_from_options(options)

    # Input video
    if video_path:
        cap = cv2.VideoCapture(video_path)
    else:
        # Default to webcam if no video provided
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30

    # Output video writer
    # We'll save to 'output.mp4'
    # 'avc1' (H.264) is more compatible with mobile devices/WhatsApp than 'mp4v'
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))
    
    if not out.isOpened():
        print("Warning: 'avc1' codec failed. Falling back to 'mp4v'.")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

    visualizer = Visualizer(width, height)

    print("Processing video... Press 'q' to quit.")
    
    frame_timestamp_ms = 0
    
    try:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break

            # Convert the BGR image to RGB for MediaPipe
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            
            # Increment timestamp for VIDEO mode
            # MediaPipe expects strictly increasing timestamps in ms
            frame_timestamp_ms += int(1000 / fps)
            
            # Detect landmarks
            detection_result = detector.detect_for_video(mp_image, frame_timestamp_ms)

            # Draw effects
            # Darken the original image slightly to make the white lines pop
            image = cv2.addWeighted(image, 0.7, np.zeros(image.shape, image.dtype), 0, 0)

            if detection_result.pose_landmarks:
                # visualizer.update expects a list of landmarks for one person
                # We take the first detected person
                visualizer.update(detection_result.pose_landmarks[0])
                image = visualizer.draw(image)

            out.write(image)
            cv2.imshow('Dance Math Visualizer', image)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
    finally:
        # Ensure resources are released even if an error occurs
        print("releasing resources...")
        detector.close()
        cap.release()
        out.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # You can pass a video path as an argument
    video_file = sys.argv[1] if len(sys.argv) > 1 else None
    main(video_file)
