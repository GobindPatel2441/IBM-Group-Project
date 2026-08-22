import cv2
from deepface import DeepFace
import logging

# Disable TensorFlow logging for a cleaner console
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def start_emotion_tracker():
    """
    Starts a real-time webcam feed that detects and overlays facial emotions.
    """
    # Initialize the webcam (usually 0 is the default camera)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam. Please check your connection.")
        return

    print("--- Emotion Tracker Started ---")
    print("Instructions:")
    print("1. Look directly at the camera.")
    print("2. The script will detect your face and analyze emotions in real-time.")
    print("3. Press 'q' to quit.")

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Exiting...")
            break

        try:
            # analyze() returns a list of results (one per face)
            # actions=['emotion'] specifies we only care about emotions
            # enforce_detection=False prevents the code from crashing if no face is found
            results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

            for res in results:
                # Get bounding box coordinates
                region = res['region']
                x, y, w, h = region['x'], region['y'], region['w'], region['h']

                # Get the dominant emotion
                dominant_emotion = res['dominant_emotion']
                confidence = res['emotion'][dominant_emotion]

                # Draw rectangle around the face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Overlay the emotion text
                label = f"{dominant_emotion} ({confidence:.1f}%)"
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        except Exception as e:
            # Minor errors during analysis (like rapid movements) are skipped
            pass

        # Display the resulting frame
        cv2.imshow('Facial Emotion Detector - Press Q to Quit', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    print("--- Emotion Tracker Closed ---")

if __name__ == "__main__":
    start_emotion_tracker()
