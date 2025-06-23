import cv2
from deepface import DeepFace
import os

# Path to the database of known criminals
criminal_db_path = "D:\criminals"

# Verify that the folder exists
if not os.path.exists(criminal_db_path):
    print(f"[ERROR] Criminal database path '{criminal_db_path}' not found.")
    exit()

# Start webcam
cap = cv2.VideoCapture(0)
print("[INFO] Starting camera...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to capture frame.")
        break

    # Save the current frame temporarily (DeepFace requires image path or BGR array)
    try:
        result = DeepFace.find(
            img_path=frame,
            db_path=criminal_db_path,
            enforce_detection=False,
            detector_backend='opencv'  # You can change to 'retinaface', 'mtcnn', etc.
        )

        if len(result) > 0 and len(result[0]) > 0:
            # Extract matched identity
            identity_path = result[0].iloc[0]['identity']
            name = os.path.basename(identity_path)
            cv2.putText(frame, f"Match: {name}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "No Match", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    except Exception as e:
        print(f"[ERROR] {e}")

    # Display the frame
    cv2.imshow("Criminal Face Detection", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
