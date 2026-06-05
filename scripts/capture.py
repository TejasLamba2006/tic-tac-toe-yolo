import cv2
import os
from datetime import datetime

# Create output directory
SAVE_DIR = "dataset"
os.makedirs(SAVE_DIR, exist_ok=True)

camera = cv2.VideoCapture(1)
# camera.release()
camera.set(cv2.CAP_PROP_AUTOFOCUS, 0)
# camera.set(cv2.CAP_PROP_FOCUS, 0)
if camera is None:
    print("[-] No camera found")
    exit()

print("\nControls:")
print("  S = Save Image")
print("  Q = Quit\n")

img_count = 0

while True:
    ret, frame = camera.read()

    if not ret:
        print("[-] Failed to grab frame")
        break

    cv2.imshow("Dataset Capture", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = os.path.join(
            SAVE_DIR,
            f"ball_{timestamp}.jpg"
        )

        cv2.imwrite(filename, frame)
        img_count += 1

        print(f"[+] Saved: {filename}")
        print(f"    Total Images: {img_count}")

    elif key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()

print(f"\nFinished. Captured {img_count} images.")
