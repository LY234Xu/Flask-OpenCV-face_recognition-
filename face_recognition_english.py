# Import required libraries
from picamera2 import Picamera2
import cv2
import numpy as np
import face_recognition
import os
import time
import sys
import webbrowser

# Load reference images and encode faces
def load_reference_image(path):
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Failed to load image: {path}")
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb)
    if not face_locations:
        raise ValueError("No face detected in reference image")
    return face_recognition.face_encodings(rgb, face_locations)[0]

# Initialize camera
def init_camera():
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={
            "size": (640, 480),  # Camera resolution
            "format": "BGR888"
        },
        controls={
            "FrameDurationLimits": (33333, 66666),  # 30fps
            "AwbMode": 0,  # Auto white balance
            "ExposureTime": 20000  # 20ms exposure
        }
    )
    picam2.configure(config)
    picam2.start()
    return picam2

# Save new face image and update known encodings
def save_new_face(frame, face_location, name):
    top, right, bottom, left = face_location
    face_image = frame[top:bottom, left:right]
    # Convert BGR to RGB
    face_image_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    # Create a directory to save face images if it doesn't exist
    if not os.path.exists('faces'):
        os.makedirs('faces')
    image_path = os.path.join('faces', f'{name}.jpg')
    cv2.imwrite(image_path, face_image_rgb)
    new_encoding = load_reference_image(image_path)
    return new_encoding

# Load all known faces from the faces directory
def load_all_known_faces():
    known_encodings = []
    known_names = []
    if os.path.exists('faces'):
        for filename in os.listdir('faces'):
            if filename.endswith('.jpg'):
                name = os.path.splitext(filename)[0]
                image_path = os.path.join('faces', filename)
                try:
                    encoding = load_reference_image(image_path)
                    known_encodings.append(encoding)
                    known_names.append(name)
                except Exception as e:
                    print(f"Error loading {image_path}: {str(e)}")
    return known_encodings, known_names

# Main program
def main():
    if len(sys.argv) != 2:
        print("Please provide a mode (1 or 2) as a command-line argument.")
        return
    choice = sys.argv[1]

    # Load known faces
    try:
        known_encodings, known_names = load_all_known_faces()
    except Exception as e:
        print(f"Initialization failed: {str(e)}")
        return

    # Initialize the camera
    camera = init_camera()

    # Display parameters
    SCALE_FACTOR = 0.5
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    THRESHOLD = 0.5

    # Timer settings
    TIMEOUT = 3  # Waiting time to ask for saving (seconds)
    unknown_start_time = None

    try:
        while True:
            try:
                # Capture a frame
                frame = camera.capture_array()
            except Exception as e:
                print(f"Error capturing frame: {e}")
                continue
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Preprocess
            small_frame = cv2.resize(
                frame_rgb,
                (0, 0),
                fx=SCALE_FACTOR,
                fy=SCALE_FACTOR,
                interpolation=cv2.INTER_AREA
            )

            # Face detection
            face_locations = face_recognition.face_locations(small_frame)
            try:
                face_encodings = face_recognition.face_encodings(small_frame, face_locations)
            except Exception as e:
                print(f"Error encoding faces: {e}")
                continue

            # Recognition processing
            all_unknown = True
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                # Restore original coordinates
                top = int(top / SCALE_FACTOR)
                right = int(right / SCALE_FACTOR)
                bottom = int(bottom / SCALE_FACTOR)
                left = int(left / SCALE_FACTOR)

                # Calculate matching distances
                distances = face_recognition.face_distance(known_encodings, face_encoding)
                if len(distances) > 0:
                    min_distance = np.min(distances)
                    match_index = np.argmin(distances)
                    # Determine identity
                    name = "Unknown"
                    color = (0, 0, 255)  # Red
                    if min_distance <= THRESHOLD:
                        name = known_names[match_index]
                        color = (0, 255, 0)  # Green
                        all_unknown = False
                else:
                    name = "Unknown"
                    color = (0, 0, 255)
                    min_distance = float('nan')

                # Draw bounding box and label
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                text = f"{name} ({min_distance:.2f})"
                cv2.putText(frame, text, (left + 6, bottom - 6),
                            FONT, 0.5, color, 1)

            # Handle unknown faces
            if choice == '2' and all_unknown:
                if unknown_start_time is None:
                    unknown_start_time = time.time()
                elif time.time() - unknown_start_time >= TIMEOUT:
                    # 打开保存人脸的网页，并传递当前帧和人脸位置信息
                    # 这里简单假设第一个检测到的人脸为要保存的人脸
                    if face_locations:
                        face_location = (
                            int(face_locations[0][0] / SCALE_FACTOR),
                            int(face_locations[0][1] / SCALE_FACTOR),
                            int(face_locations[0][2] / SCALE_FACTOR),
                            int(face_locations[0][3] / SCALE_FACTOR)
                        )
                        # 保存当前帧
                        cv2.imwrite('temp_frame.jpg', frame)
                        # 保存人脸位置
                        with open('temp_face_location.txt', 'w') as f:
                            f.write(f"{face_location[0]},{face_location[1]},{face_location[2]},{face_location[3]}")
                        # 打开网页
                        webbrowser.open('http://127.0.0.1:5000/save_face')
                    unknown_start_time = None
            else:
                unknown_start_time = None

            # Display output
            cv2.imshow('Face Recognition', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            # Exit condition
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("User terminated the program")
                break

    finally:
        # Clean up resources
        camera.stop()
        camera.close()
        cv2.destroyAllWindows()
        print("System resources have been released")

if __name__ == "__main__":
    main()
    