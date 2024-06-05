import cv2
import torch
import numpy as np
from PyQt5.QtWidgets import QMessageBox

class CameraHandler:
    def __init__(self, model):
        self.model = model
        self.model_loaded = model is not None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.model_loaded:
            self.model.to(self.device)
            self.model.eval()

    def open_camera(self):
        # Check if the model is loaded
        if not self.model_loaded:
            QMessageBox.critical(None, "Error", "No model loaded. Please load a model first.")
            return

        # Setup the OpenCV window
        cv_window_name = 'Press "c" to capture or "q" to quit'
        cv2.namedWindow(cv_window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(cv_window_name, 640, 480)

        # Open the camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            QMessageBox.critical(None, "Error", "Failed to open the camera.")
            return

        # Capture video frames
        while True:
            ret, frame = cap.read()
            if not ret:
                QMessageBox.warning(None, "Warning", "Failed to capture video from camera.")
                break

            cv2.imshow(cv_window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                # Get bounding box, crop, preprocess, and predict hand sign
                bbox = self.get_bounding_box(frame)
                cropped_frame = self.crop_to_bbox(frame, bbox)
                processed_frame = self.preprocess_frame(cropped_frame)
                label = self.predict_hand_sign(processed_frame)
                self.display_prediction(frame, label, bbox)
                if cv2.waitKey(0) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyWindow(cv_window_name)

    def get_bounding_box(self, frame):
        # Calculate a bounding box at the center of the frame
        height, width, _ = frame.shape
        box_size = int(min(height, width) * 0.5)
        x_center, y_center = width // 2, height // 2
        return (x_center - box_size // 2, y_center - box_size // 2, box_size, box_size)

    def crop_to_bbox(self, frame, bbox):
        # Crop the frame to the bounding box
        x, y, w, h = bbox
        return frame[y:y+h, x:x+w]

    def preprocess_frame(self, frame):
        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Resize to model input size
        resized = cv2.resize(gray, (28, 28))  # Adjust according to the model input size
        # Normalize the image
        normalized = resized / 255.0
        # Reshape if necessary for the model input
        reshaped = normalized.reshape(1, 1, 28, 28)

        # Debug: Print shapes and types of processed frame
        print(f"Preprocessed Frame Shape: {reshaped.shape}, Type: {reshaped.dtype}")
        return reshaped.astype(np.float32)  # Convert to float32 for the model

    def predict_hand_sign(self, image):
        # Convert image to a tensor and send to device
        image_tensor = torch.tensor(image, dtype=torch.float).to(self.device)
        with torch.no_grad():
            # Get model output
            output = self.model(image_tensor)
            _, predicted = torch.max(output, 1)

            # Debug: Print output tensor and predicted label
            print(f"Model Output: {output}, Predicted: {predicted}")
            return predicted.item()

    def display_prediction(self, frame, label, bbox):
        # Draw bounding box and label on the frame
        x, y, w, h = bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f'Label: {label}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        # Show the frame with prediction
        cv2.imshow("Prediction", frame)
        cv2.waitKey(0)  # Wait for key press
        cv2.destroyAllWindows()  # Ensure to destroy all windows to prevent hanging
