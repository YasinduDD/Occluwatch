import os
import cv2
import torch
import numpy as np
from PIL import Image
from fastreid.config import get_cfg
from fastreid.engine import DefaultPredictor
from fastreid.utils.visualizer import Visualizer
from fastreid.data import build_reid_test_loader
from fastreid.utils.checkpoint import Checkpointer
from fastreid.data.transforms import build_transforms
from ultralytics import YOLO
from collections import defaultdict

# Initialize YOLO model
yolo_model = YOLO("Model/best.pt")


class VehicleReID:
    def __init__(self, config_path, model_path):
        # Load configuration
        cfg = get_cfg()
        cfg.merge_from_file(config_path)
        cfg.MODEL.WEIGHTS = model_path
        cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        # Create predictor and transforms
        self.predictor = DefaultPredictor(cfg)
        self.cfg = cfg
        self.transforms = build_transforms(cfg, is_train=False)

    def extract_features(self, image, bboxes):
        """Extract features for vehicles in an image"""
        features = []
        for bbox in bboxes:
            x1, y1, x2, y2 = map(int, bbox)
            vehicle_img = image[y1:y2, x1:x2]

            if vehicle_img.size == 0:
                continue

            # Convert BGR to RGB and then to PIL Image
            vehicle_img = cv2.cvtColor(vehicle_img, cv2.COLOR_BGR2RGB)
            vehicle_img = Image.fromarray(vehicle_img)

            # Apply transforms and convert to tensor
            vehicle_img = self.transforms(vehicle_img)
            vehicle_img = torch.unsqueeze(vehicle_img, 0).to(self.cfg.MODEL.DEVICE)

            # Predict features
            with torch.no_grad():
                outputs = self.predictor(vehicle_img)
            features.append(outputs.cpu().numpy())

        return np.array(features) if features else None

    def compute_similarity(self, features1, features2):
        """Compute cosine similarity between two sets of features"""
        if features1 is None or features2 is None:
            return None

        # Reshape features to 2D arrays (n_samples, n_features)
        features1 = features1.reshape(features1.shape[0], -1)
        features2 = features2.reshape(features2.shape[0], -1)

        # Normalize features
        features1 = features1 / np.linalg.norm(features1, axis=1, keepdims=True)
        features2 = features2 / np.linalg.norm(features2, axis=1, keepdims=True)

        # Compute similarity matrix
        similarity = np.dot(features1, features2.T)
        return similarity


def detect_vehicles(image):
    """Detect vehicles using YOLO"""
    results = yolo_model(image)[0]
    boxes = []
    for box in results.boxes:
        cls_id = int(box.cls)
        # COCO classes: car=2, truck=7, bus=5, motorcycle=3
        if cls_id in [2, 3, 5, 7]:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            boxes.append([x1, y1, x2, y2])
    return boxes


def process_video(video_path, reid_system, frame_interval=10):
    """Process a video and extract vehicle features"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None, None

    frame_count = 0
    all_features = []
    all_boxes = []
    all_frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_interval != 0:
            continue

        # Detect vehicles
        boxes = detect_vehicles(frame)
        if not boxes:
            continue

        # Extract features
        features = reid_system.extract_features(frame, boxes)
        if features is None:
            continue

        all_features.append(features)
        all_boxes.append(boxes)
        all_frames.append(frame)

    cap.release()

    if not all_features:
        return None, None, None

    # Combine all features and boxes
    features = np.concatenate(all_features)
    boxes = np.concatenate(all_boxes)
    return features, boxes, all_frames


def find_matches(features1, features2, threshold=0.7):
    """Find matching vehicles between two sets of features"""
    # Reshape features to 2D arrays (n_samples, n_features)
    features1 = features1.reshape(features1.shape[0], -1)  # Shape: (n, 2048)
    features2 = features2.reshape(features2.shape[0], -1)  # Shape: (m, 2048)

    # Normalize features
    features1 = features1 / np.linalg.norm(features1, axis=1, keepdims=True)
    features2 = features2 / np.linalg.norm(features2, axis=1, keepdims=True)

    # Compute similarity matrix
    similarity = np.dot(features1, features2.T)  # Shape: (n, m)

    matches = []
    for i in range(similarity.shape[0]):
        best_match = np.argmax(similarity[i])
        score = similarity[i][best_match]
        if score > threshold:
            matches.append((i, best_match, score))

    return matches


def visualize_matches(video1_frames, video2_frames, boxes1, boxes2, matches, output_path="output.mp4"):
    """Create a video showing the matching vehicles"""
    if not video1_frames or not video2_frames:
        print("No frames to visualize")
        return

    # Get frame dimensions (use first frame)
    h1, w1 = video1_frames[0].shape[:2]
    h2, w2 = video2_frames[0].shape[:2]
    max_height = max(h1, h2)

    # Resize frames to have same height
    def resize_frame(frame, target_height):
        h, w = frame.shape[:2]
        return cv2.resize(frame, (int(w * target_height / h), target_height))

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 10.0, (w1 + w2, max_height))

    # Create visualization
    for frame1, frame2 in zip(video1_frames, video2_frames):
        frame1 = resize_frame(frame1.copy(), max_height)
        frame2 = resize_frame(frame2.copy(), max_height)

        # Draw boxes on frame1
        for i, box in enumerate(boxes1):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame1, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame1, f"V{i + 1}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw boxes on frame2
        for j, box in enumerate(boxes2):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame2, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame2, f"V{j + 1}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw match lines
        for i, j, score in matches:
            # Calculate center points for both boxes
            pt1_x = int((boxes1[i][0] + boxes1[i][2]) // 2)
            pt1_y = int((boxes1[i][1] + boxes1[i][3]) // 2)

            pt2_x = int((boxes2[j][0] + boxes2[j][2]) // 2) + w1  # Add w1 to account for frame1 width
            pt2_y = int((boxes2[j][1] + boxes2[j][3]) // 2)

            # Draw line between matched vehicles
            cv2.line(frame1, (pt1_x, pt1_y), (pt2_x, pt2_y), (0, 0, 255), 2)

            # Put similarity score text
            mid_x = (pt1_x + pt2_x) // 2
            mid_y = (pt1_y + pt2_y) // 2
            cv2.putText(frame1, f"{score:.2f}", (mid_x, mid_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        combined = np.concatenate((frame1, frame2), axis=1)
        out.write(combined)

    out.release()
    print(f"Saved visualization to {output_path}")


def main(video_path1, video_path2, config_path, model_path):
    # Initialize ReID system
    reid_system = VehicleReID(config_path, model_path)

    # Process both videos
    print(f"Processing {video_path1}...")
    features1, boxes1, frames1 = process_video(video_path1, reid_system)

    print(f"Processing {video_path2}...")
    features2, boxes2, frames2 = process_video(video_path2, reid_system)

    if features1 is None or features2 is None:
        print("No vehicles detected in one or both videos")
        return

    # Find matches
    matches = find_matches(features1, features2, threshold=0.6)

    # Print results
    print("\nVehicle Re-identification Results:")
    print(f"Found {len(matches)} matching vehicles between the videos")
    for i, j, score in matches:
        print(f"Vehicle {i + 1} in Video1 matches Vehicle {j + 1} in Video2 with similarity {score:.3f}")

    # Create visualization
    visualize_matches(frames1, frames2, boxes1, boxes2, matches)


if __name__ == "__main__":
    # Paths to your files
    CONFIG_PATH = "configs/VeRi/sbs_R50-ibn.yml"
    MODEL_PATH = "Model/veri_sbs_R50-ibn.pth"  # Update with your model filename
    VIDEO_PATH1 = "test/1.mp4"  # Update with your video paths
    VIDEO_PATH2 = "test/1_2.mp4"

    main(VIDEO_PATH1, VIDEO_PATH2, CONFIG_PATH, MODEL_PATH)