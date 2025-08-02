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

yolo_model = YOLO("yolov8s.pt")

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
    results = yolo_model(image)[0]
    boxes = []
    for box in results.boxes:
        cls_id = int(box.cls)
        # COCO classes: car=2, truck=7, bus=5, motorcycle=3
        if cls_id in [2, 5, 7]:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            boxes.append([x1, y1, x2, y2])
    return boxes



def process_image(image_path, reid_system):
    """Process an image and extract vehicle features"""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return None, None

    # Detect vehicles
    boxes = detect_vehicles(image)

    # Extract features
    features = reid_system.extract_features(image, boxes)

    return features, boxes


def main(image_path1, image_path2, config_path, model_path):
    # Initialize ReID system
    reid_system = VehicleReID(config_path, model_path)

    # Process both images
    print(f"Processing {image_path1}...")
    features1, boxes1 = process_image(image_path1, reid_system)

    print(f"Processing {image_path2}...")
    features2, boxes2 = process_image(image_path2, reid_system)

    if features1 is None or features2 is None:
        print("No vehicles detected in one or both images")
        return

    # Compute similarity
    similarity = reid_system.compute_similarity(features1, features2)

    # Find matches (simple threshold approach)
    threshold = 0.7  # Adjust based on your needs
    matches = similarity > threshold

    # Print results
    print("\nVehicle Re-identification Results:")
    print(f"Found {matches.sum()} matching vehicles between the images")

    # For each vehicle in image1, find matches in image2
    for i in range(similarity.shape[0]):
        best_match = np.argmax(similarity[i])
        print(
            f"Vehicle {i + 1} in Image1 best matches Vehicle {best_match + 1} in Image2 with similarity {similarity[i][best_match]:.3f}"
        )

    # Visualize results (optional)
    visualize_results(image_path1, image_path2, boxes1, boxes2, similarity)


def visualize_results(image_path1, image_path2, boxes1, boxes2, similarity):
    img1 = cv2.imread(image_path1)
    img2 = cv2.imread(image_path2)

    for i, box in enumerate(boxes1):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img1, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img1, f"V{i + 1}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    for j, box in enumerate(boxes2):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img2, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img2, f"V{j + 1}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Resize both to same height
    target_height = max(img1.shape[0], img2.shape[0])
    def resize_keep_aspect(img, target_height):
        scale = target_height / img.shape[0]
        new_width = int(img.shape[1] * scale)
        return cv2.resize(img, (new_width, target_height))

    img1 = resize_keep_aspect(img1, target_height)
    img2 = resize_keep_aspect(img2, target_height)

    # Pad to equal width
    max_width = max(img1.shape[1], img2.shape[1])
    def pad_to_width(img, target_width):
        pad = target_width - img.shape[1]
        return cv2.copyMakeBorder(img, 0, 0, 0, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0)) if pad > 0 else img

    img1 = pad_to_width(img1, max_width)
    img2 = pad_to_width(img2, max_width)

    # Stack left-right
    combined = np.hstack((img1, img2))

    # Scale down for viewing
    def scale_down(img, max_width=2000):
        if img.shape[1] > max_width:
            scale = max_width / img.shape[1]
            new_size = (int(img.shape[1] * scale), int(img.shape[0] * scale))
            return cv2.resize(img, new_size)
        return img

    combined = scale_down(combined)

    # Draw similarity scores
    for i in range(similarity.shape[0]):
        best_match = np.argmax(similarity[i])
        score = similarity[i][best_match]
        cv2.putText(combined, f"V{i+1} <-> V{best_match+1}: {score:.2f}", (10, 30 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imwrite("Vehicle ReID Comparison.png", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()





if __name__ == "__main__":
    # Paths to your files
    CONFIG_PATH = "configs/VeRi/sbs_R50-ibn.yml"
    MODEL_PATH = "Model/veri_sbs_R50-ibn.pth"  # Update with your model filename
    IMAGE_PATH1 = "test/7_1.png"  # Update with your image paths
    IMAGE_PATH2 = "test/7_2.png"

    main(IMAGE_PATH1, IMAGE_PATH2, CONFIG_PATH, MODEL_PATH)