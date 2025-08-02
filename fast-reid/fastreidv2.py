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
    """Placeholder for vehicle detection - replace with your detector"""
    # This is a dummy function - in practice, use YOLO, Faster R-CNN, etc.
    # Returns list of bounding boxes in format [x1, y1, x2, y2]
    # For now, we'll just return some hardcoded boxes for demonstration
    height, width = image.shape[:2]
    return [[100, 100, 200, 200], [300, 150, 400, 250]]  # Example boxes


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
    """Visualize the comparison results with bounding boxes and similarity scores"""
    img1 = cv2.imread(image_path1)
    img2 = cv2.imread(image_path2)

    # Draw boxes on image1
    for i, box in enumerate(boxes1):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img1, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img1, f"V{i + 1}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Draw boxes on image2
    for j, box in enumerate(boxes2):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img2, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img2, f"V{j + 1}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Resize images to have the same height (use the larger height)
    max_height = max(img1.shape[0], img2.shape[0])
    img1 = cv2.resize(img1, (int(img1.shape[1] * max_height / img1.shape[0]), max_height))
    img2 = cv2.resize(img2, (int(img2.shape[1] * max_height / img2.shape[0]), max_height))

    # Combine images
    combined = np.concatenate((img1, img2), axis=1)

    # Display similarity scores
    for i in range(similarity.shape[0]):
        best_match = np.argmax(similarity[i])
        score = similarity[i][best_match]
        cv2.putText(combined,
                    f"V{i + 1} <-> V{best_match + 1}: {score:.2f}",
                    (10, 30 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2)

    cv2.imshow("Vehicle ReID Comparison", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Paths to your files
    CONFIG_PATH = "configs/VeRi/sbs_R50-ibn.yml"
    MODEL_PATH = "Model/veri_sbs_R50-ibn.pth"  # Update with your model filename
    IMAGE_PATH1 = "test/1.jpg"  # Update with your image paths
    IMAGE_PATH2 = "test/1_1.jpg"

    main(IMAGE_PATH1, IMAGE_PATH2, CONFIG_PATH, MODEL_PATH)