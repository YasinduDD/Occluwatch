import os
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
from model.make_model import make_model
from config import cfg
from torch.nn import functional as F


# Use YOLOv8s for vehicle detection
yolo_model = YOLO("yolov8s.pt")


def build_transforms(is_train=False):
    """Basic ImageNet-style transform for inference."""
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transform


class VehicleReID:
    def __init__(self, config_path, model_path):
        # Load config
        cfg.merge_from_file(config_path)
        cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        cfg.freeze()

        # Load model
        self.model = make_model(cfg, num_class=576, camera_num=20, view_num=8)
        state_dict = torch.load(model_path, map_location="cpu")
        if "model" in state_dict:
            self.model.load_state_dict(state_dict["model"], strict=False)
        else:
            self.model.load_state_dict(state_dict, strict=False)
        self.model.to(cfg.MODEL.DEVICE)
        self.model.eval()

        # Preprocessing
        self.transforms = build_transforms(is_train=False)
        self.device = cfg.MODEL.DEVICE

    def extract_features(self, image, bboxes):
        features = []
        for bbox in bboxes:
            x1, y1, x2, y2 = map(int, bbox)
            vehicle_img = image[y1:y2, x1:x2]

            if vehicle_img.size == 0:
                continue

            vehicle_img = cv2.cvtColor(vehicle_img, cv2.COLOR_BGR2RGB)
            vehicle_img = Image.fromarray(vehicle_img)
            vehicle_img = self.transforms(vehicle_img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                # Use cam_label=0, view_label=0 for all, or set appropriately if you have this info
                feat = self.model(
                    vehicle_img,
                    cam_label=torch.zeros(1, dtype=torch.long, device=self.device),
                    view_label=torch.zeros(1, dtype=torch.long, device=self.device),
                )
                feat = F.normalize(feat, p=2, dim=1)

            features.append(feat.cpu().numpy())

        return np.array(features) if features else None

    def compute_similarity(self, features1, features2):
        if features1 is None or features2 is None:
            return None

        features1 = features1.reshape(features1.shape[0], -1)
        features2 = features2.reshape(features2.shape[0], -1)

        return np.dot(features1, features2.T)


def detect_vehicles(image):
    results = yolo_model(image)[0]
    boxes = []
    for box in results.boxes:
        cls_id = int(box.cls)
        if cls_id in [2, 5, 7]:  # car, bus, truck
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            boxes.append([x1, y1, x2, y2])
    return boxes


def process_image(image_path, reid_system):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return None, None

    boxes = detect_vehicles(image)
    features = reid_system.extract_features(image, boxes)

    return features, boxes


def main(image_path1, image_path2, config_path, model_path):
    reid_system = VehicleReID(config_path, model_path)

    print(f"Processing {image_path1}...")
    features1, boxes1 = process_image(image_path1, reid_system)

    print(f"Processing {image_path2}...")
    features2, boxes2 = process_image(image_path2, reid_system)

    if features1 is None or features2 is None:
        print("No vehicles detected in one or both images")
        return

    similarity = reid_system.compute_similarity(features1, features2)
    threshold = 0.7
    matches = similarity > threshold

    print("\nVehicle Re-identification Results:")
    print(f"Found {matches.sum()} matching vehicles between the images")

    for i in range(similarity.shape[0]):
        best_match = np.argmax(similarity[i])
        print(
            f"Vehicle {i + 1} in Image1 best matches Vehicle {best_match + 1} in Image2 with similarity {similarity[i][best_match]:.3f}"
        )

    visualize_results(image_path1, image_path2, boxes1, boxes2, similarity)


def visualize_results(image_path1, image_path2, boxes1, boxes2, similarity):
    img1 = cv2.imread(image_path1)
    img2 = cv2.imread(image_path2)

    for i, box in enumerate(boxes1):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img1, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img1,
            f"V{i + 1}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    for j, box in enumerate(boxes2):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img2, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img2,
            f"V{j + 1}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    target_height = max(img1.shape[0], img2.shape[0])

    def resize_keep_aspect(img, target_height):
        scale = target_height / img.shape[0]
        new_width = int(img.shape[1] * scale)
        return cv2.resize(img, (new_width, target_height))

    img1 = resize_keep_aspect(img1, target_height)
    img2 = resize_keep_aspect(img2, target_height)

    max_width = max(img1.shape[1], img2.shape[1])

    def pad_to_width(img, target_width):
        pad = target_width - img.shape[1]
        return (
            cv2.copyMakeBorder(img, 0, 0, 0, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            if pad > 0
            else img
        )

    img1 = pad_to_width(img1, max_width)
    img2 = pad_to_width(img2, max_width)

    combined = np.hstack((img1, img2))

    def scale_down(img, max_width=2000):
        if img.shape[1] > max_width:
            scale = max_width / img.shape[1]
            new_size = (int(img.shape[1] * scale), int(img.shape[0] * scale))
            return cv2.resize(img, new_size)
        return img

    combined = scale_down(combined)

    for i in range(similarity.shape[0]):
        best_match = np.argmax(similarity[i])
        score = similarity[i][best_match]
        cv2.putText(
            combined,
            f"V{i+1} <-> V{best_match+1}: {score:.2f}",
            (10, 30 + i * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

    cv2.imwrite("vehicle_reid_comparison.png", combined)
    print("Result image saved as vehicle_reid_comparison.png")


if __name__ == "__main__":
    CONFIG_PATH = "transreid/configs/VeRi/deit_transreid_stride.yml"
    MODEL_PATH = "transreid/deit_transreid_veri.pth"
    IMAGE_PATH1 = "transreid/test/7_1.png"
    IMAGE_PATH2 = "transreid/test/7_2.png"

    main(IMAGE_PATH1, IMAGE_PATH2, CONFIG_PATH, MODEL_PATH)
