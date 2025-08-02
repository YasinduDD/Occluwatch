# pipeline without violations

import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
import supervision as sv
from collections import defaultdict, deque
import time
import pickle
from model.make_model import make_model
from config import cfg as transreid_cfg
from torch.nn import functional as F


# Define transforms for TransReID
def build_transforms():
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transform


class VehicleTracker:
    def __init__(self, yolo_model_path, reid_config_path, reid_model_path):
        # Initialize YOLO with built-in tracking
        self.yolo_model = YOLO(yolo_model_path)
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()

        # Initialize TransReID
        self.reid_cfg = self._init_reid_config(reid_config_path)
        self.model = make_model(self.reid_cfg, num_class=576, camera_num=20, view_num=8)
        self._load_model_weights(reid_model_path)
        self.transforms = build_transforms()

        # Tracking storage
        self.tracks = {}  # {track_id: track_info}
        self.next_id = 1
        self.vehicle_features = defaultdict(list)  # {vehicle_id: [features]}
        self.feature_history = defaultdict(deque)  # {vehicle_id: deque of features}

        # Config
        self.CAR_CLASS_IDS = [2, 5, 7]  # COCO vehicle classes
        self.REID_THRESHOLD = 0.8  # Similarity threshold for matching
        self.FEATURE_VERIFICATION_INTERVAL = 30  # Verify IDs every 30 frames
        self.MAX_FRAMES_SINCE_SEEN = 100  # Forget tracks after 100 frames
        self.MAX_FEATURE_HISTORY = 15  # Keep last 5 features for each vehicle

    def _init_reid_config(self, config_path):
        cfg = transreid_cfg.clone()
        cfg.merge_from_file(config_path)
        cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        cfg.freeze()
        return cfg

    def _load_model_weights(self, model_path):
        state_dict = torch.load(model_path, map_location="cpu")
        if "model" in state_dict:
            self.model.load_state_dict(state_dict["model"], strict=False)
        else:
            self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.reid_cfg.MODEL.DEVICE)
        self.model.eval()

    def extract_features(self, image, detections):
        """Extract ReID features for all detections"""
        features = []
        for bbox in detections.xyxy:
            x1, y1, x2, y2 = map(int, bbox)
            vehicle_img = image[y1:y2, x1:x2]
            if vehicle_img.size == 0:
                features.append(None)
                continue

            vehicle_img = cv2.cvtColor(vehicle_img, cv2.COLOR_BGR2RGB)
            vehicle_img = Image.fromarray(vehicle_img)
            vehicle_img = self.transforms(vehicle_img)
            vehicle_img = torch.unsqueeze(vehicle_img, 0).to(self.reid_cfg.MODEL.DEVICE)

            with torch.no_grad():
                outputs = self.model(
                    vehicle_img,
                    cam_label=torch.zeros(
                        1, dtype=torch.long, device=self.reid_cfg.MODEL.DEVICE
                    ),
                    view_label=torch.zeros(
                        1, dtype=torch.long, device=self.reid_cfg.MODEL.DEVICE
                    ),
                )
                outputs = F.normalize(outputs, p=2, dim=1)
            features.append(outputs.cpu().numpy().flatten())
        return features

    def cosine_similarity(self, feat1, feat2):
        """Compute cosine similarity between two features"""
        feat1_norm = feat1 / (np.linalg.norm(feat1) + 1e-10)
        feat2_norm = feat2 / (np.linalg.norm(feat2) + 1e-10)
        return np.dot(feat1_norm, feat2_norm)

    def match_vehicle(self, feature, current_id=None):
        """Match a feature against existing vehicles"""
        best_match_id = None
        highest_similarity = -1

        if current_id is not None and current_id in self.feature_history:
            for hist_feat in self.feature_history[current_id]:
                similarity = self.cosine_similarity(feature, hist_feat)
                if similarity > highest_similarity:
                    highest_similarity = similarity
                    best_match_id = current_id

        for vehicle_id, feature_list in self.vehicle_features.items():
            if current_id is not None and vehicle_id == current_id:
                continue

            for existing_feat in feature_list:
                similarity = self.cosine_similarity(feature, existing_feat)
                if similarity > highest_similarity:
                    highest_similarity = similarity
                    best_match_id = vehicle_id

        if highest_similarity > self.REID_THRESHOLD:
            return best_match_id
        return None

    def update_tracks(self, detections, features, frame_num):
        """Update tracks using YOLO tracking and verify with ReID"""
        if not hasattr(detections, "tracker_id") or detections.tracker_id is None:
            detections.tracker_id = np.arange(len(detections.xyxy))

        current_ids = set()
        updated_tracks = {}

        for detection_idx, track_id in enumerate(detections.tracker_id):
            current_ids.add(track_id)

            if track_id in self.tracks:
                track = self.tracks[track_id]
            else:
                track = {
                    "id": self.next_id,
                    "features": [],
                    "first_seen": frame_num,
                    "last_seen": frame_num,
                    "confirmed": False,
                }
                self.next_id += 1

            track["last_seen"] = frame_num

            if features and features[detection_idx] is not None:
                feature = features[detection_idx]
                track["features"].append(feature)

                if frame_num % self.FEATURE_VERIFICATION_INTERVAL == 0:
                    matched_id = self.match_vehicle(feature, track["id"])
                    if matched_id is not None:
                        track["id"] = matched_id
                        track["confirmed"] = True
                    else:
                        track["confirmed"] = False

                if track["id"] not in self.feature_history:
                    self.feature_history[track["id"]] = deque(
                        maxlen=self.MAX_FEATURE_HISTORY
                    )
                self.feature_history[track["id"]].append(feature)

                if track["confirmed"]:
                    self.vehicle_features[track["id"]].append(feature)

            updated_tracks[track_id] = track

        for track_id in list(self.tracks.keys()):
            if track_id not in current_ids:
                if (
                    frame_num - self.tracks[track_id]["last_seen"]
                    <= self.MAX_FRAMES_SINCE_SEEN
                ):
                    updated_tracks[track_id] = self.tracks[track_id]

        self.tracks = updated_tracks
        return updated_tracks

    def process_frame(self, frame, frame_num):
        """Process frame with YOLO tracking and ReID verification"""
        results = self.yolo_model.track(frame, persist=True, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)

        vehicle_mask = np.isin(detections.class_id, self.CAR_CLASS_IDS)
        vehicle_detections = detections[vehicle_mask]
        if len(vehicle_detections) == 0:
            return frame

        features = None
        if frame_num % self.FEATURE_VERIFICATION_INTERVAL == 0:
            features = self.extract_features(frame, vehicle_detections)

        self.update_tracks(vehicle_detections, features, frame_num)

        labels = []
        for i in range(len(vehicle_detections.xyxy)):
            if (
                hasattr(vehicle_detections, "tracker_id")
                and vehicle_detections.tracker_id is not None
            ):
                track_id = vehicle_detections.tracker_id[i]
            else:
                track_id = i

            if track_id in self.tracks:
                vehicle_id = self.tracks[track_id]["id"]
                conf_status = "✓" if self.tracks[track_id]["confirmed"] else "?"
                labels.append(f"ID: {vehicle_id}{conf_status}")
            else:
                labels.append("ID: ?")

        annotated_frame = self.box_annotator.annotate(
            frame.copy(), detections=vehicle_detections
        )
        annotated_frame = self.label_annotator.annotate(
            annotated_frame, detections=vehicle_detections, labels=labels
        )
        return annotated_frame

    def save_features(self, file_path):
        """Save all vehicle features to file"""
        with open(file_path, "wb") as f:
            pickle.dump(dict(self.vehicle_features), f)

    def load_features(self, file_path):
        """Load vehicle features from file"""
        with open(file_path, "rb") as f:
            features = pickle.load(f)
        return features


class SecondVideoProcessor:
    def __init__(
        self, yolo_model_path, reid_config_path, reid_model_path, first_video_features
    ):
        self.yolo_model = YOLO(yolo_model_path)
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()

        self.reid_cfg = self._init_reid_config(reid_config_path)
        self.model = make_model(self.reid_cfg, num_class=576, camera_num=20, view_num=8)
        self._load_model_weights(reid_model_path)
        self.transforms = build_transforms()

        self.tracks = {}
        self.next_id = 1
        self.first_video_features = first_video_features
        self.feature_history = defaultdict(deque)

        self.CAR_CLASS_IDS = [2, 5, 7]
        self.REID_THRESHOLD = 0.8
        self.FEATURE_VERIFICATION_INTERVAL = 30
        self.MAX_FRAMES_SINCE_SEEN = 100
        self.MAX_FEATURE_HISTORY = 15

    def _init_reid_config(self, config_path):
        cfg = transreid_cfg.clone()
        cfg.merge_from_file(config_path)
        cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        cfg.freeze()
        return cfg

    def _load_model_weights(self, model_path):
        state_dict = torch.load(model_path, map_location="cpu")
        if "model" in state_dict:
            self.model.load_state_dict(state_dict["model"], strict=False)
        else:
            self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.reid_cfg.MODEL.DEVICE)
        self.model.eval()

    def extract_features(self, image, detections):
        """Extract ReID features for all detections"""
        features = []
        for bbox in detections.xyxy:
            x1, y1, x2, y2 = map(int, bbox)
            vehicle_img = image[y1:y2, x1:x2]
            if vehicle_img.size == 0:
                features.append(None)
                continue

            vehicle_img = cv2.cvtColor(vehicle_img, cv2.COLOR_BGR2RGB)
            vehicle_img = Image.fromarray(vehicle_img)
            vehicle_img = self.transforms(vehicle_img)
            vehicle_img = torch.unsqueeze(vehicle_img, 0).to(self.reid_cfg.MODEL.DEVICE)

            with torch.no_grad():
                outputs = self.model(
                    vehicle_img,
                    cam_label=torch.zeros(
                        1, dtype=torch.long, device=self.reid_cfg.MODEL.DEVICE
                    ),
                    view_label=torch.zeros(
                        1, dtype=torch.long, device=self.reid_cfg.MODEL.DEVICE
                    ),
                )
                outputs = F.normalize(outputs, p=2, dim=1)
            features.append(outputs.cpu().numpy().flatten())
        return features

    def cosine_similarity(self, feat1, feat2):
        """Compute cosine similarity between two features"""
        feat1_norm = feat1 / (np.linalg.norm(feat1) + 1e-10)
        feat2_norm = feat2 / (np.linalg.norm(feat2) + 1e-10)
        return np.dot(feat1_norm, feat2_norm)

    def match_with_first_video(self, feature):
        """Match current feature with first video features"""
        best_match_id = None
        highest_similarity = -1

        for vehicle_id, feature_list in self.first_video_features.items():
            for first_feat in feature_list:
                similarity = self.cosine_similarity(feature, first_feat)
                if similarity > highest_similarity:
                    highest_similarity = similarity
                    best_match_id = vehicle_id

        if highest_similarity > self.REID_THRESHOLD:
            return best_match_id
        return None

    def update_tracks(self, detections, features, frame_num):
        """Update tracks using YOLO tracking and match with first video"""
        current_ids = set()
        updated_tracks = {}

        for detection_idx, track_id in enumerate(detections.tracker_id):
            current_ids.add(track_id)

            if track_id in self.tracks:
                track = self.tracks[track_id]
            else:
                track = {
                    "id": self.next_id,
                    "features": [],
                    "first_seen": frame_num,
                    "last_seen": frame_num,
                    "matched": False,
                }
                self.next_id += 1

            track["last_seen"] = frame_num

            if features and features[detection_idx] is not None:
                feature = features[detection_idx]
                track["features"].append(feature)

                if frame_num % self.FEATURE_VERIFICATION_INTERVAL == 0:
                    matched_id = self.match_with_first_video(feature)
                    if matched_id is not None:
                        track["id"] = matched_id
                        track["matched"] = True
                    else:
                        if track["id"] in self.feature_history:
                            for hist_feat in self.feature_history[track["id"]]:
                                similarity = self.cosine_similarity(feature, hist_feat)
                                if similarity > self.REID_THRESHOLD:
                                    track["matched"] = True
                                    break

                if track["id"] not in self.feature_history:
                    self.feature_history[track["id"]] = deque(
                        maxlen=self.MAX_FEATURE_HISTORY
                    )
                self.feature_history[track["id"]].append(feature)

            updated_tracks[track_id] = track

        for track_id in list(self.tracks.keys()):
            if track_id not in current_ids:
                if (
                    frame_num - self.tracks[track_id]["last_seen"]
                    <= self.MAX_FRAMES_SINCE_SEEN
                ):
                    updated_tracks[track_id] = self.tracks[track_id]

        self.tracks = updated_tracks
        return updated_tracks

    def process_frame(self, frame, frame_num):
        """Process frame with YOLO tracking and ReID matching"""
        results = self.yolo_model.track(frame, persist=True, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)

        vehicle_mask = np.isin(detections.class_id, self.CAR_CLASS_IDS)
        vehicle_detections = detections[vehicle_mask]

        features = None
        if frame_num % self.FEATURE_VERIFICATION_INTERVAL == 0:
            features = self.extract_features(frame, vehicle_detections)

        self.update_tracks(vehicle_detections, features, frame_num)

        labels = []
        for track_id in vehicle_detections.tracker_id:
            if track_id in self.tracks:
                vehicle_id = self.tracks[track_id]["id"]
                match_status = "✓" if self.tracks[track_id]["matched"] else "?"
                labels.append(f"ID: {vehicle_id}{match_status}")
            else:
                labels.append("ID: ?")

        annotated_frame = self.box_annotator.annotate(
            frame.copy(), detections=vehicle_detections
        )
        annotated_frame = self.label_annotator.annotate(
            annotated_frame, detections=vehicle_detections, labels=labels
        )
        return annotated_frame


def process_first_video(input_path, output_path, feature_file):
    """Process the first video, save features, and measure FPS"""
    tracker = VehicleTracker(
        yolo_model_path="yolov8s.pt",
        reid_config_path="transreid/configs/VeRi/deit_transreid_stride.yml",
        reid_model_path="transreid/deit_transreid_veri.pth",
    )

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_num = 0
    total_time = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()
        processed_frame = tracker.process_frame(frame, frame_num)
        processing_time = time.time() - start_time

        total_time += processing_time
        out.write(processed_frame)

        frame_num += 1
        if frame_num % 10 == 0:
            print(f"Processed frame {frame_num} for first video")

    cap.release()
    out.release()

    # Calculate and report FPS
    avg_fps = frame_num / total_time if total_time > 0 else 0
    print(f"First video processed: {frame_num} frames in {total_time:.2f} seconds")
    print(f"Average FPS: {avg_fps:.2f}")

    tracker.save_features(feature_file)
    print(f"Saved features to {feature_file}")

    return tracker.vehicle_features


def process_second_video(input_path, output_path, first_video_features):
    """Process the second video using features from first video and measure FPS"""
    processor = SecondVideoProcessor(
        yolo_model_path="yolov8s.pt",
        reid_config_path="transreid/configs/VeRi/deit_transreid_stride.yml",
        reid_model_path="transreid/deit_transreid_veri.pth",
        first_video_features=first_video_features,
    )

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_num = 0
    total_time = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()
        processed_frame = processor.process_frame(frame, frame_num)
        processing_time = time.time() - start_time

        total_time += processing_time
        out.write(processed_frame)

        frame_num += 1
        if frame_num % 10 == 0:
            print(f"Processed frame {frame_num} for second video")

    cap.release()
    out.release()

    # Calculate and report FPS
    avg_fps = frame_num / total_time if total_time > 0 else 0
    print(f"Second video processed: {frame_num} frames in {total_time:.2f} seconds")
    print(f"Average FPS: {avg_fps:.2f}")


def main():
    # Phase 1: Process first video
    first_video_features = process_first_video(
        input_path="transreid/test/9.mp4",
        output_path="output_cam1.mp4",
        feature_file="first_video_features.pkl",
    )

    # Phase 2: Process second video
    process_second_video(
        input_path="transreid/test/9_1.mp4",
        output_path="output_cam2.mp4",
        first_video_features=first_video_features,
    )


if __name__ == "__main__":
    main()
