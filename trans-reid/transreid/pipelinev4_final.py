# pipeline with lane violations, number plate detection for violators (kavindu - yasindu)

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
from shapely.geometry import LineString
import os
from datetime import datetime
import pandas as pd
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image, ImageEnhance
import re
from difflib import SequenceMatcher
from collections import Counter
import shutil


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
        self.violator_ids = set()  # Store IDs of vehicles that violated the lane
        self.position_history = defaultdict(
            lambda: deque(maxlen=2)
        )  # Store last 2 positions
        self.violator_data = (
            {}
        )  # {vehicle_id: {"frame": frame_num, "timestamp": datetime}}

        # ROI line for lane violation
        self.roi_line = None

        # Snapshot directory
        self.snapshot_dir = "lane_violation_frames"
        if os.path.exists(self.snapshot_dir):
            shutil.rmtree(self.snapshot_dir)
        os.makedirs(self.snapshot_dir, exist_ok=True)

        # Config
        self.CAR_CLASS_IDS = [2, 5, 7]  # COCO vehicle classes
        self.REID_THRESHOLD = 0.80  # Similarity threshold for matching
        self.FEATURE_VERIFICATION_INTERVAL = 30  # Verify IDs every 30 frames
        self.MAX_FRAMES_SINCE_SEEN = 100  # Forget tracks after 100 frames
        self.MAX_FEATURE_HISTORY = 15  # Keep last 15 features for each vehicle

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

    def setup_roi(self, video_path):
        """Allow user to draw a freestyle ROI line on the first frame."""
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print("Error: Cannot read video for ROI selection.")
            return

        clone = frame.copy()
        line_points = []
        drawing = False

        def draw_line(event, x, y, flags, param):
            nonlocal drawing, line_points
            if event == cv2.EVENT_LBUTTONDOWN:
                drawing = True
                line_points = [(x, y)]
            elif event == cv2.EVENT_MOUSEMOVE and drawing:
                line_points.append((x, y))
            elif event == cv2.EVENT_LBUTTONUP:
                drawing = False
                line_points.append((x, y))

        cv2.namedWindow("Draw Lane Line ROI")
        cv2.setMouseCallback("Draw Lane Line ROI", draw_line)

        print(
            "Hold the left mouse button and draw the lane line ROI. Press ENTER to finish."
        )
        while True:
            temp = clone.copy()
            if len(line_points) > 1:
                for i in range(1, len(line_points)):
                    cv2.line(temp, line_points[i - 1], line_points[i], (0, 255, 255), 2)
            cv2.imshow("Draw Lane Line ROI", temp)
            key = cv2.waitKey(1)
            if key == 13:  # ENTER key
                break

        cv2.destroyAllWindows()

        if len(line_points) > 1:
            self.roi_line = LineString(line_points)
            print("Lane line ROI set successfully.")

    def extract_features(self, image, detections):
        """Extract ReID features for all detections."""
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
        """Compute cosine similarity between two features."""
        feat1_norm = feat1 / (np.linalg.norm(feat1) + 1e-10)
        feat2_norm = feat2 / (np.linalg.norm(feat2) + 1e-10)
        return np.dot(feat1_norm, feat2_norm)

    def match_vehicle(self, feature, current_id=None):
        """Match a feature against existing vehicles."""
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

    def save_violation_snapshot(self, frame, violator_id, frame_num):
        """Save frame with timestamp and violator ID."""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        snapshot_path = os.path.join(
            self.snapshot_dir, f"{timestamp}_{violator_id}_{frame_num}.jpg"
        )
        cv2.imwrite(snapshot_path, frame)
        print(f"Saved violation snapshot for video 1: {snapshot_path}")
        return snapshot_path, timestamp

    def update_tracks(self, detections, features, frame_num, frame):
        """Update tracks using YOLO tracking and verify with ReID, check for lane violations."""
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

            # Track position and check for lane violation
            if self.roi_line:
                x1, y1, x2, y2 = detections.xyxy[detection_idx]
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                self.position_history[track["id"]].append((cx, cy))

                if len(self.position_history[track["id"]]) == 2:
                    p1, p2 = self.position_history[track["id"]]
                    movement_line = LineString([p1, p2])
                    if (
                        movement_line.crosses(self.roi_line)
                        and track["id"] not in self.violator_ids
                    ):
                        self.violator_ids.add(track["id"])
                        snapshot_path, timestamp = self.save_violation_snapshot(
                            frame, track["id"], frame_num
                        )
                        self.violator_data[track["id"]] = {
                            "frame": frame_num,
                            "timestamp": timestamp,
                            "snapshot_path": snapshot_path,
                        }
                        print(
                            f"Vehicle {track['id']} violated the lane rule at frame {frame_num}."
                        )

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
        """Process frame with YOLO tracking and ReID verification, mark violators."""
        results = self.yolo_model.track(frame, persist=True, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)

        vehicle_mask = np.isin(detections.class_id, self.CAR_CLASS_IDS)
        vehicle_detections = detections[vehicle_mask]
        if len(vehicle_detections) == 0:
            return frame

        features = None
        if frame_num % self.FEATURE_VERIFICATION_INTERVAL == 0:
            features = self.extract_features(frame, vehicle_detections)

        self.update_tracks(vehicle_detections, features, frame_num, frame)

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
                if vehicle_id in self.violator_ids:
                    labels.append(f"ID: {vehicle_id} VIOLATION")
                else:
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

        # Draw ROI line if set
        # if self.roi_line:
        #     coords = list(self.roi_line.coords)
        #     for i in range(1, len(coords)):
        #         cv2.line(
        #             annotated_frame,
        #             (int(coords[i - 1][0]), int(coords[i - 1][1])),
        #             (int(coords[i][0]), int(coords[i][1])),
        #             (255, 255, 0),
        #             2,
        #         )

        return annotated_frame

    def save_features(self, file_path):
        """Save all vehicle features, violator IDs, and violator data to file."""
        data = {
            "features": dict(self.vehicle_features),
            "violator_ids": self.violator_ids,
            "violator_data": self.violator_data,
        }
        with open(file_path, "wb") as f:
            pickle.dump(data, f)

    def load_features(self, file_path):
        """Load vehicle features, violator IDs, and violator data from file."""
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        return data


class SecondVideoProcessor:
    def __init__(
        self, yolo_model_path, reid_config_path, reid_model_path, first_video_data
    ):
        self.yolo_model = YOLO(yolo_model_path)
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()

        # Initialize TransReID
        self.reid_cfg = self._init_reid_config(reid_config_path)
        self.model = make_model(self.reid_cfg, num_class=576, camera_num=20, view_num=8)
        self._load_model_weights(reid_model_path)
        self.transforms = build_transforms()

        # Initialize number plate detection
        self.plate_model = YOLO("best.pt")
        self.trocr_processor = TrOCRProcessor.from_pretrained(
            "microsoft/trocr-large-printed"
        )
        self.trocr_model = VisionEncoderDecoderModel.from_pretrained(
            "microsoft/trocr-large-printed"
        )
        self.trocr_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.trocr_model.to(self.trocr_device)

        # Tracking storage
        self.tracks = {}
        self.next_id = 1
        self.first_video_features = first_video_data["features"]
        self.violator_ids = first_video_data["violator_ids"]
        self.violator_data = first_video_data.get("violator_data", {})
        self.feature_history = defaultdict(deque)

        # Number plate storage
        self.vehicle_reads = defaultdict(list)
        self.plate_dir = "cropped_plates"
        self.preprocessed_dir = "preprocessed_plates"
        self.snapshot_dir = "lane_violation_frames"
        if os.path.exists(self.plate_dir):
            shutil.rmtree(self.plate_dir)
        os.makedirs(self.plate_dir, exist_ok=True)
        if os.path.exists(self.preprocessed_dir):
            shutil.rmtree(self.preprocessed_dir)
        os.makedirs(self.preprocessed_dir, exist_ok=True)
        if not os.path.exists(self.snapshot_dir):
            os.makedirs(self.snapshot_dir, exist_ok=True)

        # Config
        self.CAR_CLASS_IDS = [2, 5, 7]
        self.REID_THRESHOLD = 0.50
        self.FEATURE_VERIFICATION_INTERVAL = 30
        self.MAX_FRAMES_SINCE_SEEN = 100
        self.MAX_FEATURE_HISTORY = 15
        self.plate_pattern1 = re.compile(r"^[A-Z]{2}-\d{4}$")  # AB-1234
        self.plate_pattern2 = re.compile(r"^[A-Z]{3}\d{4}$")  # ABC1234

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

    def preprocess_plate(self, image, save_path=None):
        """Preprocess number plate image for TrOCR."""
        width, height = image.size
        image = image.crop((int(width * 0.2), 0, width, height))  # Crop province

        # Upscale
        upscale_size = (int(image.width * 2), int(image.height * 2))
        image = image.resize(upscale_size, resample=Image.BICUBIC)

        # Convert to grayscale → histogram equalize → blur → sharpen
        gray = np.array(image.convert("L"))
        equalized = cv2.equalizeHist(gray)
        blurred = cv2.GaussianBlur(equalized, (3, 3), 0)

        sharp = ImageEnhance.Sharpness(Image.fromarray(blurred)).enhance(2.0)
        contrast = ImageEnhance.Contrast(sharp).enhance(1.5)

        processed_image = contrast.convert("RGB")

        if save_path is not None:
            processed_image.save(save_path)

        return processed_image

    def extract_features(self, image, detections):
        """Extract ReID features for all detections."""
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
        """Compute cosine similarity between two features."""
        feat1_norm = feat1 / (np.linalg.norm(feat1) + 1e-10)
        feat2_norm = feat2 / (np.linalg.norm(feat2) + 1e-10)
        return np.dot(feat1_norm, feat2_norm)

    def match_with_first_video(self, feature):
        """Match current feature with first video features."""
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

    def group_similar(self, strings, threshold=0.75):
        """Group similar strings based on similarity threshold."""
        groups = []
        for s in strings:
            placed = False
            for g in groups:
                if SequenceMatcher(None, s, g[0]).ratio() >= threshold:
                    g.append(s)
                    placed = True
                    break
            if not placed:
                groups.append([s])
        return groups

    def process_number_plates(self, frame, vehicle_detections, frame_num):
        """Detect and process number plates for violators, save to CSV."""
        vehicle_boxes = []
        for i, box in enumerate(vehicle_detections.xyxy):
            if (
                hasattr(vehicle_detections, "tracker_id")
                and vehicle_detections.tracker_id is not None
            ):
                track_id = vehicle_detections.tracker_id[i]
                if track_id in self.tracks:
                    vehicle_id = self.tracks[track_id]["id"]
                    if vehicle_id in self.violator_ids:
                        x1, y1, x2, y2 = map(int, box)
                        vehicle_boxes.append(
                            {"id": vehicle_id, "box": (x1, y1, x2, y2)}
                        )

        # Detect number plates
        result_plates = self.plate_model(frame)
        for box in result_plates[0].boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            assigned_id = None

            for v in vehicle_boxes:
                vx1, vy1, vx2, vy2 = v["box"]
                if vx1 <= cx <= vx2 and vy1 <= cy <= vy2:
                    assigned_id = v["id"]
                    break

            if assigned_id is not None and assigned_id in self.violator_ids:
                # Crop and preprocess number plate
                plate_crop = frame[y1:y2, x1:x2]
                if plate_crop.size > 0:
                    plate_crop = cv2.resize(
                        plate_crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC
                    )
                    filename = f"{self.plate_dir}/plate_vid{assigned_id}_frame{frame_num:05d}.png"
                    cv2.imwrite(filename, plate_crop)

                    # Process with TrOCR
                    plate_img = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB)
                    plate_img = Image.fromarray(plate_img)
                    save_path = f"{self.preprocessed_dir}/plate_vid{assigned_id}_frame{frame_num:05d}.png"
                    processed = self.preprocess_plate(plate_img, save_path=save_path)

                    pixel_values = self.trocr_processor(
                        images=processed, return_tensors="pt"
                    ).pixel_values.to(self.trocr_device)
                    generated_ids = self.trocr_model.generate(pixel_values)
                    generated_text = self.trocr_processor.batch_decode(
                        generated_ids, skip_special_tokens=True
                    )[0]
                    cleaned = generated_text.replace(" ", "").upper()

                    if self.plate_pattern1.match(cleaned) or self.plate_pattern2.match(
                        cleaned
                    ):
                        print(f"Plate for Vehicle {assigned_id} → {cleaned} ✅ (valid)")
                        self.vehicle_reads[assigned_id].append((filename, cleaned))
                    else:
                        print(
                            f"Plate for Vehicle {assigned_id} → {cleaned} ❌ (invalid)"
                        )

                    # Draw number plate on frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"Plate of ID {assigned_id}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )

    # def save_violation_snapshot(self, frame, violator_ids):
    # """Save frame with timestamp and violator IDs for video 2."""
    # timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # violator_ids_str = "_".join(str(vid) for vid in sorted(violator_ids))
    # snapshot_path = os.path.join(
    #     self.snapshot_dir, f"cam2_{timestamp}_{violator_ids_str}.jpg"
    # )
    # cv2.imwrite(snapshot_path, frame)
    # print(f"Saved violation snapshot for video 2: {snapshot_path}")

    def update_tracks(self, detections, features, frame_num, frame):
        """Update tracks using YOLO tracking and match with first video."""
        if not hasattr(detections, "tracker_id") or detections.tracker_id is None:
            detections.tracker_id = np.arange(len(detections.xyxy))

        current_ids = set()
        updated_tracks = {}
        frame_violator_ids = []  # Track new violators in this frame

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
                        if (
                            matched_id in self.violator_ids
                            and matched_id not in frame_violator_ids
                        ):
                            frame_violator_ids.append(matched_id)
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

        # Save snapshot if there are new violators
        # if frame_violator_ids:
        # self.save_violation_snapshot(frame, frame_violator_ids)

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
        """Process frame with YOLO tracking, ReID matching, mark violators, and detect number plates."""
        results = self.yolo_model.track(frame, persist=True, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)

        vehicle_mask = np.isin(detections.class_id, self.CAR_CLASS_IDS)
        vehicle_detections = detections[vehicle_mask]

        features = None
        if frame_num % self.FEATURE_VERIFICATION_INTERVAL == 0:
            features = self.extract_features(frame, vehicle_detections)

        self.update_tracks(vehicle_detections, features, frame_num, frame)

        # Process number plates for violators
        self.process_number_plates(frame, vehicle_detections, frame_num)

        labels = []
        for track_id in vehicle_detections.tracker_id:
            if track_id in self.tracks:
                vehicle_id = self.tracks[track_id]["id"]
                if vehicle_id in self.violator_ids:
                    labels.append(f"ID: {vehicle_id} VIOLATION")
                else:
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

    def save_number_plates(self):
        """Save recognized number plates of violators to CSV with violation frame and timestamp."""
        final_rows = []
        for vehicle_id, entries in self.vehicle_reads.items():
            if not entries or vehicle_id not in self.violator_ids:
                continue

            all_valid_texts = [text for _, text in entries]
            groups = self.group_similar(all_valid_texts, threshold=0.75)
            if not groups:
                continue

            best_group = max(groups, key=lambda g: len(g))
            best_plate = Counter(best_group).most_common(1)[0][0]
            last_frame = sorted([img for img, text in entries if text in best_group])[
                -1
            ]

            violation_info = self.violator_data.get(vehicle_id, {})
            violation_frame = violation_info.get("snapshot_path", "N/A")
            violation_time = violation_info.get("timestamp", "N/A")

            print(
                f"Vehicle {vehicle_id} → selected plate: {best_plate} (from {last_frame})"
            )
            final_rows.append(
                {
                    "vehicle_id": vehicle_id,
                    "image": last_frame,
                    "predicted_plate": best_plate,
                    "violation_frame": violation_frame,
                    "violation_time": violation_time,
                }
            )

        if final_rows:
            df = pd.DataFrame(final_rows)
            df.to_csv("recognized_plates_violators.csv", index=False)
            print("✅ Final violator plates saved to recognized_plates_violators.csv")
        else:
            print("No valid number plates detected for violators.")


def process_first_video(input_path, output_path, feature_file):
    """Process the first video, save features, violator IDs, and violator data, and show progress."""
    tracker = VehicleTracker(
        yolo_model_path="yolov8s.pt",
        reid_config_path="transreid/configs/VeRi/deit_transreid_stride.yml",
        reid_model_path="transreid/deit_transreid_veri.pth",
    )

    # Setup ROI before processing
    tracker.setup_roi(input_path)

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_num = 0
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = tracker.process_frame(frame, frame_num)
        out.write(processed_frame)

        frame_num += 1
        if frame_num % 10 == 0:
            elapsed = time.time() - start_time
            fps_current = frame_num / elapsed if elapsed > 0 else 0
            remaining = (
                (total_frames - frame_num) / fps_current if fps_current > 0 else 0
            )
            print(
                f"First video: Processed {frame_num}/{total_frames} frames "
                f"({frame_num / total_frames:.1%}) | "
                f"Elapsed: {elapsed:.1f}s | Remaining: {remaining:.1f}s | FPS: {fps_current:.1f}"
            )

    cap.release()
    out.release()

    tracker.save_features(feature_file)
    print(f"Saved features, violator IDs, and violator data to {feature_file}")

    return {
        "features": tracker.vehicle_features,
        "violator_ids": tracker.violator_ids,
        "violator_data": tracker.violator_data,
    }


def process_second_video(input_path, output_path, first_video_data):
    """Process the second video using features from the first video, mark violators, detect number plates, and show progress."""
    processor = SecondVideoProcessor(
        yolo_model_path="yolov8s.pt",
        reid_config_path="transreid/configs/VeRi/deit_transreid_stride.yml",
        reid_model_path="transreid/deit_transreid_veri.pth",
        first_video_data=first_video_data,
    )

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_num = 0
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = processor.process_frame(frame, frame_num)
        out.write(processed_frame)

        frame_num += 1
        if frame_num % 10 == 0:
            elapsed = time.time() - start_time
            fps_current = frame_num / elapsed if elapsed > 0 else 0
            remaining = (
                (total_frames - frame_num) / fps_current if fps_current > 0 else 0
            )
            print(
                f"Second video: Processed {frame_num}/{total_frames} frames "
                f"({frame_num / total_frames:.1%}) | "
                f"Elapsed: {elapsed:.1f}s | Remaining: {remaining:.1f}s | FPS: {fps_current:.1f}"
            )

    cap.release()
    out.release()

    processor.save_number_plates()
    print(f"Second video processing complete. Output saved to {output_path}")


def main():
    # Phase 1: Process first video
    first_video_data = process_first_video(
        input_path="transreid/test/6.mp4",
        output_path="output_cam1.mp4",
        feature_file="first_video_features.pkl",
    )

    # Phase 2: Process second video
    process_second_video(
        input_path="transreid/test/6_3.mp4",
        output_path="output_cam2.mp4",
        first_video_data=first_video_data,
    )


if __name__ == "__main__":
    main()
