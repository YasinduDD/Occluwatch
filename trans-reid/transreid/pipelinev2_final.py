# pipeline with red light violation using TransReID and number plate detection

import cv2
import torch
import numpy as np
from PIL import Image, ImageEnhance
from torchvision import transforms
import torch.nn.functional as F
from ultralytics import YOLO
import supervision as sv
from collections import defaultdict, deque
import time
import pickle
from Src.colorRecog import recognize_color, chooseOne
from Src.violationRecog import point_to_line_distance
import os
from datetime import datetime
from config import cfg
from model.make_model import make_model
import pandas as pd
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import re
from difflib import SequenceMatcher
from collections import Counter
import shutil

# Global variables for ROI drawing
line_points = []
drawing = False
start_point = None
current_point = None


def build_transforms(is_train=False):
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transform


class VehicleTracker:
    def __init__(self, yolo_model_path, reid_config_path, reid_model_path, video_path):
        self.yolo_model = YOLO(yolo_model_path)
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()
        self.video_path = video_path

        self.cfg = cfg.clone()
        self.cfg.merge_from_file(reid_config_path)
        self.cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.cfg.MODEL.DEVICE}")
        self.cfg.freeze()
        self.model = make_model(self.cfg, num_class=576, camera_num=20, view_num=8)
        state_dict = torch.load(reid_model_path, map_location="cpu")
        if "model" in state_dict:
            self.model.load_state_dict(state_dict["model"], strict=False)
        else:
            self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.cfg.MODEL.DEVICE)
        self.model.eval()
        self.device = self.cfg.MODEL.DEVICE
        self.transforms = build_transforms()

        self.tracks = {}
        self.next_id = 1
        self.vehicle_features = defaultdict(list)
        self.feature_history = defaultdict(deque)
        self.violator_ids = set()
        self.violator_data = (
            {}
        )  # {vehicle_id: {"frame": frame_num, "timestamp": str, "snapshot_path": str}}

        self.roi_lines = []
        self.CAR_CLASS_IDS = [2, 5, 7]
        self.TRAFFIC_LIGHT_CLASS = 9
        self.REID_THRESHOLD = 0.8
        self.FEATURE_VERIFICATION_INTERVAL = 30
        self.MAX_FRAMES_SINCE_SEEN = 100
        self.MAX_FEATURE_HISTORY = 15

        self.PALLETES = {
            "red": (0, 0, 255),
            "green": (0, 255, 0),
            "yellow": (0, 255, 255),
            "light's off": (192, 192, 192),
            "unknown": (255, 255, 255),
        }

        self.snapshot_dir = "redlight_violation_frames"
        if not os.path.exists(self.snapshot_dir):
            os.makedirs(self.snapshot_dir)

    def setup_roi(self):
        global line_points, drawing, start_point, current_point
        line_points = []
        drawing = False
        start_point = None
        current_point = None

        cap = cv2.VideoCapture(self.video_path)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print("Error reading video")
            return

        display_frame = (
            cv2.resize(frame, (1280, 720)) if frame.shape[1] > 1280 else frame.copy()
        )

        def mouse_callback(event, x, y, flags, param):
            global drawing, start_point, current_point, line_points
            if event == cv2.EVENT_LBUTTONDOWN:
                drawing = True
                start_point = (x, y)
                current_point = (x, y)
            elif event == cv2.EVENT_MOUSEMOVE and drawing:
                current_point = (x, y)
            elif event == cv2.EVENT_LBUTTONUP:
                if drawing and start_point:
                    line_points.append(
                        (
                            start_point[0],
                            start_point[1],
                            current_point[0],
                            current_point[1],
                        )
                    )
                    print(f"ROI Line added: {start_point} to {current_point}")
                drawing = False
                start_point = None
                current_point = None

        cv2.imshow("ROI Selection", display_frame)
        cv2.setMouseCallback("ROI Selection", mouse_callback)

        print("Draw ROI lines on the first frame (Left click and drag)")
        print("Press SPACE to confirm and start processing")
        print("Press 'c' to clear all lines")
        print("Press ESC to exit")

        while True:
            temp_frame = display_frame.copy()
            for x1, y1, x2, y2 in line_points:
                cv2.line(temp_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            if drawing and start_point and current_point:
                cv2.line(temp_frame, start_point, current_point, (0, 255, 255), 2)
            cv2.imshow("ROI Selection", temp_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("c"):
                line_points = []
                print("All ROI lines cleared")
            elif key == 32:  # SPACE
                if line_points:
                    break
                else:
                    print("Please draw at least one ROI line first")
            elif key == 27:  # ESC
                print("Exiting")
                exit()

        cv2.destroyAllWindows()

        if display_frame.shape != frame.shape:
            scale_x = frame.shape[1] / display_frame.shape[1]
            scale_y = frame.shape[0] / display_frame.shape[0]
            self.roi_lines = [
                (
                    int(x1 * scale_x),
                    int(y1 * scale_y),
                    int(x2 * scale_x),
                    int(y2 * scale_y),
                )
                for x1, y1, x2, y2 in line_points
            ]
        else:
            self.roi_lines = line_points
        print(f"\nStarting processing with {len(self.roi_lines)} ROI lines...")

    def extract_features(self, image, detections):
        features = []
        for bbox in detections.xyxy:
            x1, y1, x2, y2 = map(int, bbox)
            vehicle_img = image[y1:y2, x1:x2]
            if vehicle_img.size == 0:
                features.append(None)
                continue
            vehicle_img = cv2.cvtColor(vehicle_img, cv2.COLOR_BGR2RGB)
            vehicle_img = Image.fromarray(vehicle_img)
            vehicle_img = self.transforms(vehicle_img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                feat = self.model(
                    vehicle_img,
                    cam_label=torch.zeros(1, dtype=torch.long, device=self.device),
                    view_label=torch.zeros(1, dtype=torch.long, device=self.device),
                )
                feat = F.normalize(feat, p=2, dim=1)
            features.append(feat.cpu().numpy().flatten())
        return features

    def cosine_similarity(self, feat1, feat2):
        feat1_norm = feat1 / (np.linalg.norm(feat1) + 1e-10)
        feat2_norm = feat2 / (np.linalg.norm(feat2) + 1e-10)
        return np.dot(feat1_norm, feat2_norm)

    def match_vehicle(self, feature, current_id=None):
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

    def update_tracks(self, detections, features, frame_num, chosen, frame):
        if not hasattr(detections, "tracker_id") or detections.tracker_id is None:
            detections.tracker_id = np.arange(len(detections.xyxy))
        current_ids = set()
        updated_tracks = {}
        frame_violator_ids = []
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
            if chosen is not None and chosen[1] > 0 and frame_num > 60:
                cx = (
                    detections.xyxy[detection_idx][0]
                    + detections.xyxy[detection_idx][2]
                ) / 2
                cy = (
                    detections.xyxy[detection_idx][1]
                    + detections.xyxy[detection_idx][3]
                ) / 2
                for x1, y1, x2, y2 in self.roi_lines:
                    distance = point_to_line_distance(cx, cy, x1, y1, x2, y2)
                    if distance <= 10:
                        if track["id"] not in self.violator_ids:
                            self.violator_ids.add(track["id"])
                            frame_violator_ids.append(track["id"])
                        break
            updated_tracks[track_id] = track
        if frame_violator_ids:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            violator_ids_str = "_".join(str(vid) for vid in sorted(frame_violator_ids))
            snapshot_path = os.path.join(
                self.snapshot_dir,
                f"violation_frame{frame_num}_{timestamp}_{violator_ids_str}.jpg",
            )
            cv2.imwrite(snapshot_path, frame)
            print(f"Saved snapshot: {snapshot_path}")
            for vid in frame_violator_ids:
                self.violator_data[vid] = {
                    "frame": frame_num,
                    "timestamp": timestamp,
                    "snapshot_path": snapshot_path,
                }
        for track_id in list(self.tracks.keys()):
            if track_id not in current_ids:
                if (
                    frame_num - self.tracks[track_id]["last_seen"]
                    <= self.MAX_FRAMES_SINCE_SEEN
                ):
                    updated_tracks[track_id] = self.tracks[track_id]
        self.tracks = updated_tracks

    def process_frame(self, frame, frame_num):
        results = self.yolo_model.track(frame, persist=True, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        vehicle_mask = np.isin(detections.class_id, self.CAR_CLASS_IDS)
        traffic_light_mask = detections.class_id == self.TRAFFIC_LIGHT_CLASS
        vehicle_detections = detections[vehicle_mask]
        traffic_light_detections = detections[traffic_light_mask]

        chosen = None
        light_colors = {}
        if len(traffic_light_detections) > 0:
            traffic_light_boxes = torch.tensor(
                traffic_light_detections.xyxy, dtype=torch.float32
            )
            conf = torch.ones((len(traffic_light_detections), 1))
            cls = torch.full(
                (len(traffic_light_detections), 1),
                self.TRAFFIC_LIGHT_CLASS,
                dtype=torch.float32,
            )
            traffic_light_boxes = torch.cat([traffic_light_boxes, conf, cls], dim=1)
            light_colors = recognize_color(frame, traffic_light_boxes)
            chosen = chooseOne(light_colors)

        features = None
        if frame_num % self.FEATURE_VERIFICATION_INTERVAL == 0:
            features = self.extract_features(frame, vehicle_detections)
        self.update_tracks(vehicle_detections, features, frame_num, chosen, frame)

        annotated_frame = frame.copy()
        for x1, y1, x2, y2 in self.roi_lines:
            cv2.line(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        labels = []
        for i, track_id in enumerate(vehicle_detections.tracker_id):
            if track_id in self.tracks:
                vehicle_id = self.tracks[track_id]["id"]
                if vehicle_id in self.violator_ids:
                    labels.append(f"ID: {vehicle_id} VIOLATION")
                else:
                    labels.append(f"ID: {vehicle_id}")
            else:
                labels.append("ID: ?")
        annotated_frame = self.box_annotator.annotate(
            annotated_frame, detections=vehicle_detections
        )
        annotated_frame = self.label_annotator.annotate(
            annotated_frame, detections=vehicle_detections, labels=labels
        )

        if len(traffic_light_detections) > 0:
            traffic_light_colors = {}
            for color, detections in light_colors.items():
                for n, conf in detections:
                    traffic_light_colors[n] = color
            for i, bbox in enumerate(traffic_light_detections.xyxy):
                x1, y1, x2, y2 = map(int, bbox)
                if i in traffic_light_colors:
                    color = traffic_light_colors[i]
                    label = f"{color.upper()}"
                    box_color = self.PALLETES.get(color, self.PALLETES["unknown"])
                else:
                    label = "UNKNOWN"
                    box_color = self.PALLETES["unknown"]
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2)
                cv2.putText(
                    annotated_frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    box_color,
                    2,
                )

        return annotated_frame

    def save_features(self, file_path):
        data = {
            "features": dict(self.vehicle_features),
            "violator_ids": self.violator_ids,
            "violator_data": self.violator_data,
        }
        with open(file_path, "wb") as f:
            pickle.dump(data, f)


class SecondVideoProcessor:
    def __init__(
        self, yolo_model_path, reid_config_path, reid_model_path, first_video_data
    ):
        self.yolo_model = YOLO(yolo_model_path)
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()

        self.cfg = cfg.clone()
        self.cfg.merge_from_file(reid_config_path)
        self.cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.cfg.MODEL.DEVICE}")
        self.cfg.freeze()
        self.model = make_model(self.cfg, num_class=576, camera_num=20, view_num=8)
        state_dict = torch.load(reid_model_path, map_location="cpu")
        if "model" in state_dict:
            self.model.load_state_dict(state_dict["model"], strict=False)
        else:
            self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.cfg.MODEL.DEVICE)
        self.model.eval()
        self.device = self.cfg.MODEL.DEVICE
        self.transforms = build_transforms()

        self.plate_model = YOLO("best.pt")
        self.trocr_processor = TrOCRProcessor.from_pretrained(
            "microsoft/trocr-large-printed"
        )
        self.trocr_model = VisionEncoderDecoderModel.from_pretrained(
            "microsoft/trocr-large-printed"
        )
        self.trocr_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.trocr_model.to(self.trocr_device)

        self.tracks = {}
        self.next_id = 1
        self.first_video_features = first_video_data["features"]
        self.violator_ids = first_video_data["violator_ids"]
        self.violator_data = first_video_data.get("violator_data", {})
        self.feature_history = defaultdict(deque)
        self.vehicle_reads = defaultdict(list)

        self.plate_dir = "cropped_plates"
        self.preprocessed_dir = "preprocessed_plates"
        if os.path.exists(self.plate_dir):
            shutil.rmtree(self.plate_dir)
        os.makedirs(self.plate_dir, exist_ok=True)
        if os.path.exists(self.preprocessed_dir):
            shutil.rmtree(self.preprocessed_dir)
        os.makedirs(self.preprocessed_dir, exist_ok=True)

        self.CAR_CLASS_IDS = [2, 5, 7]
        self.REID_THRESHOLD = 0.6
        self.FEATURE_VERIFICATION_INTERVAL = 30
        self.MAX_FRAMES_SINCE_SEEN = 100
        self.MAX_FEATURE_HISTORY = 15
        self.plate_pattern1 = re.compile(r"^[A-Z]{2}-\d{4}$")
        self.plate_pattern2 = re.compile(r"^[A-Z]{3}\d{4}$")

    def preprocess_plate(self, image, save_path=None):
        width, height = image.size
        image = image.crop((int(width * 0.2), 0, width, height))
        upscale_size = (int(image.width * 2), int(image.height * 2))
        image = image.resize(upscale_size, resample=Image.BICUBIC)
        gray = np.array(image.convert("L"))
        equalized = cv2.equalizeHist(gray)
        blurred = cv2.GaussianBlur(equalized, (3, 3), 0)
        sharp = ImageEnhance.Sharpness(Image.fromarray(blurred)).enhance(2.0)
        contrast = ImageEnhance.Contrast(sharp).enhance(1.5)
        processed_image = contrast.convert("RGB")
        if save_path:
            processed_image.save(save_path)
        return processed_image

    def extract_features(self, image, detections):
        features = []
        for bbox in detections.xyxy:
            x1, y1, x2, y2 = map(int, bbox)
            vehicle_img = image[y1:y2, x1:x2]
            if vehicle_img.size == 0:
                features.append(None)
                continue
            vehicle_img = cv2.cvtColor(vehicle_img, cv2.COLOR_BGR2RGB)
            vehicle_img = Image.fromarray(vehicle_img)
            vehicle_img = self.transforms(vehicle_img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                feat = self.model(
                    vehicle_img,
                    cam_label=torch.zeros(1, dtype=torch.long, device=self.device),
                    view_label=torch.zeros(1, dtype=torch.long, device=self.device),
                )
                feat = F.normalize(feat, p=2, dim=1)
            features.append(feat.cpu().numpy().flatten())
        return features

    def cosine_similarity(self, feat1, feat2):
        feat1_norm = feat1 / (np.linalg.norm(feat1) + 1e-10)
        feat2_norm = feat2 / (np.linalg.norm(feat2) + 1e-10)
        return np.dot(feat1_norm, feat2_norm)

    def match_with_first_video(self, feature):
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

    def process_number_plates(self, frame, vehicle_detections, frame_num):
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
            if assigned_id is not None:
                plate_crop = frame[y1:y2, x1:x2]
                if plate_crop.size > 0:
                    plate_crop = cv2.resize(
                        plate_crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC
                    )
                    filename = f"{self.plate_dir}/plate_vid{assigned_id}_frame{frame_num:05d}.png"
                    cv2.imwrite(filename, plate_crop)
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

    def group_similar(self, strings, threshold=0.75):
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

    def update_tracks(self, detections, features, frame_num):
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

    def process_frame(self, frame, frame_num):
        results = self.yolo_model.track(frame, persist=True, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        vehicle_mask = np.isin(detections.class_id, self.CAR_CLASS_IDS)
        vehicle_detections = detections[vehicle_mask]
        features = None
        if frame_num % self.FEATURE_VERIFICATION_INTERVAL == 0:
            features = self.extract_features(frame, vehicle_detections)
        self.update_tracks(vehicle_detections, features, frame_num)
        self.process_number_plates(frame, vehicle_detections, frame_num)
        labels = []
        for track_id in vehicle_detections.tracker_id:
            if track_id in self.tracks:
                vehicle_id = self.tracks[track_id]["id"]
                if vehicle_id in self.violator_ids:
                    labels.append(f"ID: {vehicle_id} VIOLATION")
                else:
                    labels.append(f"ID: {vehicle_id}")
            else:
                labels.append("ID: ?")
        annotated_frame = self.box_annotator.annotate(
            frame.copy(), detections=vehicle_detections
        )
        return self.label_annotator.annotate(
            annotated_frame, detections=vehicle_detections, labels=labels
        )

    def save_number_plates(self):
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
    tracker = VehicleTracker(
        yolo_model_path="yolo11m.pt",
        reid_config_path="transreid/configs/VeRi/deit_transreid_stride.yml",
        reid_model_path="transreid/deit_transreid_veri.pth",
        video_path=input_path,
    )
    tracker.setup_roi()
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
        if frame_num % 100 == 0:
            elapsed = time.time() - start_time
            fps_current = frame_num / elapsed
            remaining = (total_frames - frame_num) / fps_current
            print(
                f"Processed {frame_num}/{total_frames} frames ({frame_num / total_frames:.1%}) | "
                f"Elapsed: {elapsed:.1f}s | Remaining: {remaining:.1f}s | FPS: {fps_current:.1f}"
            )
    cap.release()
    out.release()
    tracker.save_features(feature_file)
    print(f"Saved features to {feature_file}")
    return {
        "features": tracker.vehicle_features,
        "violator_ids": tracker.violator_ids,
        "violator_data": tracker.violator_data,
    }


def process_second_video(input_path, output_path, first_video_data):
    processor = SecondVideoProcessor(
        yolo_model_path="yolo11s.pt",
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
        if frame_num % 100 == 0:
            elapsed = time.time() - start_time
            fps_current = frame_num / elapsed
            remaining = (total_frames - frame_num) / fps_current
            print(
                f"Processed {frame_num}/{total_frames} frames ({frame_num / total_frames:.1%}) | "
                f"Elapsed: {elapsed:.1f}s | Remaining: {remaining:.1f}s | FPS: {fps_current:.1f}"
            )
    cap.release()
    out.release()
    processor.save_number_plates()
    print(f"Second video processing complete. Output saved to {output_path}")


def main():
    first_video_data = process_first_video(
        input_path="transreid/test/9.mp4",
        output_path="output_cam1.mp4",
        feature_file="first_video_features.pkl",
    )
    process_second_video(
        input_path="transreid/test/9_2.mp4",
        output_path="output_cam2.mp4",
        first_video_data=first_video_data,
    )


if __name__ == "__main__":
    main()
