# pipeline with parking violation and number plate recognition

import cv2
import torch
import numpy as np
from PIL import Image, ImageEnhance
from fastreid.config import get_cfg
from fastreid.engine import DefaultPredictor
from fastreid.data.transforms import build_transforms
from ultralytics import YOLO
import supervision as sv
from collections import defaultdict, deque
import time
import pickle
import os
from datetime import datetime
import pandas as pd
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import re
from collections import Counter
from difflib import SequenceMatcher
import shutil

# Global settings for violation detection
VIOLATION_OUTPUT_DIR = "parking_violation_frames"
os.makedirs(VIOLATION_OUTPUT_DIR, exist_ok=True)
VIOLATION_THRESHOLD_SECONDS = 3  # 3 seconds in violation area

# Number plate directories
PLATE_OUTPUT_DIR = "cropped_plates"
PREPROCESSED_PLATE_DIR = "preprocessed_plates"
os.makedirs(PLATE_OUTPUT_DIR, exist_ok=True)
os.makedirs(PREPROCESSED_PLATE_DIR, exist_ok=True)

# ROI drawing callback
def draw_polygon(event, x, y, flags, param):
    if paused and not roi_done:
        if event == cv2.EVENT_LBUTTONDOWN:
            roi_points.append((x, y))

# Updated VehicleTracker class with violation detection
class VehicleTracker:
    def __init__(self, yolo_model_path, reid_config_path, reid_model_path, video_path):
        self.yolo_model = YOLO(yolo_model_path)
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()
        self.video_path = video_path

        # Initialize FastReID
        self.reid_cfg = self._init_reid_config(reid_config_path, reid_model_path)
        self.reid_predictor = DefaultPredictor(self.reid_cfg)
        self.transforms = build_transforms(self.reid_cfg, is_train=False)

        # Tracking storage
        self.tracks = {}
        self.next_id = 1
        self.vehicle_features = defaultdict(list)
        self.feature_history = defaultdict(deque)
        self.violator_ids = set()

        # Violation tracking
        self.zebra_roi = None
        self.violation_timers = {}
        self.violation_data = {}  # {track_id: {'timestamp': ..., 'image': ...}}

        # Config
        self.CAR_CLASS_IDS = [2, 5, 7]
        self.REID_THRESHOLD = 0.5
        self.FEATURE_VERIFICATION_INTERVAL = 30
        self.MAX_FRAMES_SINCE_SEEN = 100
        self.MAX_FEATURE_HISTORY = 5

    def _init_reid_config(self, config_path, model_path):
        cfg = get_cfg()
        cfg.merge_from_file(config_path)
        cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        cfg.MODEL.WEIGHTS = model_path
        return cfg

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
            vehicle_img = self.transforms(vehicle_img)
            vehicle_img = torch.unsqueeze(vehicle_img, 0).to(self.reid_cfg.MODEL.DEVICE)
            with torch.no_grad():
                outputs = self.reid_predictor(vehicle_img)
            features.append(outputs.cpu().numpy().flatten())
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

    def update_tracks(self, detections, features, frame_num, frame):
        if not hasattr(detections, 'tracker_id') or detections.tracker_id is None:
            detections.tracker_id = np.arange(len(detections.xyxy))
        current_ids = set()
        updated_tracks = {}
        for detection_idx, track_id in enumerate(detections.tracker_id):
            current_ids.add(track_id)
            if track_id in self.tracks:
                track = self.tracks[track_id]
            else:
                track = {
                    'id': self.next_id,
                    'features': [],
                    'first_seen': frame_num,
                    'last_seen': frame_num,
                    'confirmed': False,
                    'violated': False
                }
                self.next_id += 1
            track['last_seen'] = frame_num
            if features and features[detection_idx] is not None:
                feature = features[detection_idx]
                track['features'].append(feature)
                if frame_num % self.FEATURE_VERIFICATION_INTERVAL == 0:
                    matched_id = self.match_vehicle(feature, track['id'])
                    if matched_id is not None:
                        track["id"] = matched_id
                        track["confirmed"] = True
                    else:
                        track["confirmed"] = False
                if track["id"] not in self.feature_history:
                    self.feature_history[track["id"]] = deque(maxlen=self.MAX_FEATURE_HISTORY)
                self.feature_history[track["id"]].append(feature)
                if track["confirmed"]:
                    self.vehicle_features[track["id"]].append(feature)
            if self.zebra_roi is not None:
                bbox = detections.xyxy[detection_idx]
                x1, y1, x2, y2 = map(int, bbox)
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                if cv2.pointPolygonTest(self.zebra_roi, (cx, cy), False) >= 0:
                    if track_id not in self.violation_timers:
                        self.violation_timers[track_id] = time.time()
                    else:
                        duration = time.time() - self.violation_timers[track_id]
                        if duration >= VIOLATION_THRESHOLD_SECONDS and not track["violated"]:
                            track["violated"] = True
                            self.violator_ids.add(track["id"])
                            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                            filename = f"{VIOLATION_OUTPUT_DIR}/violation_{timestamp}_id{track['id']}.jpg"
                            cv2.imwrite(filename, frame)
                            self.violation_data[track["id"]] = {"timestamp": timestamp, "image": filename}
                            print(f"🚨 Violation detected! Saved: {filename}")
                else:
                    if track_id in self.violation_timers:
                        del self.violation_timers[track_id]
            updated_tracks[track_id] = track
        for track_id in list(self.tracks.keys()):
            if track_id not in current_ids:
                if frame_num - self.tracks[track_id]["last_seen"] <= self.MAX_FRAMES_SINCE_SEEN:
                    updated_tracks[track_id] = self.tracks[track_id]
                if track_id in self.violation_timers:
                    del self.violation_timers[track_id]
        self.tracks = updated_tracks

    def process_frame(self, frame, frame_num):
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
            if hasattr(vehicle_detections, "tracker_id") and vehicle_detections.tracker_id is not None:
                track_id = vehicle_detections.tracker_id[i]
            else:
                track_id = i
            if track_id in self.tracks:
                vehicle_id = self.tracks[track_id]["id"]
                conf_status = "✓" if self.tracks[track_id]["confirmed"] else "?"
                violator_status = " (VIOLATOR)" if self.tracks[track_id]["violated"] else ""
                labels.append(f"ID: {vehicle_id}{conf_status}{violator_status}")
            else:
                labels.append("ID: ?")
        annotated_frame = self.box_annotator.annotate(frame.copy(), detections=vehicle_detections)
        annotated_frame = self.label_annotator.annotate(annotated_frame, detections=vehicle_detections, labels=labels)
        if self.zebra_roi is not None:
            cv2.polylines(annotated_frame, [self.zebra_roi], isClosed=True, color=(0, 0, 255), thickness=2)
        return annotated_frame

    def save_features(self, file_path):
        data = {
            "features": dict(self.vehicle_features),
            "violator_ids": list(self.violator_ids),
            "violation_data": self.violation_data,
        }
        with open(file_path, "wb") as f:
            pickle.dump(data, f)

    def load_features(self, file_path):
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        return data["features"], set(data["violator_ids"]), data.get("violation_data", {})

    def setup_roi(self):
        global roi_points, paused, roi_done
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Error opening video: {self.video_path}")
            return None
        ret, frame = cap.read()
        if not ret:
            print("Error reading first frame")
            cap.release()
            return None
        cap.release()
        roi_points = []
        paused = True
        roi_done = False
        cv2.namedWindow("Draw Parking Violation ROI")
        cv2.setMouseCallback("Draw Parking Violation ROI", draw_polygon)
        print("▶ Press 'p' to pause/play")
        print("🖱 While paused, click to draw ROI points")
        print("✅ Press 'r' to finalize ROI")
        print("❌ Press 'q' to quit")
        while True:
            display_frame = frame.copy()
            if roi_points:
                for pt in roi_points:
                    cv2.circle(display_frame, pt, 5, (0, 255, 0), -1)
                if len(roi_points) >= 2:
                    cv2.polylines(display_frame, [np.array(roi_points)], isClosed=False, color=(0, 255, 255), thickness=2)
                if len(roi_points) >= 3:
                    cv2.putText(display_frame, "Press 'r' to lock ROI", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow("Draw Parking Violation ROI", display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("r") and len(roi_points) >= 3:
                roi_done = True
                break
            elif key == ord("q"):
                break
        cv2.destroyAllWindows()
        if roi_done:
            self.zebra_roi = np.array(roi_points, dtype=np.int32)
            print("✅ ROI finalized")
            return self.zebra_roi
        return None

class SecondVideoProcessor:
    def __init__(self, yolo_model_path, reid_config_path, reid_model_path, first_video_data):
        self.yolo_model = YOLO(yolo_model_path)
        self.plate_model = YOLO("best.pt")
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()
        self.reid_cfg = self._init_reid_config(reid_config_path, reid_model_path)
        self.reid_predictor = DefaultPredictor(self.reid_cfg)
        self.transforms = build_transforms(self.reid_cfg, is_train=False)
        self.processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-printed")
        self.ocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-printed")
        self.ocr_model.to("cuda" if torch.cuda.is_available() else "cpu")
        self.tracks = {}
        self.next_id = 1
        self.first_video_features = first_video_data["features"]
        self.violator_ids = first_video_data["violator_ids"]
        self.violation_data = first_video_data.get("violation_data", {})
        self.feature_history = defaultdict(deque)
        self.vehicle_reads = defaultdict(list)
        self.frame_count = 0
        self.CAR_CLASS_IDS = [2, 5, 7]
        self.REID_THRESHOLD = 0.5
        self.FEATURE_VERIFICATION_INTERVAL = 30
        self.MAX_FRAMES_SINCE_SEEN = 100
        self.MAX_FEATURE_HISTORY = 5
        # Clear and recreate plate directories
        if os.path.exists(PLATE_OUTPUT_DIR):
            shutil.rmtree(PLATE_OUTPUT_DIR)
        if os.path.exists(PREPROCESSED_PLATE_DIR):
            shutil.rmtree(PREPROCESSED_PLATE_DIR)
        os.makedirs(PLATE_OUTPUT_DIR, exist_ok=True)
        os.makedirs(PREPROCESSED_PLATE_DIR, exist_ok=True)

    def _init_reid_config(self, config_path, model_path):
        cfg = get_cfg()
        cfg.merge_from_file(config_path)
        cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        cfg.MODEL.WEIGHTS = model_path
        return cfg

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
            vehicle_img = self.transforms(vehicle_img)
            vehicle_img = torch.unsqueeze(vehicle_img, 0).to(self.reid_cfg.MODEL.DEVICE)
            with torch.no_grad():
                outputs = self.reid_predictor(vehicle_img)
            features.append(outputs.cpu().numpy().flatten())
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

    def preprocess_plate(self, image, save_path=None):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        width, height = image.size
        image = image.crop((int(width * 0.2), 0, width, height))  # Crop province
        upscale_size = (int(image.width * 2), int(image.height * 2))
        image = image.resize(upscale_size, resample=Image.BICUBIC)
        gray = np.array(image.convert("L"))
        equalized = cv2.equalizeHist(gray)
        blurred = cv2.GaussianBlur(equalized, (3, 3), 0)
        sharp = ImageEnhance.Sharpness(Image.fromarray(blurred)).enhance(2.0)
        contrast = ImageEnhance.Contrast(sharp).enhance(1.5)
        processed_image = contrast.convert("RGB")
        if save_path is not None:
            processed_image.save(save_path)
        return processed_image

    def read_plate(self, plate_image):
        processed_image = self.preprocess_plate(plate_image)
        pixel_values = self.processor(images=processed_image, return_tensors="pt").pixel_values.to(self.ocr_model.device)
        generated_ids = self.ocr_model.generate(pixel_values)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        cleaned = generated_text.replace(" ", "").upper()
        return cleaned if cleaned else None

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

    def update_tracks(self, detections, features, frame_num, frame):
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
                    "is_violator": False,
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
                        track["is_violator"] = matched_id in self.violator_ids
                    else:
                        if track["id"] in self.feature_history:
                            for hist_feat in self.feature_history[track["id"]]:
                                similarity = self.cosine_similarity(feature, hist_feat)
                                if similarity > self.REID_THRESHOLD:
                                    track["matched"] = True
                                    break
                if track["id"] not in self.feature_history:
                    self.feature_history[track["id"]] = deque(maxlen=self.MAX_FEATURE_HISTORY)
                self.feature_history[track["id"]].append(feature)
            updated_tracks[track_id] = track
        for track_id in list(self.tracks.keys()):
            if track_id not in current_ids:
                if frame_num - self.tracks[track_id]["last_seen"] <= self.MAX_FRAMES_SINCE_SEEN:
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
        self.update_tracks(vehicle_detections, features, frame_num, frame)
        plate_results = self.plate_model(frame)
        for plate_box in plate_results[0].boxes.xyxy:
            px1, py1, px2, py2 = map(int, plate_box)
            pcx, pcy = (px1 + px2) // 2, (py1 + py2) // 2
            for i, track_id in enumerate(vehicle_detections.tracker_id):
                if track_id in self.tracks and self.tracks[track_id]["id"] in self.violator_ids:
                    vx1, vy1, vx2, vy2 = map(int, vehicle_detections.xyxy[i])
                    if vx1 <= pcx <= vx2 and vy1 <= pcy <= vy2:
                        plate_crop = frame[py1:py2, px1:px2]
                        if plate_crop.size > 0:
                            filename = f"{PLATE_OUTPUT_DIR}/plate_vid{self.tracks[track_id]['id']}_frame{self.frame_count:05d}.png"
                            plate_crop = cv2.resize(plate_crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
                            cv2.imwrite(filename, plate_crop)
                            preprocessed_filename = f"{PREPROCESSED_PLATE_DIR}/plate_vid{self.tracks[track_id]['id']}_frame{self.frame_count:05d}.png"
                            plate_text = self.read_plate(plate_crop)
                            pattern1 = re.compile(r"^[A-Z]{2}-\d{4}$")
                            pattern2 = re.compile(r"^[A-Z]{3}\d{4}$")
                            if plate_text and (pattern1.match(plate_text) or pattern2.match(plate_text)):
                                violator_id = self.tracks[track_id]["id"]
                                violation_info = self.violation_data.get(violator_id, {})
                                self.vehicle_reads[violator_id].append((filename, plate_text))
                                print(f"{filename} → {plate_text} ✅ (valid)")
                            else:
                                print(f"{filename} → {plate_text} ❌ (invalid)")
        self.frame_count += 1
        labels = []
        for i, track_id in enumerate(vehicle_detections.tracker_id):
            if track_id in self.tracks:
                track = self.tracks[track_id]
                vehicle_id = track["id"]
                match_status = "✓" if track["matched"] else "?"
                violator_status = " (VIOLATOR)" if track["is_violator"] else ""
                labels.append(f"ID: {vehicle_id}{match_status}{violator_status}")
            else:
                labels.append("ID: ?")
        annotated_frame = self.box_annotator.annotate(frame.copy(), detections=vehicle_detections)
        return self.label_annotator.annotate(annotated_frame, detections=vehicle_detections, labels=labels)

    def save_violator_plates(self, csv_path):
        final_rows = []
        for vehicle_id, entries in self.vehicle_reads.items():
            if not entries:
                continue
            all_valid_texts = [text for _, text in entries]
            groups = self.group_similar(all_valid_texts, threshold=0.75)
            best_group = max(groups, key=lambda g: len(g))
            best_plate = Counter(best_group).most_common(1)[0][0]
            last_frame = sorted([img for img, text in entries if text in best_group])[-1]
            violation_info = self.violation_data.get(int(vehicle_id), {})
            print(f"Vehicle {vehicle_id} → selected plate: {best_plate} (from {last_frame})")
            final_rows.append({
                "vehicle_id": vehicle_id,
                "predicted_plate": best_plate,
                "violation_image": violation_info.get("image", "unknown.jpg"),
                "timestamp": violation_info.get("timestamp", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
            })
        df = pd.DataFrame(final_rows)
        df.to_csv(csv_path, index=False)
        print(f"Saved violator plates to {csv_path}")

def process_first_video(input_path, output_path, feature_file):
    tracker = VehicleTracker(
        yolo_model_path="yolov8s.pt",
        reid_config_path="configs/VeRi/sbs_R50-ibn.yml",
        reid_model_path="Model/veri_sbs_R50-ibn.pth",
        video_path=input_path
    )
    if tracker.setup_roi() is None:
        print("ROI setup failed. Exiting.")
        return None
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error opening video: {input_path}")
        return None
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
            print(f"Processed {frame_num}/{total_frames} frames ({frame_num / total_frames:.1%}) | "
                  f"Elapsed: {elapsed:.1f}s | Remaining: {remaining:.1f}s | FPS: {fps_current:.1f}")
    cap.release()
    out.release()
    tracker.save_features(feature_file)
    print(f"Saved features to {feature_file}")
    return {
        "features": tracker.vehicle_features,
        "violator_ids": tracker.violator_ids,
        "violation_data": tracker.violation_data,
    }

def process_second_video(input_path, output_path, first_video_data):
    processor = SecondVideoProcessor(
        yolo_model_path="yolov8s.pt",
        reid_config_path="configs/VeRi/sbs_R50-ibn.yml",
        reid_model_path="Model/veri_sbs_R50-ibn.pth",
        first_video_data=first_video_data
    )
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error opening video: {input_path}")
        return
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
            print(f"Processed {frame_num}/{total_frames} frames ({frame_num / total_frames:.1%}) | "
                  f"Elapsed: {elapsed:.1f}s | Remaining: {remaining:.1f}s | FPS: {fps_current:.1f}")
    cap.release()
    out.release()
    processor.save_violator_plates("violator_plates.csv")

def main():
    first_video_data = process_first_video(
        input_path="test/8.mp4",
        output_path="output_cam1.mp4",
        feature_file="first_video_data.pkl"
    )
    if first_video_data is None:
        print("First video processing failed")
        return
    process_second_video(
        input_path="test/8_1.mp4",
        output_path="output_cam2.mp4",
        first_video_data=first_video_data
    )

if __name__ == "__main__":
    main()