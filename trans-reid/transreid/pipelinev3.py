# pipeline with line violations (miriswatta - gampaha)

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
    def __init__(self, yolo_model_path, reid_config_path, reid_model_path, video_path):
        self.yolo_model = YOLO(yolo_model_path)
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()
        self.video_path = video_path

        # Initialize TransReID
        self.reid_cfg = self._init_reid_config(reid_config_path)
        self.model = make_model(self.reid_cfg, num_class=576, camera_num=20, view_num=8)
        self._load_model_weights(reid_model_path)
        self.transforms = build_transforms()

        # Tracking storage
        self.tracks = {}
        self.next_id = 1
        self.vehicle_features = defaultdict(list)
        self.feature_history = defaultdict(deque)
        self.violator_ids = set()

        # ROI config
        self.roi_line = None
        self.CAR_CLASS_IDS = [2, 3, 5, 7]
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

    def setup_roi(self):
        cap = cv2.VideoCapture(self.video_path)
        ret, first_frame = cap.read()
        cap.release()
        if not ret:
            print("Error reading video")
            return

        clone = first_frame.copy()
        line_points = []

        def draw_line(event, x, y, flags, param):
            nonlocal line_points
            if event == cv2.EVENT_LBUTTONDOWN and len(line_points) < 2:
                line_points.append((x, y))
                if len(line_points) == 2:
                    self.roi_line = LineString(line_points)

        cv2.namedWindow("Draw ROI Line")
        cv2.setMouseCallback("Draw ROI Line", draw_line)

        while True:
            temp = clone.copy()
            if len(line_points) == 1:
                cv2.circle(temp, line_points[0], 5, (0, 255, 255), -1)
            elif len(line_points) == 2:
                cv2.line(temp, line_points[0], line_points[1], (0, 255, 255), 2)
            cv2.imshow("Draw ROI Line", temp)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC to exit
                break
            elif key == 32 and len(line_points) == 2:  # SPACE to confirm
                break
        cv2.destroyAllWindows()

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
            vehicle_img = (
                self.transforms(vehicle_img).unsqueeze(0).to(self.reid_cfg.MODEL.DEVICE)
            )
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
                    "positions": deque(maxlen=5),
                    "violated": False,
                    "first_seen": frame_num,
                    "last_seen": frame_num,
                    "confirmed": False,
                }
                self.next_id += 1
            track["last_seen"] = frame_num
            cx = (
                detections.xyxy[detection_idx][0] + detections.xyxy[detection_idx][2]
            ) / 2
            cy = (
                detections.xyxy[detection_idx][1] + detections.xyxy[detection_idx][3]
            ) / 2
            track["positions"].append((cx, cy))
            if self.roi_line and not track["violated"] and len(track["positions"]) >= 2:
                y_velocity = track["positions"][-1][1] - track["positions"][0][1]
                if y_velocity > 5:
                    x2 = detections.xyxy[detection_idx][2]
                    y1 = detections.xyxy[detection_idx][1]
                    y2 = detections.xyxy[detection_idx][3]
                    right_edge = LineString([(x2, y1), (x2, y2)])
                    if right_edge.intersects(self.roi_line):
                        track["violated"] = True
                        self.violator_ids.add(track["id"])
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
        # if self.roi_line:
        #     p1, p2 = self.roi_line.coords
        #     cv2.line(
        #         annotated_frame,
        #         (int(p1[0]), int(p1[1])),
        #         (int(p2[0]), int(p2[1])),
        #         (255, 255, 0),
        #         2,
        #     )
        return annotated_frame

    def save_features(self, file_path):
        data = {
            "features": dict(self.vehicle_features),
            "violator_ids": list(self.violator_ids),
        }
        with open(file_path, "wb") as f:
            pickle.dump(data, f)

    def load_features(self, file_path):
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
        self.reid_cfg = self._init_reid_config(reid_config_path)
        self.model = make_model(self.reid_cfg, num_class=576, camera_num=20, view_num=8)
        self._load_model_weights(reid_model_path)
        self.transforms = build_transforms()
        self.tracks = {}
        self.next_id = 1
        self.first_video_features = first_video_data["features"]
        self.violator_ids = set(first_video_data["violator_ids"])
        self.feature_history = defaultdict(deque)
        self.CAR_CLASS_IDS = [2, 3, 5, 7]
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
        features = []
        for bbox in detections.xyxy:
            x1, y1, x2, y2 = map(int, bbox)
            vehicle_img = image[y1:y2, x1:x2]
            if vehicle_img.size == 0:
                features.append(None)
                continue
            vehicle_img = cv2.cvtColor(vehicle_img, cv2.COLOR_BGR2RGB)
            vehicle_img = Image.fromarray(vehicle_img)
            vehicle_img = (
                self.transforms(vehicle_img).unsqueeze(0).to(self.reid_cfg.MODEL.DEVICE)
            )
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

    def update_tracks(self, detections, features, frame_num):
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


def process_first_video(input_path, output_path, feature_file):
    tracker = VehicleTracker(
        yolo_model_path="yolov8s.pt",
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
            if elapsed > 0:
                fps_current = frame_num / elapsed
                remaining = (
                    (total_frames - frame_num) / fps_current if fps_current > 0 else 0
                )
                print(
                    f"Processed {frame_num}/{total_frames} frames, Elapsed: {elapsed:.2f}s, Remaining: {remaining:.2f}s, FPS: {fps_current:.2f}"
                )
    cap.release()
    out.release()
    tracker.save_features(feature_file)
    print(f"Saved features to {feature_file}")
    return {"features": tracker.vehicle_features, "violator_ids": tracker.violator_ids}


def process_second_video(input_path, output_path, first_video_data):
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
        if frame_num % 100 == 0:
            elapsed = time.time() - start_time
            if elapsed > 0:
                fps_current = frame_num / elapsed
                remaining = (
                    (total_frames - frame_num) / fps_current if fps_current > 0 else 0
                )
                print(
                    f"Processed {frame_num}/{total_frames} frames, Elapsed: {elapsed:.2f}s, Remaining: {remaining:.2f}s, FPS: {fps_current:.2f}"
                )
    cap.release()
    out.release()


def main():
    first_video_data = process_first_video(
        input_path="transreid/test/3.mp4",
        output_path="output_cam1.mp4",
        feature_file="first_video_features.pkl",
    )
    process_second_video(
        input_path="transreid/test/3_1.mp4",
        output_path="output_cam2.mp4",
        first_video_data=first_video_data,
    )


if __name__ == "__main__":
    main()
