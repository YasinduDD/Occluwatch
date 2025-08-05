\# Occluwatch 🚦📷  

\*\*A multi-camera based road violation detection system with occlusion handling and vehicle re-identification\*\*



!\[Flow Diagram](docs/flow diagram.png)

---



\## 📌 Overview

\*\*Occluwatch\*\* is our Final Year Project for the \*\*BSc (Hons) in Computer Engineering\*\* at the University of Sri Jayewardenepura.  

It is an \*\*AI-powered multi-camera traffic surveillance system\*\* designed for \*\*highly congested urban areas\*\* where vehicle \*\*occlusion\*\*, \*\*identity switches\*\*, and \*\*complex camera layouts\*\* make traffic monitoring challenging.



Occluwatch can:

\- \*\*Handle occlusions\*\* and \*\*re‑identify\*\* vehicles across cameras

\- \*\*Track\*\* vehicles in real time

\- Detect \*\*traffic violations\*\*:

&nbsp; - 🚦 \*\*Red-light running\*\*

&nbsp; - ↔ \*\*Illegal lane changes / line crossing\*\*

&nbsp; - ⛔ \*\*Illegal parking\*\*

\- \*\*Recover license plates\*\* even under poor visibility

\- \*\*Automatically issue violation tickets\*\* via email through an integrated web portal



---



\## 🎯 Objectives

\- \*\*Occlusion handling\*\* – Maintain continuous tracking despite vehicles being hidden by other vehicles or objects.

\- \*\*Vehicle re‑identification\*\* – Match the same vehicle across multiple CCTV cameras.

\- \*\*Violation detection\*\* – Identify red-light, lane, and parking violations automatically.

\- \*\*Automated ticketing\*\* – Send violation notices to drivers via email, with attached evidence.

\- \*\*Real‑time operation\*\* – Achieve near real-time processing on standard GPU hardware.



---



\## 🛠 System Architecture



\### 1. \*\*Detection\*\*

* \*\*YOLOv11s\*\* fine-tuned on a \*\*Sri Lankan custom dataset\*\* for:

&nbsp; - Vehicles (cars, motorcycles, trucks, buses, three-wheelers)

&nbsp; - Traffic lights (red/yellow/green)

&nbsp; - License plates

* Optimized for \*\*speed/accuracy\*\* trade-off (≈50 FPS for detection).



\### 2. \*\*Violation Logic\*\*

* \*\*Red-light violation\*\* – Detect when a vehicle crosses the stop line during a red signal.
* \*\*Lane violation\*\* – Detect vehicles crossing forbidden lines (solid/double).
* \*\*Parking violation\*\* – Identify vehicles stopped in no-parking zones for >3 seconds.



\### 3. \*\*Tracking\*\*

* \*\*ByteTrack\*\* for robust \*\*multi-object tracking\*\*.
* Recovers tracks through \*\*occlusions\*\* using both high- and low-confidence detections.



\### 4. \*\*License Plate Recovery \& OCR\*\*

* Cropping, preprocessing (grayscale, histogram equalization, Gaussian blur, sharpening).
* \*\*Zero-DCE\*\* for low-light enhancement.
* \*\*Microsoft TrOCR\*\* for text extraction.
* Validation with Sri Lankan license plate patterns.



\### 5. \*\*Cross-Camera Re‑Identification\*\*

* \*\*TransReID\*\* (Transformer-based) for vehicle appearance feature extraction.
* \*\*GCN\*\* (Graph Convolutional Network) for \*\*spatio-temporal filtering\*\* to guide camera handoffs.
* Ensures vehicle identity consistency across multiple feeds.



\### 6. \*\*Reporting \& Ticketing\*\*

* Violation logs stored in CSV with:

&nbsp; - Plate number

&nbsp; - Timestamp

&nbsp; - Vehicle ID

&nbsp; - Evidence image paths

* \*\*Automated email notifications\*\* via Nodemailer.
* \*\*Manual verification\*\* triggered in case of plate/ID ambiguities.



\### 7. \*\*Web Platform\*\*

* \*\*Driver portal\*\* – Register/login, view ticket history, manage profile.
* \*\*Admin dashboard\*\* – Monitor pending/sent tickets, review flagged cases.
* \*\*Security\*\*:

&nbsp; - Sessions via `express-session`

&nbsp; - Password hashing via `bcrypt`

&nbsp; - Authentication via `passport` (local \& Google OAuth2)

&nbsp; - Secrets managed with `.env`



---



\## 📊 Performance

| Component                  | Model       | Accuracy (%) | FPS |

|---------------------------|-------------|--------------|-----|

| Vehicle Detection          | YOLOv11s    | mAP ≈ 78     | 50\* |

| Vehicle Re‑ID              | FastReID    | 81.9–82.9    | 12  |

| Vehicle Re‑ID              | \*\*TransReID\*\* | \*\*82.3–85.2\*\*| 11  |

| Full Pipeline (TransReID)  | —           | 80.4         | 11  |



\\\* Detection-only performance on RTX 2070; full pipeline runs at ≈11 FPS.



---



\##💻 Hardware Requirements

* GPU: NVIDIA RTX 2070 (8GB VRAM) or higher recommended
* RAM: ≥16 GB
* CPU: Intel i7 or equivalent
* OS: Linux / Windows 10+



\##👨‍💻 Authors

* S.D.A.Y.D. Dissanayake – Vehicle Re‑ID, Occlusion Handling, System Integration
* W.N.R. Fernando – Cross-Camera Mapping (GCN), Violation Detection
* K.S. Waththegama – Web Application, License Plate Recognition
* Supervisors – Dr. Randima Dinalankara, Mr. Lakshan Madhushanka



\##📬 Contact

For inquiries, please contact:

📧 y.dissanayake.yd@gmail.com | 📧 nirufernando03@gmail.com | 📧 waththegamaks@gmail.com

