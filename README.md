# Real-Time Road Scene Perception for ADAS

A computer vision and deep learning project for road-scene perception, developed to explore core perception capabilities used in Advanced Driver Assistance Systems (ADAS).

The system integrates object perception, lane detection, and road damage detection into a single video-processing pipeline.

## Features

### 1. Object Perception
- Detects vehicles, pedestrians and other relevant road users using YOLO26n.
- Tracks detected objects across video frames using ByteTrack.
- Includes an experimental monocular depth-estimation component.

### 2. Lane Detection
- Detects lane markings using OpenCV.
- Uses grayscale conversion, Gaussian blur, Canny edge detection, region-of-interest masking and Hough Line Transform.

### 3. Road Damage Detection
- Detects road surface damage using a custom-trained YOLO26n model.
- Trained to identify:
  - Potholes
  - Cracks
  - Manholes

## Project Structure

```text
adas-road-perception/
│
├── data/
│   └── road_damage/
│       └── dataset/
│           └── data.yaml
│
├── evaluation/
├── models/
├── outputs/
│
├── src/
│   ├── object_perception.py
│   ├── lane_detection.py
│   └── pothole_detection.py
│
├── main.py
├── train.py
├── split_dataset.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Technologies Used

- Python
- OpenCV
- Ultralytics YOLO26n
- PyTorch
- ByteTrack
- NumPy

## Dataset

Road damage detection uses the **Road Damage Dataset – Potholes, Cracks and Manholes**.

- Images: 2,009 in the original dataset
- Classes: Pothole, Crack and Manhole
- Prepared training split: 400 images
- Prepared validation split: 100 images

Dataset source: https://zenodo.org/records/18528034

Dataset license: CC BY 4.0.

## Installation

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd adas-road-perception
```

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Model Weights

Model weights are excluded from this repository to avoid uploading large binary files.

Download or obtain the required weights separately:

- YOLO26n detection weights
- YOLO26 depth-estimation weights
- Custom-trained road damage detection weights

Place the weights in the locations expected by the Python modules, or update the model paths in the code.

## Running the Project

After installing dependencies and placing the required model weights in the correct locations:

```bash
python main.py
```

## Current Project Scope

This project currently integrates three perception modules:

1. Object perception
2. Lane detection
3. Road damage detection

The project is developed for learning and portfolio purposes. It is not a production-ready or safety-certified ADAS system.

## Future Improvements

- Traffic sign recognition
- Traffic signal state recognition
- Improved distance estimation
- Road hazard and risk assessment
- Performance optimization for edge devices

## Author

Developed as a computer vision and ADAS learning project.
