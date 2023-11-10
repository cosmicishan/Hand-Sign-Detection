# Hand-Sign-Detection

![image](https://github.com/cosmicishan/Hand-Sign-Detection/assets/37193732/20b3d935-7837-4e1c-aca6-1ed1ac9fa3f0)



This application can recognize and interpret 5 signs from gestures from images, videos, and live webcam feeds.

## Features

- **Run on Image:** Upload an image to recognize sign language gestures.
- **Run on Video:** Analyze sign language gestures in a video file.
- **Run on WebCam:** Real-time sign language interpretation through your webcam.
- **About App:** Learn more about Gesture Lingo and our mission.

## How to Use

### 1. Run Locally

To run the Gesture Lingo app locally, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/gesture-lingo.git
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the app:

   ```bash
   streamlit run gesture_lingo_app.py
   ```

4. Open your browser and go to [http://localhost:8501](http://localhost:8501).


## Technologies Used

- [Streamlit](https://streamlit.io/) - Frontend framework for creating web apps with Python.
- [MediaPipe](https://mediapipe.dev/) - Library for hand tracking and holistic detection.
- [OpenCV](https://opencv.org/) - Computer vision library for image and video processing.
- [scikit-learn](https://scikit-learn.org/) - Machine learning library for gesture classification.

## Model

Gesture Lingo uses a machine learning model trained to recognize sign language gestures. The model is trained on hand keypoints extracted from videos.


## Acknowledgments

- Special thanks to [MediaPipe](https://mediapipe.dev/) for providing the hand tracking and holistic detection library.
- Inspired by the goal of making communication more inclusive for the Deaf and Hard of Hearing community.
