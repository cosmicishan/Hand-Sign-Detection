import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import tempfile
import av
from io import BytesIO
import pickle
import pandas as pd
import math
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration


import warnings
warnings.filterwarnings('ignore')
  

with open ('rf.pkl', 'rb') as f:
    rf = pickle.load(f)

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

DEMO_VIDEO = 'demo_video.mp4'
DEMO_IMAGE = 'demo_image.jpg'

def extract_keypoints(results):
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([lh, rh])

st.title('Gesture Lingo')

st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 400px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 400px;
        margin-left: -350px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title('Face Mesh Application using MediaPipe')
st.sidebar.subheader('Parameters')

app_mode = st.sidebar.selectbox('Choose the App mode',
['About App','Run on Image','Run on Video', 'Run on WebCam']
)

if app_mode =='About App':

    st.markdown(
        '''
        <style>
        /* Define custom CSS styles */
        .header-text {
            color: red;
            font-size: 36px; /* Increased font size for the header */
        }
        </style>
        ''',
        unsafe_allow_html=True
    )

    # Markdown for text content with the increased header size
    st.markdown('<span class="header-text">Introducing Gesture Lingo: <br>Bridging Communication Barriers with Technology!</span>', unsafe_allow_html=True)
    st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 360px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 360px;
        margin-left: -360px;
    }
    </style>
    """,
    unsafe_allow_html=True,
    )
    
    st.markdown(
        '''
        <style>
        /* Define custom CSS styles */
        .header-text {
            color: blue;
            font-size: 24px;
        }
        .emphasis-text {
            font-style: italic;
            font-weight: bold;
        }
        .list-item {
            color: green;
        }
        </style>
        ''',
        unsafe_allow_html=True
    )

    # Markdown for text content
    st.markdown(
        '''# <span class="header-text">About us</span>''',
        unsafe_allow_html=True)
    
    st.markdown(
    '''

        Hey this is <span class="emphasis-text">Ishan Purohit</span> from <span class="emphasis-text">Gesture Lingo</span>.

        At Gesture Lingo, we are on a mission to make the world more inclusive and accessible by harnessing the power of cutting-edge technology. Our startup is dedicated to developing a state-of-the-art sign language interpreter, and we are excited to showcase a glimpse of our innovation through our demo website.

        Our demo website features an incredible hand gesture detection system that can recognize and interpret gestures from images, videos, and even live webcam feeds. This technology holds the potential to revolutionize communication for the Deaf and Hard of Hearing community, making it easier for them to interact with the world around them.

        With Gesture Lingo, we are taking a significant step towards breaking down communication barriers and ensuring that everyone has the opportunity to express themselves effectively, regardless of their hearing ability. Explore our demo and witness the future of sign language interpretation. Join us on our journey to create a more inclusive world through the language of gestures.
        ''',
        unsafe_allow_html=True
    )

elif app_mode =='Run on Video':

    st.set_option('deprecation.showfileUploaderEncoding', False)

    st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 400px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 400px;
        margin-left: -400px;
    }
    </style>
    """,
    unsafe_allow_html=True,
        )
    
    st.sidebar.markdown('---')
    detection_confidence = st.sidebar.slider('Min Detection Confidence', min_value =0.0,max_value = 1.0,value = 0.5)
    tracking_confidence = st.sidebar.slider('Min Tracking Confidence', min_value = 0.0,max_value = 1.0,value = 0.5)

    st.sidebar.markdown('---')

    st.markdown(' ## Output')

    stframe = st.empty()
    video_file_buffer = st.sidebar.file_uploader("Upload a video", type=[ "mp4", "mov",'avi','asf', 'm4v','webm' ])
    tfflie = tempfile.NamedTemporaryFile(delete=False)

    with mp_holistic.Holistic(min_detection_confidence=detection_confidence, min_tracking_confidence=tracking_confidence) as holistic:
    
        
        if not video_file_buffer:
            cap = cv2.VideoCapture(DEMO_VIDEO)
            tfflie.name = DEMO_VIDEO
    
        else:
            tfflie.write(video_file_buffer.read())
            cap = cv2.VideoCapture(tfflie.name)

        while cap.isOpened():

            ret, frame = cap.read()
        
            # Recolor Feed
            try:
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            except:
                if not video_file_buffer:
                    cap = cv2.VideoCapture(DEMO_VIDEO)
                    tfflie.name = DEMO_VIDEO
            
                else:
                    tfflie.write(video_file_buffer.read())
                    cap = cv2.VideoCapture(tfflie.name)

            image.flags.writeable = False        
            
            # Make Detections
            results = holistic.process(image)

            image.flags.writeable = True   
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            try:

                data = extract_keypoints(results)

                X = pd.DataFrame([data])
                class_label = rf.predict(X)[0]

                height, width, _ = image.shape

                FONT_SCALE = 2e-3  # Adjust for larger font size in all images
                THICKNESS_SCALE = 1e-3  # Adjust for larger thickness in all images
                
                font_scale = min(width, height) * FONT_SCALE
                thickness = math.ceil(min(width, height) * THICKNESS_SCALE)
                                
                cv2.rectangle(image, (0,0), (int(0.3 * width), int(0.07 * height)), (256, 7, 3), -1)

                cv2.putText(image, class_label, (int(0.05 * width), int(0.05 * height)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

            except:

                pass

            stframe.image(image,channels = 'BGR',use_column_width=True)


elif app_mode == 'Run on WebCam':

    st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 400px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 400px;
        margin-left: -400px;
    }
    </style>
    """,
    unsafe_allow_html=True)

    st.sidebar.markdown('---')
    detection_confidence = st.sidebar.slider('Min Detection Confidence', min_value =0.0,max_value = 1.0,value = 0.5)
    tracking_confidence = st.sidebar.slider('Min Tracking Confidence', min_value = 0.0,max_value = 1.0,value = 0.5)

    st.sidebar.markdown('---')
    stframe = st.empty()

    cap = cv2.VideoCapture(0)

    with mp_holistic.Holistic(min_detection_confidence=detection_confidence, min_tracking_confidence=tracking_confidence) as holistic:
    
        while cap.isOpened():

                ret, frame = cap.read()

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


                image.flags.writeable = False        
                
                # Make Detections
                results = holistic.process(image)

                image.flags.writeable = True   
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                try:

                    data = extract_keypoints(results)

                    X = pd.DataFrame([data])
                    class_label = rf.predict(X)[0]

                    height, width, _ = image.shape

                    FONT_SCALE = 2e-3  # Adjust for larger font size in all images
                    THICKNESS_SCALE = 1e-3  # Adjust for larger thickness in all images
                    
                    font_scale = min(width, height) * FONT_SCALE
                    thickness = math.ceil(min(width, height) * THICKNESS_SCALE)
                                    
                    cv2.rectangle(image, (0,0), (int(0.3 * width), int(0.07 * height)), (256, 7, 3), -1)

                    cv2.putText(image, class_label, (int(0.05 * width), int(0.05 * height)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

                except:

                    pass

                stframe.image(image,channels = 'BGR',use_column_width=True)

elif app_mode =='Run on Image':

    drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=1)

    st.sidebar.markdown('---')

    st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 400px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 400px;
        margin-left: -400px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

    detection_confidence = st.sidebar.slider('Min Detection Confidence', min_value =0.0,max_value = 1.0,value = 0.5)
    tracking_confidence = st.sidebar.slider('Min Tracking Confidence', min_value = 0.0,max_value = 1.0,value = 0.5)
    st.sidebar.markdown('---')

    img_file_buffer = st.sidebar.file_uploader("Upload an image", type=[ "jpg", "jpeg",'png'])

    with mp_holistic.Holistic(min_detection_confidence=detection_confidence, min_tracking_confidence=tracking_confidence) as holistic:
    
        
        if img_file_buffer is not None:
                image_bytes = BytesIO(img_file_buffer.read())
                frame = cv2.imdecode(np.frombuffer(image_bytes.read(), np.uint8), cv2.IMREAD_COLOR)
        else:
            frame = cv2.imread('demo_image.jpg')
        
        # Recolor Feed
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False        
        
        # Make Detections
        results = holistic.process(image)

        image.flags.writeable = True   
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        try:

            data = extract_keypoints(results)

            X = pd.DataFrame([data])
            class_label = rf.predict(X)[0]

            height, width, _ = image.shape

            FONT_SCALE = 2e-3  # Adjust for larger font size in all images
            THICKNESS_SCALE = 1e-3  # Adjust for larger thickness in all images
            
            font_scale = min(width, height) * FONT_SCALE
            thickness = math.ceil(min(width, height) * THICKNESS_SCALE)
                            
            cv2.rectangle(image, (0,0), (int(0.4 * width), int(0.06 * height)), (256, 7, 3), -1)

            cv2.putText(image, class_label, (int(0.05 * width), int(0.04 * height)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

        except Exception as e:

            print(e)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        st.subheader('Output Image')
        st.image(image,use_column_width= True)
