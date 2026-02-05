import os
import dlib
from glob import glob
import shutil
import imutils
import cv2
import numpy as np
from os import listdir
import math
import mediapipe as mp
from sklearn.metrics import roc_curve, auc
import sys

# ------------------ DYNAMIC PATH SETUP ------------------
# This grabs the path we set in st.py so it works on any machine
datFile = os.environ.get("DLIB_LANDMARK_PATH")

# Fallback if running standalone
if datFile is None or not os.path.exists(datFile):
    # Try looking in current directory
    datFile = "shape_predictor_68_face_landmarks.dat"

if not os.path.exists(datFile):
    print(f"WARNING: Dlib shape predictor not found at {datFile}")
    # We don't crash here, but downstream functions will fail if called

detector_pre = dlib.get_frontal_face_detector()
try:
    predictor = dlib.shape_predictor(datFile)
except RuntimeError:
    print("Error loading dlib predictor. Check path.")
    predictor = None

# ------------------ FUNCTIONS ------------------

def create_face_array(video_path):
    # Initialize MediaPipe Face Mesh
    # using 'mp.solutions' is the safe way to avoid AttributeErrors
    mp_face_mesh = mp.solutions.face_mesh
    
    face_array = []
    
    # Context manager ensures resources are freed
    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as mp_face:

        cap = cv2.VideoCapture(video_path)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = mp_face.process(rgb)

            if not result.multi_face_landmarks:
                continue

            landmarks = result.multi_face_landmarks[0].landmark

            # MediaPipe lip landmark indices
            lip_ids = [
                61, 146, 91, 181, 84, 17,
                314, 405, 321, 375, 291, 308
            ]

            xs = [int(landmarks[i].x * w) for i in lip_ids]
            ys = [int(landmarks[i].y * h) for i in lip_ids]

            # Get bounding box of lips
            x, y, bw, bh = cv2.boundingRect(np.array(list(zip(xs, ys))))
            
            # Add padding or ensure valid crop
            # Simple boundary checks
            y = max(0, y)
            x = max(0, x)
            bh = min(bh, h - y)
            bw = min(bw, w - x)

            lip_crop = frame[y:y+bh, x:x+bw]

            if lip_crop.size == 0:
                continue

            try:
                lip_crop = cv2.resize(lip_crop, (144, 64))
                face_array.append(lip_crop)
            except Exception as e:
                continue

        cap.release()
        
    return np.asarray(face_array)


def find_global_frames(face_array, local_face_id, lh, lw, lch, rch, predictor):
    simPoseVideos = []
    g_id = []
    face_id = -1
    
    # Safety check for predictor
    if predictor is None:
        return 0, 0

    while face_id < (len(face_array)) - 1:
        face_id += 1
        if face_id == local_face_id:
            continue

        frame = face_array[face_id]
        imggr = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        faces = detector_pre(imggr)
        if len(faces) == 0:
            continue

        landmark = predictor(imggr, faces[0])

        mypoints = []
        for n in range(68):
            x = landmark.part(n).x
            y = landmark.part(n).y
            mypoints.append([x, y])
            
        points = np.array(mypoints[48:])
        bbox = cv2.boundingRect(points)
        x, y, w, h = bbox
        
        lipU_x, lipU_y = landmark.part(62).x, landmark.part(62).y
        lipL_x, lipL_y = landmark.part(66).x, landmark.part(66).y

        lipLeft_x, lipLeft_y = landmark.part(60).x, landmark.part(60).y
        lipRight_x, lipRight_y = landmark.part(64).x, landmark.part(64).y

        Outerlip_Left_x, Outerlip_Left_y = landmark.part(48).x, landmark.part(48).y
        CheekLeft_x, CheekLeft_y = landmark.part(4).x, landmark.part(4).y

        Outerlip_Right_x, Outerlip_Right_y = landmark.part(54).x, landmark.part(54).y
        CheekRight_x, CheekRight_y = landmark.part(12).x, landmark.part(12).y

        total_width = max(1, faces[0].right() - faces[0].left())
        total_height = max(1, faces[0].bottom() - faces[0].top())

        lheight = int(math.dist([lipU_x, lipU_y], [lipL_x, lipL_y]) / total_height * 100)
        lwidth = int(math.dist([lipLeft_x, lipLeft_y], [lipRight_x, lipRight_y]) / total_width * 100)
        lcheek = int(math.dist([Outerlip_Left_x, Outerlip_Left_y], [CheekLeft_x, CheekLeft_y]) / total_width * 100)
        rcheek = int(math.dist([Outerlip_Right_x, Outerlip_Right_y], [CheekRight_x, CheekRight_y]) / total_width * 100)

        ran = 3

        if int(lh * 100) in range(lheight - ran, lheight + ran) and \
           int(lw * 100) in range(lwidth - ran, lwidth + ran) and \
           int(lch * 100) in range(lcheek - ran, lcheek + ran) and \
           int(rch * 100) in range(rcheek - ran, rcheek + ran):
            
            lipcrop = frame[y:y+h, x:x+w]
            lipcrop = cv2.resize(lipcrop, (144, 64))
            simPoseVideos.append(lipcrop)
            g_id.append(face_id)
            face_id += 3

            if len(simPoseVideos) == 3:
                return simPoseVideos, g_id

    return 0, 0


def get_color_structure_frames(n_frames, path):
    # Extract faces using MediaPipe
    face_array = create_face_array(path)

    print("Number of frames with lips:", len(face_array))

    if len(face_array) == 0:
        raise RuntimeError("No face detected in any frame")

    if len(face_array) < n_frames + 3:
        raise RuntimeError(f"Video too short. Need at least {n_frames + 3} frames, got {len(face_array)}")

    # Simplified selection strategy for speed/demo
    # (Note: Original code had find_LGframes but this version uses direct slicing)
    local_frames = face_array[:n_frames]
    global_frames = face_array[n_frames:n_frames+3]

    combined_frames = np.concatenate((local_frames, global_frames))
    residue_frames = np.abs(np.diff(combined_frames, axis=0))

    l_id = list(range(n_frames))
    g_id = list(range(n_frames, n_frames + 3))

    return False, face_array[0], combined_frames, residue_frames, l_id, g_id