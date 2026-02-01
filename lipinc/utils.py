
# from google.colab.patches import cv2_imshow # comment this if not using colab
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



datFile =  "B:\download\TruthSeeker-app\lipinc\shape_predictor_68_face_landmarks.dat"
detector_pre = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(datFile)
# n_frames = 5 #number of local frames



"""##Creating Local frames and Global frames"""





###################################################################################################

def create_face_array(video_path):
    mp_face = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    face_array = []
    lip_boxes = []

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

        x, y, bw, bh = cv2.boundingRect(np.array(list(zip(xs, ys))))
        lip_crop = frame[y:y+bh, x:x+bw]

        if lip_crop.size == 0:
            continue

        lip_crop = cv2.resize(lip_crop, (144, 64))
        face_array.append(lip_crop)

    cap.release()
    return np.asarray(face_array)


###################################################################################################

def find_global_frames(face_array,local_face_id,lh,lw,lch,rch,predictor):

    simPoseVideos = []
    g_id = []
    face_id =-1
    while face_id < (len(face_array))-1:
      face_id+=1
      if face_id == local_face_id:
        continue

      frame = face_array[face_id]
      imggr =  cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)

      faces= detector_pre(imggr)

      top = max(0, faces[0].top())
      bottom = min(faces[0].bottom(), imggr.shape[0])
      left = max(0, faces[0].left())
      right = min(faces[0].right(), imggr.shape[1])

      # try:
      landmark = predictor(imggr,faces[0])

      mypoints =[]
      for n in range(68):
        x=landmark.part(n).x
        y=landmark.part(n).y
        mypoints.append([x,y])
      points =np.array(mypoints[48:])
      bbox = cv2.boundingRect(points)
      x,y,w,h = bbox
      lipU_x,lipU_y = landmark.part(62).x,landmark.part(62).y
      lipL_x,lipL_y = landmark.part(66).x,landmark.part(66).y

      lipLeft_x,lipLeft_y = landmark.part(60).x,landmark.part(60).y
      lipRight_x,lipRight_y = landmark.part(64).x,landmark.part(64).y

      Outerlip_Left_x,Outerlip_Left_y = landmark.part(48).x,landmark.part(48).y
      CheekLeft_x, CheekLeft_y = landmark.part(4).x,landmark.part(4).y


      Outerlip_Right_x,Outerlip_Right_y = landmark.part(54).x,landmark.part(54).y
      CheekRight_x, CheekRight_y = landmark.part(12).x,landmark.part(12).y

      total_width = right -left
      total_height = bottom-top

      #calculating the distance and ratios between lower and upper lips
      lheight = int(math.dist([lipU_x,lipU_y],[lipL_x,lipL_y])/total_height*100)
      lwidth = int(math.dist([lipLeft_x,lipLeft_y],[lipRight_x,lipRight_y])/total_width*100)
      lcheek = int(math.dist([Outerlip_Left_x,Outerlip_Left_y],[CheekLeft_x,CheekLeft_y])/total_width*100)
      rcheek= int(math.dist([Outerlip_Right_x,Outerlip_Right_y],[CheekRight_x,CheekRight_y])/total_width*100)

      # print(int(lh*100),lheight )
      # print(int(lw*100),lwidth )
      # print(int(lch*100),lcheek )
      # print(int(rch*100),rcheek )
      ran = 3

      if int(lh*100) in range(lheight-ran,lheight+ran) and int(lw*100) in range(lwidth-ran,lwidth+ran) and int(lch*100) in range(lcheek-ran,lcheek+ran) and int(rch*100) in range(rcheek-ran,rcheek+ran):
        lipcrop = frame[y:y+h,x:x+w]
        lipcrop = cv2.resize(lipcrop, (144,64))
        simPoseVideos.append(lipcrop)
        g_id.append(face_id)
        face_id+=3


        if len(simPoseVideos) == 3:
          return simPoseVideos,g_id

      # except Exception as e:
      #   continue
    return 0,0

###################################################################################################

def find_LGframes(n_frames,face_array,predictor):

  #saving the local frames and global
  Local_frames = 0
  Global_frames = 0

  adj_f = []
  adj_f_id =[]
  No_face_count = 0

  # extract all frames
  for face_id in range(len(face_array)):


      frame = face_array[face_id]
      imggr =  cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)

      faces= detector_pre(imggr)
      success = 1

      try:
        top = max(0, faces[0].top())
        bottom = min(faces[0].bottom(), imggr.shape[0])
        left = max(0, faces[0].left())
        right = min(faces[0].right(), imggr.shape[1])

        landmark = predictor(imggr,faces[0])

        mypoints =[]
        for n in range(68):
          x=landmark.part(n).x
          y=landmark.part(n).y
          mypoints.append([x,y])
        points =np.array(mypoints[48:])
        bbox = cv2.boundingRect(points)
        x,y,w,h = bbox

        lipU_x,lipU_y = landmark.part(62).x,landmark.part(62).y
        lipL_x,lipL_y = landmark.part(66).x,landmark.part(66).y

        lipLeft_x,lipLeft_y = landmark.part(60).x,landmark.part(60).y
        lipRight_x,lipRight_y = landmark.part(64).x,landmark.part(64).y

        Outerlip_Left_x,Outerlip_Left_y = landmark.part(48).x,landmark.part(48).y
        CheekLeft_x, CheekLeft_y = landmark.part(4).x,landmark.part(4).y


        Outerlip_Right_x,Outerlip_Right_y = landmark.part(54).x,landmark.part(54).y
        CheekRight_x, CheekRight_y = landmark.part(12).x,landmark.part(12).y

        total_width = right -left
        total_height = bottom-top


        #calculating the distance between lower and upper lips
        lheight = math.dist([lipU_x,lipU_y],[lipL_x,lipL_y])/total_height
        lwidth = math.dist([lipLeft_x,lipLeft_y],[lipRight_x,lipRight_y])/total_width
        lcheek = math.dist([Outerlip_Left_x,Outerlip_Left_y],[CheekLeft_x,CheekLeft_y])/total_width
        rcheek= math.dist([Outerlip_Right_x,Outerlip_Right_y],[CheekRight_x,CheekRight_y])/total_width


        height_ratio = lheight*100
        if height_ratio < 1:
          adj_f = []
          # success+=1 
          # if success>2:
          #   adj_f = []
          continue
        else:
          lipcrop = frame[y:y+h,x:x+w]
          lipcrop = cv2.resize(lipcrop, (144,64))
          adj_f.append(lipcrop)
          adj_f_id.append(face_id)
        
          # cv2_imshow(lipcrop)
          # print(len(adj_f))
          if len(adj_f)==n_frames:
            print("Local Frames = 5")
            Global_frames,g_id = find_global_frames(face_array,face_id,lheight,lwidth,lcheek,rcheek,predictor)
            if Global_frames == 0:
              adj_f = []
              adj_f_id = []
              continue
            Local_frames= adj_f
            break

      except Exception as e:
        No_face_count+=1
        
  print("Face/lips not Detected in {} out of {} frames".format(No_face_count,len(face_array)))
  
  Global_frames = np.asarray(Global_frames)
  Local_frames = np.asarray(Local_frames)

  # print(Global_frames.shape,Local_frames.shape)

  return Local_frames,Global_frames,adj_f_id,g_id

###################################################################################################

def create_residue(frames):
  residue = []
  for index in range(1,len(frames)):
    residue_frame = abs(frames[index]-frames[index-1])
    residue.append(residue_frame)
  residue = np.asarray(residue)
  return residue

###################################################################################################

def get_color_structure_frames(n_frames, path):
    face_array = create_face_array(path)

    print("Number of frames with lips:", len(face_array))

    if len(face_array) == 0:
        raise RuntimeError("No face detected in any frame")

    if len(face_array) < n_frames + 3:
        raise RuntimeError("Insufficient lip frames")

    local_frames = face_array[:n_frames]
    global_frames = face_array[n_frames:n_frames+3]

    combined_frames = np.concatenate((local_frames, global_frames))
    residue_frames = np.abs(np.diff(combined_frames, axis=0))

    l_id = list(range(n_frames))
    g_id = list(range(n_frames, n_frames + 3))

    return False, face_array[0], combined_frames, residue_frames, l_id, g_id


# combined_frames,residue_frames = get_color_structure_frames(n_frames,path)