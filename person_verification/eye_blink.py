import cv2  
import time   
import sys
import os                       
import dlib                                    #for face and landmark detection
from scipy.spatial import distance as dist     #for calculating dist b/w the eye landmarks
from imutils import face_utils                 #to get the landmark ids of the left and right eyes ----you can do this manually too
from utils.exception_handler import AppException


class EyeBlinkVerification:
    def __init__(self):
        try:    
            self.cam = cv2.VideoCapture(0)
            self.blink_thresh = 0.5
            self.succ_frame = 2
            self.count_frame = 0
            
            #-------Eye landmarks------#
            (self.L_start, self.L_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
            (self.R_start, self.R_end) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

            #------Initializing the Models for Landmark and face Detection---------#
            self.detector = dlib.get_frontal_face_detector()
            self.landmark_predict = dlib.shape_predictor(os.path.join('model','shape_predictor_68_face_landmarks.dat'))
        except Exception as e:
            raise AppException(e,sys) from e
       

    #------function to calulate the EAR-----------#
    def calculate_EAR(self,eye) :
        try:
            #---calculate the verticle distances---#
            y1 = dist.euclidean(eye[1] , eye[5])
            y2 = dist.euclidean(eye[2] , eye[4])

            #----calculate the horizontal distance---#
            x1 = dist.euclidean(eye[0],eye[3])

            #----------calculate the EAR--------#
            EAR = (y1+y2) / x1
            return EAR
        except Exception as e:
            raise AppException(e,sys) from e
    

    #---------Mark the eye landmarks-------#
    def mark_eyeLandmark(self,img , eyes):
        try:
            for eye in eyes:
                pt1,pt2 = (eye[1] , eye[5])
                pt3,pt4 = (eye[0],eye[3])
                cv2.line(img,pt1,pt2,(200,00,0),2)
                cv2.line(img, pt3, pt4, (200, 0, 0), 2)
            return img
        except Exception as e:
            raise AppException(e,sys) from e

    
    def verify_eye_blink(self):
        try:
            blink_count_list = []
            t_end = time.time() + 5   # 5 seconds

            while time.time() < t_end:
                success, img_ = self.cam.read()

                img = img_.copy()
                #---converting frame to gray scale to pass to detector----#
                img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                #---detecting the faces---#
                faces = self.detector(img_gray)

                for face in faces :
                    #----landmark detection-----#
                    shape = self.landmark_predict(img_gray,face)
                    #----converting the shape class directly to a list of (x,y) cordinates-----#
                    shape = face_utils.shape_to_np(shape)
                    for lm in shape:
                        cv2.circle(img,(lm),3,(10,2,200))
                    #----parsing the landmarks list to extract lefteye and righteye landmarks--#
                    lefteye = shape[self.L_start : self.L_end]
                    righteye = shape[self.R_start: self.R_end]

                    #-----Calculate the EAR (eyes aspect ratio)---#
                    left_EAR = self.calculate_EAR(lefteye)
                    right_EAR = self.calculate_EAR(righteye)

                    #----mark the landmarks----#
                    img = self.mark_eyeLandmark(img,[lefteye,righteye])

                    #-----Avg of left and right eye EAR----#
                    avg = (left_EAR+right_EAR)/2
                    
                    if avg<self.blink_thresh :
                        self.count_frame+=1
                    
                    elif self.count_frame >= self.succ_frame :
                        cv2.putText(img, 'Blink Detected',(30,30) , cv2.FONT_HERSHEY_DUPLEX , 1,(0,255,0),1)
                        # print("Blink Detected")
                        blink_count_list.append("Blink Detected")
                        self.count_frame=0

                    else:
                        cv2.putText(img, 'Blink your eys',(30,30) , cv2.FONT_HERSHEY_DUPLEX , 1,(0,0,255),1)

                cv2.imshow("Eye Blink Verification", img)
                print(len(blink_count_list))

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            self.cam.release()
            cv2.destroyAllWindows()
            return blink_count_list

        except Exception as e:
            raise AppException(e,sys) from e
            