from person_verification.eye_blink import EyeBlinkVerification
import cv2
import time
import numpy as np
import os
import sys
from utils.util import read_yaml_file
from utils.exception_handler import AppException


ROOT_DIR = os.getcwd()
# Main config file path
CONFIG_FILE_NAME = "config.yaml"
CONFIG_FILE_PATH = os.path.join(ROOT_DIR,CONFIG_FILE_NAME)

class ImageCollection:
    def __init__(self, configs = read_yaml_file(CONFIG_FILE_PATH)):
        try:
            self.IMAGE_PATH = configs['artifacts_config']['artifacts_dir']
            self.number_of_images = configs['image_params']['number_of_images']
            self.blank_img = np.zeros((512, 512, 3), np.uint8)
            self.blink_obj = EyeBlinkVerification()
        except Exception as e:
            raise AppException(e,sys) from e



    def collect_images(self, uuid: list):
        try:
            if len(self.blink_obj.verify_eye_blink()) == 0:
                cv2.putText(self.blank_img, "You are not a real person!", (20,200), cv2.FONT_HERSHEY_COMPLEX,1, (0,0,255), 1)
                cv2.imshow('Message', self.blank_img)
                cv2.waitKey(3000)
                exit()
            
            else:
                for label in uuid:
                    img_path = os.path.join(self.IMAGE_PATH, label)
                    os.makedirs(img_path, exist_ok=True)
                    cap=cv2.VideoCapture(0)
                    print('Collecting images for {}'.format(label))
                    time.sleep(1)
                    for imgnum in range(self.number_of_images):
                        ret,frame=cap.read()
                        img = frame.copy()
                        cv2.putText(img, 'Collecting Images..',(30,30) , cv2.FONT_HERSHEY_DUPLEX , 1,(0,0,255),1)
                        cv2.putText(img, 'Please hold your face to the camera',(20,60) , cv2.FONT_HERSHEY_DUPLEX , 1,(0,0,255),1)
                        imagename=os.path.join(self.IMAGE_PATH,label,label+'_'+'{}.jpg'.format(imgnum))
                        cv2.imwrite(imagename,frame)
                        cv2.imshow('frame',img)
                        time.sleep(1)
                        
                        if cv2.waitKey(1) & 0xFF==ord('q'):
                            break
                    cap.release()
                    cv2.destroyAllWindows()
                    
        except Exception as e:
            raise AppException(e,sys) from e



if __name__ == "__main__":
    obj = ImageCollection()
    obj.collect_images(["Bappy123"])




