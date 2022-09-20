from person_verification.eye_blink import EyeBlinkVerification
import cv2
import time
import numpy as np
import os
import uuid


blink_obj = EyeBlinkVerification()

IMAGE_PATH='artifacts'
labels=['Bappy']
number_of_images=15

blank_img = np.zeros((512, 512, 3), np.uint8)


def collect_images():
    if len(blink_obj.verify_eye_blink()) == 0:
        cv2.putText(blank_img, "You are not a real person!", (20,200), cv2.FONT_HERSHEY_COMPLEX,1, (0,0,255), 1)
        cv2.imshow('Message', blank_img)
        cv2.waitKey(3000)
        exit()
    
    else:
        for label in labels:
            img_path = os.path.join(IMAGE_PATH, label)
            os.makedirs(img_path, exist_ok=True)
            cap=cv2.VideoCapture(0)
            print('Collecting images for {}'.format(label))
            time.sleep(1)
            for imgnum in range(number_of_images):
                ret,frame=cap.read()
                img = frame.copy()
                cv2.putText(img, 'Collecting Images..',(30,30) , cv2.FONT_HERSHEY_DUPLEX , 1,(0,0,255),1)
                cv2.putText(img, 'Please hold your face to the camera',(20,60) , cv2.FONT_HERSHEY_DUPLEX , 1,(0,0,255),1)
                imagename=os.path.join(IMAGE_PATH,label,label+'.'+'{}.jpg'.format(str(uuid.uuid1())))
                cv2.imwrite(imagename,frame)
                cv2.imshow('frame',img)
                time.sleep(1)
                
                if cv2.waitKey(1) & 0xFF==ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()



if __name__ == "__main__":
    collect_images()




