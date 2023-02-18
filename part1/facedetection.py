from djitellopy import tello
import cv2
from cvzone.FaceDetectionModule import FaceDetector

me = tello.Tello()
me.connect()
print(me.get_battery())
me.streamoff()
me.streamon()

detector = FaceDetector(0.7)

while True:
    img = me.get_frame_read().frame
    img, bboxs = detector.findFaces(img)
    
    cv2.imshow('Image', img)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()