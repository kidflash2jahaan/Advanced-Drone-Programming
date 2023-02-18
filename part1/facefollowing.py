from djitellopy import tello
import cv2
import cvzone
from cvzone.FaceDetectionModule import FaceDetector

detector = FaceDetector(minDetectionCon=0.5)

# cap = cv2.VideoCapture(0)
# _, img = cap.read()

hi, wi, _ = 640, 480

xPID = cvzone.PID([0.3, 0, 0], wi // 2)
yPID = cvzone.PID([0.3, 0, 0], hi // 2, axis=1)
zPID = cvzone.PID([0.003, 0, 0], 30000)

myPlotX = cvzone.LivePlot(yLimit=[-100, 100], char='X')
myPlotY = cvzone.LivePlot(yLimit=[-100, 100], char='Y')
myPlotZ = cvzone.LivePlot(yLimit=[-100, 100], char='Z')

me = tello.Tello()
me.connect()
print(me.get_battery())
me.streamoff()
me.streamon()

while True:
    # _, img = cap.read()
    img = me.get_frame_read().frame
    img, bboxs = detector.findFaces(img)

    if bboxs:
        cx, cy = bboxs[0]["center"]
        x, y, w, h = bboxs[0]["bbox"]
        area = w * h

        xVal = int(xPID.update(cx))
        yVal = int(yPID.update(cy))
        zVal = int(zPID.update(area))
        print(zVal)

        imgPlotX = myPlotX.update(xVal)
        imgPlotY = myPlotY.update(yVal)
        imgPlotZ = myPlotZ.update(zVal)

        img = xPID.draw(img, [cx, cy])
        img = yPID.draw(img, [cx, cy])

    imgStacked = cvzone.stackImages(
        [img, imgPlotX, imgPlotY, imgPlotZ], 2, 0.75)

    # me.send_rc_control(0, zVal, yVal, xVal)
    me.send_rc_control(0, 0, 0, xVal)
    cv2.imshow('Image', imgStacked)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
