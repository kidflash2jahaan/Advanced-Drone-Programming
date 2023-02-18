from djitellopy import tello
import cv2
import cvzone
from cvzone.FaceDetectionModule import FaceDetector

cap = cv2.VideoCapture(0)
detector = FaceDetector(minDetectionCon=0.8)

_, img = cap.read()
hi, wi, _ = img.shape

xPID = cvzone.PID([0.3, 0, 0.1], wi // 2)
yPID = cvzone.PID([0.3, 0, 0.1], hi // 2, axis=1)
zPID = cvzone.PID([0.003, 0, 0.1], 30000)

myPlotX = cvzone.LivePlot(yLimit=[-100, 100], char='X')
myPlotY = cvzone.LivePlot(yLimit=[-100, 100], char='Y')
myPlotZ = cvzone.LivePlot(yLimit=[-100, 100], char='Z')

# me = tello.Tello()
# me.connect()
# print(me.get_battery())
# me.streamoff()
# me.streamon()

while True:
    _, img = cap.read()
    # img = me.get_frame_read().frame
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

        # cv2.putText(img, str(xVal), (50, 100),
        # cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        # cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        # cv2.line(img, (wi // 2, 0), (wi // 2, hi), (255, 0, 255), 1)
        # cv2.line(img, (wi // 2, hi // 2), (cx, cy), (255, 0, 255), 3)
        # error = wi // 2 - cx
        # cv2.putText(img, str(error), (50, 100),
        #             cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    imgStacked = cvzone.stackImages(
        [img, imgPlotX, imgPlotY, imgPlotZ], 2, 0.75)

    cv2.imshow('Image', imgStacked)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
