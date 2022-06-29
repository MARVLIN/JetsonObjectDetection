import jetson.inference
import jetson.utils
import cv2

net = jetson.inference.detectNet('ssd-mobilenet-v2', threshold = 0.5)

cap = cv2.VideoCapture(f'v4l2src device=/dev/video0 io-mode=2 ! image/jpeg, width=(int)3840, height=(int)2160 !  nvjpegdec ! video/x-raw, format=I420 ! appsink', cv2.CAP_GSTREAMER)
cap.set(3,640)
cap.set(4,480)

while True:
    success, img = cap.read()
    imgCuda = jetson.utils.cudaFromNumpy(img)

    detections = net.Detect(imgCuda)
    for d in detections:
        # print(d)
        x1,y1,x2,y2 = int(d.Left),int(d.Top),int(d.Rifht),int(d.Bottom)
        className = net.GetClassDesc(d.ClassID)
        cv2.rectangle(img,(x1,y1), (x2,y2), (255,0,255),2)
        cv2.putText(img, className, (x1+5,y1+15), 0.75,(255,0,255),2)





    # img = jetson.utils.cudaToNumpy(imgCuda)
    cv2.imshow('image', img)
    cv2.waitKey(1)
