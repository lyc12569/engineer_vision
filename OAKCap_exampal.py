import cv2
import numpy as np
import depthai as dai
import math
from OAK import OAKCap
from OAK import HostSpatialsCalc

def clamp(num, v0, v1):
    return max(v0, min(num, v1))
expTime = 20000
sensIso = 800
wbManual = 4000
EXP_STEP = 500  # us
ISO_STEP = 50 
WB_STEP = 200


cap=OAKCap()
pipeline,device=cap.startRgb()
stereo = cap.startDisparity(pipeline)

rgbWindowName = "rgb"
depthWindowName = "depth"
cv2.namedWindow(rgbWindowName)
cv2.namedWindow(depthWindowName)

with device:
    device.startPipeline(pipeline)
    controlQueue = device.getInputQueue('control')
    while True:
        frameRgb = cap.read(device)
        frameDisp = cap.readDisp(device,stereo)
        cv2.imshow('rgb',frameRgb)
        cv2.imshow('disp',frameDisp)
        key = cv2.waitKey(1)
        if key == ord('q'):
                break
        elif key in [ord('i'), ord('o'), ord('k'), ord('l')]:
            if key == ord('i'):
                expTime -= EXP_STEP
            if key == ord('o'):
                expTime += EXP_STEP
            if key == ord('k'):
                sensIso -= ISO_STEP
            if key == ord('l'):
                sensIso += ISO_STEP
            expTime = clamp(expTime, 1, 33000)
            sensIso = clamp(sensIso, 100, 1600)
            print("Setting manual exposure, time: ", expTime, "iso: ", sensIso)
            ctrl = dai.CameraControl()
            ctrl.setManualExposure(expTime, sensIso)
            controlQueue.send(ctrl)
        elif key in [ord('n'), ord('m')]:
            if key == ord('n'):
                wbManual -= WB_STEP
            if key == ord('m'):
                wbManual += WB_STEP
            wbManual = clamp(wbManual, 1000, 12000)
            print("Setting manual white balance, temperature: ", wbManual, "K")
            ctrl = dai.CameraControl()
            ctrl.setManualWhiteBalance(wbManual)
            controlQueue.send(ctrl)



