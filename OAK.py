import cv2
import numpy as np
import depthai as dai
import math

def getmonocamera(pipeline, isleft):
        mono = pipeline.createMonoCamera()  # mono   black and white

        mono.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)

        if isleft:
            # create something called XLink in node,connecting your host and camera.
            mono.setBoardSocket(dai.CameraBoardSocket.LEFT)  # get left camera
        else:
            mono.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        return mono

def getstereopair(pipeline, monoleft, monoright):
    # configure stereo pair for depth estimation
    stereo = pipeline.createStereoDepth()  # then we configure the node stereodepth
    stereo.setLeftRightCheck(True)

    # configure left and right cameras to work as a stereo pair
    monoleft.out.link(stereo.left)
    monoright.out.link(stereo.right)
    return stereo

def getframe(queue):
    frame=queue.get()          #the last frame from the queue
    return frame.getCvFrame()  #convert the frame into opencv format

class OAKCap:
    def startRgb(self):
        fps = 30
        # The disparity is computed at this resolution, then upscaled to RGB resolution
        monoResolution = dai.MonoCameraProperties.SensorResolution.THE_720_P

        # Create pipeline
        pipeline = dai.Pipeline()
        device = dai.Device()
        queueNames = []

        # Define sources and outputs
        camRgb = pipeline.create(dai.node.ColorCamera)
        controlIn = pipeline.create(dai.node.XLinkIn)
        controlIn.setStreamName('control')
        rgbOut = pipeline.create(dai.node.XLinkOut)
        rgbOut.setStreamName("rgb")
        queueNames.append("rgb")
        #Properties
        camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
        camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        camRgb.setIspScale(2, 3)  # 1080P -> 720P
        camRgb.setFps(fps)
        # Linking
        camRgb.isp.link(rgbOut.input)
        controlIn.out.link(camRgb.inputControl)
        return pipeline,device

    def startDisparity(self, pipeline):
        monoleft = getmonocamera(pipeline, isleft=1)
        monoright = getmonocamera(pipeline, isleft=0)
        stereo = getstereopair(pipeline, monoleft, monoright)
        # set xlinkout for disparity rectifiedleft and rectifiedright
        xoutdisp = pipeline.createXLinkOut()
        xoutdisp.setStreamName("disparity")
        stereo.disparity.link(xoutdisp.input)
        return stereo 
        

    def read(self, device): #get rgbframe
        latestPacket = {}
        latestPacket["rgb"] = None
        queueEvents = device.getQueueEvents(("rgb"))
        for queueName in queueEvents:
            packets = device.getOutputQueue(queueName).tryGetAll()
            if len(packets) > 0:
                latestPacket[queueName] = packets[-1]

        if latestPacket["rgb"] is not None:
            frameRgb = latestPacket["rgb"].getCvFrame()
            return frameRgb

    def readDisp(self,device,stereo): #get disparity frame
        disparityqueue = device.getOutputQueue(name="disparity",maxSize=1,blocking=False)
        disparitymultiplier = 255/stereo.getMaxDisparity()
        disparity = getframe(disparityqueue)
        #colormap disparity for display
        disparity = (disparity * disparitymultiplier).astype(np.uint8)
        
        disparity = cv2.applyColorMap(disparity,cv2.COLORMAP_JET)
        return disparity


class HostSpatialsCalc:
    # We need device object to get calibration data
    def __init__(self, device):
        calibData = device.readCalibration()
        # Required information for calculating spatial coordinates on the host
        self.monoHFOV = np.deg2rad(
            calibData.getFov(dai.CameraBoardSocket.LEFT))

        # Values
        self.DELTA = 5
        self.THRESH_LOW = 200  # 20cm
        self.THRESH_HIGH = 30000  # 30m

    def setLowerThreshold(self, threshold_low):
        self.THRESH_LOW = threshold_low

    def setUpperThreshold(self, threshold_low):
        self.THRESH_HIGH = threshold_low

    def setDeltaRoi(self, delta):
        self.DELTA = delta

    # Check if input is ROI or point. If point, convert to ROI
    def _check_input(self, roi, frame):
        if len(roi) == 4:
            return roi
        if len(roi) != 2:
            raise ValueError(
                "You have to pass either ROI (4 values) or point (2 values)!")
        # Limit the point so ROI won't be outside the frame
        self.DELTA = 5  # Take 10x10 depth pixels around point for depth averaging
        x = min(max(roi[0], self.DELTA), frame.shape[1] - self.DELTA)
        y = min(max(roi[1], self.DELTA), frame.shape[0] - self.DELTA)
        return (x-self.DELTA, y-self.DELTA, x+self.DELTA, y+self.DELTA)

    def _calc_angle(self, frame, offset):
        return math.atan(math.tan(self.monoHFOV / 2.0) * offset / (frame.shape[1] / 2.0))

    # roi has to be list of ints
    def calc_spatials(self, depthFrame, roi, averaging_method=np.mean):
        # If point was passed, convert it to ROI
        roi = self._check_input(roi, depthFrame)
        xmin, ymin, xmax, ymax = roi

        # Calculate the average depth in the ROI.
        depthROI = depthFrame[ymin:ymax, xmin:xmax]
        inRange = (self.THRESH_LOW <= depthROI) & (
            depthROI <= self.THRESH_HIGH)

        averageDepth = averaging_method(depthROI[inRange])

        centroid = {  # Get centroid of the ROI
            'x': int((xmax + xmin) / 2),
            'y': int((ymax + ymin) / 2)
        }

        midW = int(depthFrame.shape[1] / 2)  # middle of the depth img width
        midH = int(depthFrame.shape[0] / 2)  # middle of the depth img height
        bb_x_pos = centroid['x'] - midW
        bb_y_pos = centroid['y'] - midH

        angle_x = self._calc_angle(depthFrame, bb_x_pos)
        angle_y = self._calc_angle(depthFrame, bb_y_pos)

        spatials = {
            'z': averageDepth,
            'x': averageDepth * math.tan(angle_x),
            'y': -averageDepth * math.tan(angle_y)
        }
        return spatials, centroid
