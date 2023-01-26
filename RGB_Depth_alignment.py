import cv2
import numpy as np
import depthai as dai
from calc import HostSpatialsCalc
import math

def find_ore(color_image):
        # 高斯滤波
    cv2.GaussianBlur(color_image, (5, 5), 3, dst=color_image)
    # 把BGR通道转化为HSV色彩空间
    hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
    # 识别矿石的颜色特征
    ore_image = cv2.inRange(hsv_image, thresh_low, thresh_high)
    # 对识别后的图像进行腐蚀与膨胀，消除较小的连通域
    kernal = cv2.getStructuringElement(0, (3, 3))
    cv2.erode(ore_image, kernal, dst=ore_image)
    cv2.dilate(ore_image, kernal, dst=ore_image)
	# 轮廓识别
    contours, hierarchy = cv2.findContours(ore_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        # 计算所有轮廓的面积
        area_size = list(map(cv2.contourArea, contours))
        # 取最大面积
        max_size = max(area_size)
        max_area_index = area_size.index(max_size)
        # 若面积在阈值范围内，则认为识别到了矿石
        if min_area_size < max_size < max_area_size:
            # box = cv2.boundingRect(contours[max_area_index])
            rect = cv2.minAreaRect(contours[max_area_index])
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            left_point_x = np.min(box[:, 0])
            right_point_x = np.max(box[:, 0])
            top_point_y = np.min(box[:, 1])
            bottom_point_y = np.max(box[:, 1])
            left_point_y = box[:, 1][np.where(box[:, 0] == left_point_x)][0]
            right_point_y = box[:, 1][np.where(box[:, 0] == right_point_x)][0]
            top_point_x = box[:, 0][np.where(box[:, 1] == top_point_y)][0]
            bottom_point_x = box[:, 0][np.where(box[:, 1] == bottom_point_y)][0]
            vertices = np.array([[top_point_x, top_point_y], [left_point_x, left_point_y],
                      [bottom_point_x, bottom_point_y],  [right_point_x, right_point_y]])
            return vertices
    return None

params = {'hue_low': 8, 'saturation_low': 110, 'value_low': 0,
          'hue_high': 30, 'saturation_high': 255, 'value_high': 255,
          'min_area_size': 5000, 'max_area_size': 120000, 'offset_x': 350,
          'offset_y': 0, 'kp': 0.008, 'kd': 0.003, 'max_speed': 0.8,
          'forward_velocity': 0.2}
thresh_low = (params['hue_low'], params['saturation_low'], params['value_low'])
thresh_high = (params['hue_high'], params['saturation_high'], params['value_high'])
min_area_size = params['min_area_size']
max_area_size = params['max_area_size']

Camera_intrinsic = {

    "mtx": np.array([[652.90522946, 0., 321.31072448], [0., 653.00569689, 255.77694781], [0., 0., 1.]],
                    dtype=np.double),
    "dist": np.array([[-0.23397356, 0.39554466, 0.01463653, -0.00176085, -0.48223752]], dtype=np.double),

}

# Optional. If set (True), the ColorCamera is downscaled from 1080p to 720p.
# Otherwise (False), the aligned depth is automatically upscaled to 1080p
downscaleColor = True
fps = 30
# The disparity is computed at this resolution, then upscaled to RGB resolution
monoResolution = dai.MonoCameraProperties.SensorResolution.THE_720_P

# Create pipeline
pipeline = dai.Pipeline()
device = dai.Device()
queueNames = []

# Define sources and outputs
camRgb = pipeline.create(dai.node.ColorCamera)
left = pipeline.create(dai.node.MonoCamera)
right = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)

rgbOut = pipeline.create(dai.node.XLinkOut)
disparityOut = pipeline.create(dai.node.XLinkOut)
depthout = pipeline.create(dai.node.XLinkOut)

rgbOut.setStreamName("rgb")
queueNames.append("rgb")
disparityOut.setStreamName("disp")
queueNames.append("disp")
depthout.setStreamName("dep")
queueNames.append("dep")

#Properties
camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setFps(fps)
if downscaleColor: camRgb.setIspScale(2, 3)
# For now, RGB needs fixed focus to properly align with depth.
# This value was used during calibration
try:
    calibData = device.readCalibration2()
    lensPosition = calibData.getLensPosition(dai.CameraBoardSocket.RGB)
    if lensPosition:
        camRgb.initialControl.setManualFocus(lensPosition)
except:
    raise
left.setResolution(monoResolution)
left.setBoardSocket(dai.CameraBoardSocket.LEFT)
left.setFps(fps)
right.setResolution(monoResolution)
right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
right.setFps(fps)

stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
# LR-check is required for depth alignment
stereo.setLeftRightCheck(True)
stereo.setDepthAlign(dai.CameraBoardSocket.RGB)

# Linking
camRgb.isp.link(rgbOut.input)
left.out.link(stereo.left)
right.out.link(stereo.right)
stereo.disparity.link(disparityOut.input)
stereo.depth.link(depthout.input)

hostSpatials = HostSpatialsCalc(device)
delta = 5
hostSpatials.setDeltaRoi(delta)

# Connect to device and start pipeline
with device:
    device.startPipeline(pipeline)

    frameRgb = None
    frameDisp = None
    framedepth = None

    # Configure windows; trackbar adjusts blending ratio of rgb/depth
    rgbWindowName = "rgb"
    depthWindowName = "depth"
    blendedWindowName = "rgb-depth"
    cv2.namedWindow(rgbWindowName)
    cv2.namedWindow(depthWindowName)
    cv2.namedWindow(blendedWindowName)
    
    while True:
        latestPacket = {}
        latestPacket["rgb"] = None
        latestPacket["disp"] = None
        latestPacket["dep"] = None

        queueEvents = device.getQueueEvents(("rgb", "disp","dep"))
        for queueName in queueEvents:
            packets = device.getOutputQueue(queueName).tryGetAll()
            if len(packets) > 0:
                latestPacket[queueName] = packets[-1]

        if latestPacket["rgb"] is not None:
            frameRgb = latestPacket["rgb"].getCvFrame()
            cv2.imshow(rgbWindowName, frameRgb)
        

        if latestPacket["disp"] is not None:
            frameDisp = latestPacket["disp"].getFrame()
            maxDisparity = stereo.initialConfig.getMaxDisparity()
            # Optional, extend range 0..95 -> 0..255, for a better visualisation
            if 1: frameDisp = (frameDisp * 255. / maxDisparity).astype(np.uint8)
            # Optional, apply false colorization
            # if 1: frameDisp = cv2.applyColorMap(frameDisp, cv2.COLORMAP_HOT)
            frameDisp = np.ascontiguousarray(frameDisp)
            cv2.imshow(depthWindowName, frameDisp)

        if latestPacket["dep"] is not None:
            framedepth = latestPacket["dep"].getFrame()
            cv2.imshow("depth",framedepth)

        # Blend when both received
        if frameRgb is not None and frameDisp is not None and framedepth is not None:
            # Need to have both frames in BGR format before blending
            if len(frameDisp.shape) < 3:
                frameDisp = cv2.cvtColor(frameDisp, cv2.COLOR_GRAY2BGR)
                # print(frameDisp.shape)
            # blended = cv2.addWeighted(frameRgb, rgbWeight, frameDisp, depthWeight, 0)
            remove_back_rbg = np.where(frameDisp < 170, 153, frameRgb)

            bbox = find_ore(remove_back_rbg)
            if bbox is not None:
                undetected_count = 0
                # x, y, w, h = bbox
                center_x = round((bbox[0][0] + bbox[2][0]) / 2)
                center_y = round((bbox[0][1] + bbox[2][1]) / 2)
                # get 3D zuobiao
                spatials, centroid = hostSpatials.calc_spatials(framedepth, (center_x,center_y))

                obj = np.array([[-100, 100, 0], [-100, -100, 0], [100,-100 , 0],
                        [100, 100, 0]],dtype=np.float64)  # 世界坐标
                pnts = np.array([bbox[0],bbox[1],bbox[2],bbox[3]],dtype=np.float64) # 像素坐标
    
                success,rvec, tvec = cv2.solvePnP(obj, pnts, Camera_intrinsic["mtx"], Camera_intrinsic["dist"],flags=cv2.SOLVEPNP_ITERATIVE)

                distance=math.sqrt(spatials['x']**2+spatials['y']**2+spatials['z']**2)/10  # 测算距离

                rvec_matrix = cv2.Rodrigues(rvec)[0]
                proj_matrix = np.hstack((rvec_matrix, rvec))
                eulerAngles = -cv2.decomposeProjectionMatrix(proj_matrix)[6]  # 欧拉角
                pitch, yaw, roll = eulerAngles[0], eulerAngles[1], eulerAngles[2]
                rot_params = np.array([yaw, pitch, roll])  # 欧拉角 数组
                # 这里pitch要变为其相反数(不知道为啥)
                cv2.putText(remove_back_rbg, "%.2fcm,%.2f,%.2f,%.2f" % (distance, yaw, -pitch, roll),
                (remove_back_rbg.shape[1] - 500, remove_back_rbg.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)


                # cv2.rectangle(remove_back_rbg, (x, y), (x + w, y + h), (73, 245, 189), thickness=2)
                cv2.drawContours(remove_back_rbg,[bbox],0,(73, 245, 189), thickness=2)

                cv2.circle(remove_back_rbg, (center_x, center_y), 10, (73, 245, 189), cv2.FILLED)
                cv2.putText(remove_back_rbg, "Target detected.", (0, 24), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 121, 242), 2)
                cv2.putText(remove_back_rbg, f"x: {spatials['x']} mm", (bbox[0][0], bbox[0][1] - 48), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 121, 242), 2)
                cv2.putText(remove_back_rbg, f"y: {spatials['y']} mm", (bbox[0][0], bbox[0][1] - 12), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 121, 242), 2)
                cv2.putText(remove_back_rbg, f"z: {spatials['z']} mm", (bbox[0][0], bbox[0][1] + 24), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 121, 242), 2)
    
            cv2.imshow(blendedWindowName, remove_back_rbg)
            frameRgb = None
            frameDisp = None

        if cv2.waitKey(1) == ord('q'):
            break