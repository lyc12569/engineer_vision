import depthai as dai
import cv2
import numpy as np

def getframe(queue):
    frame=queue.get()          #the last frame from the queue
    return frame.getCvFrame()  #convert the frame into opencv format

def getmonocamera(pipeline,isleft):
    mono=pipeline.createMonoCamera()  #mono   black and white

    mono.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

    if isleft:
        # create something called XLink in node,connecting your host and camera.
        mono.setBoardSocket(dai.CameraBoardSocket.LEFT) #get left camera
    else:
        mono.setBoardSocket(dai.CameraBoardSocket.RIGHT)
    
    return mono

def getstereopair(pipeline,monoleft,monoright):
    #configure stereo pair for depth estimation
    stereo = pipeline.createStereoDepth()  #then we configure the node stereodepth
    stereo.setLeftRightCheck(True)

    #configure left and right cameras to work as a stereo pair
    monoleft.out.link(stereo.left)
    monoright.out.link(stereo.right)
    return stereo

def mousecallback(event,x,y,flags,param):
    global mousex,mousey
    if event == cv2.EVENT_LBUTTONDOWN:
        mousex = x
        mousey = y



mousex=0
mousey=640
#define a pipeline
pipeline = dai.Pipeline()

monoleft = getmonocamera(pipeline,isleft= 1)
monoright = getmonocamera(pipeline,isleft= 0)

stereo = getstereopair(pipeline,monoleft,monoright)

#set xlinkout for disparity rectifiedleft and rectifiedright
xoutdisp = pipeline.createXLinkOut()
xoutdisp.setStreamName("disparity")

xoutrectifiedleft = pipeline.createXLinkOut()
xoutrectifiedleft.setStreamName("rectifiedleft")

xoutrectifiedright = pipeline.createXLinkOut()
xoutrectifiedright.setStreamName("rectifiedright")

stereo.disparity.link(xoutdisp.input)
stereo.rectifiedLeft.link(xoutrectifiedleft.input)
stereo.rectifiedRight.link(xoutrectifiedright.input)

with dai.Device(pipeline) as device:
    disparityqueue = device.getOutputQueue(name="disparity",maxSize=1,blocking=False)
    rectifiedleftqueue = device.getOutputQueue(name="rectifiedleft",maxSize=1,blocking=False)
    rectifiedrightqueue = device.getOutputQueue(name="rectifiedright",maxSize=1,blocking=False)

    #calcaulate a multiplier for colormapping disparity map
    disparitymultiplier = 255/stereo.getMaxDisparity()

    cv2.namedWindow("stereo pair")
    cv2.setMouseCallback("stereo pair",mousecallback)

    sidebyside = False

    while True:
        # get disparity map
        disparity = getframe(disparityqueue)
        
        #colormap disparity for display
        disparity = (disparity * disparitymultiplier).astype(np.uint8)
        
        disparity = cv2.applyColorMap(disparity,cv2.COLORMAP_JET)

        leftframe = getframe(rectifiedleftqueue)
        rightframe = getframe(rectifiedrightqueue)
        
        # remove_background_image = np.where(disparity <153 , 153, leftframe)
        # print(remove_background_image.shape)

        if sidebyside:
             # show side by side view
             imOut = np.hstack((leftframe,rightframe))
        else:
            # show overlapping frames
            imOut = np.uint8(leftframe/2+rightframe/2)

        imOut = cv2.cvtColor(imOut,cv2.COLOR_GRAY2BGR)

        imOut = cv2.line(imOut,(mousex,mousey),(640,mousey),(0,0,255),2)
        imOut = cv2.circle(imOut,(mousex,mousey),2,(255,255,128),2)
        cv2.imshow("stereo pair",imOut)
        cv2.imshow("Disparity",disparity)
        # cv2.imshow("aa",remove_background_image)

        key=cv2.waitKey(1)

        if key == ord('q'):  #quit when q is press
           break
        elif key == ord('t'):
            sidebyside = not sidebyside

