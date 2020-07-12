import argparse
import cv2
import numpy as np
import sys
from pylibfreenect2 import Freenect2, SyncMultiFrameListener
from pylibfreenect2 import FrameType, Registration, Frame

ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

ap.add_argument("-s", "--depth-smooth", type=int, default=20,
    help="Number of samples for moving average of nearest point")
ap.add_argument("-r", "--depth-range", type=int, default=80,
    help="Range to clip from nearest object, in millimeters")
ap.add_argument("-m", "--ir-min", type=int, default=1024,
    help="IR minimum value clip, out of a maximum value of 65535")
ap.add_argument("-M", "--ir-max", type=int, default=32768,
    help="IR maximum value clip, out of a maximum value of 65535")
args = vars(ap.parse_args())

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
anterior = 0

try:
    from pylibfreenect2 import CudaPacketPipeline
    pipeline = CudaPacketPipeline()
except:
    try:
        from pylibfreenect2 import OpenGLPacketPipeline
        pipeline = OpenGLPacketPipeline()
    except:
        try:
            from pylibfreenect2 import OpenCLPacketPipeline
            pipeline = OpenCLPacketPipeline()
        except:
            from pylibfreenect2 import CpuPacketPipeline
            pipeline = CpuPacketPipeline()

print("Packet pipeline:", type(pipeline).__name__)

fn = Freenect2()
num_devices = fn.enumerateDevices()
if num_devices == 0:
    print("No device connected!")
    sys.exit(1)

serial = fn.getDeviceSerialNumber(0)
device = fn.openDevice(serial, pipeline=pipeline)

types = FrameType.Color | FrameType.Ir | FrameType.Depth
listener = SyncMultiFrameListener(types)

# Register listeners
device.setColorFrameListener(listener)
device.setIrAndDepthFrameListener(listener)

# Start streams
device.startStreams(rgb=True,depth=True)

# Initialize buffer for moving average of nearest value
nearest_buffer = np.empty(args["depth_smooth"])
nearest_buffer[:] = np.NaN

# Iterate acquiring frames
while True:
    frames = listener.waitForNewFrame()

    depth = frames["depth"].asarray(np.float32)
    color = frames["color"].asarray()

    # Flip invalid depth value (0) to maximum value, to clean up blown-out patches at infinity
    depth[depth == 0] = np.amax(depth)

    # Apply clip on infrared image
    ir = np.uint8(
        (np.clip(frames["ir"].asarray(), args["ir_min"], args["ir_max"]) - args["ir_min"] - 1) /
        ((args["ir_max"] - args["ir_min"]) / 256)
        )
    #ir = np.uint8(frames["ir"].asarray() / 256)

    faces = faceCascade.detectMultiScale(ir, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.imshow("Face IR", cv2.resize(cv2.equalizeHist(ir[y:y+h, x:x+w]), (800, 800)))
        face_depth = depth[y:y+h, x:x+w]

        # Clip noise around nearest value by taking 10th lowest value
        nearest = np.partition(face_depth, 10, None)[9]
    
        # Determine nearest value by updating buffer and taking the average
        # Needed to combat flickering due to depth noise
        #nearest_buffer[:-1] = nearest_buffer[1:]
        #nearest_buffer[-1] = nearest
        #nearest = np.average(nearest_buffer)

        # Apply clip from nearest on depth image
        face_depth = np.clip(face_depth, nearest, nearest + args["depth_range"])
        face_depth -= nearest
        face_depth /= args["depth_range"]

        cv2.imshow("Face Depth", cv2.resize(face_depth, (800, 800)))
        


    listener.release(frames)

    key = cv2.waitKey(delay=1)
    if key == ord('q'):
        break

device.stop()
device.close()

sys.exit(0)
