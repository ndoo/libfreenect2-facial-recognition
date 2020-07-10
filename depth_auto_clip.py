import argparse
import cv2
import numpy as np
import sys
from pylibfreenect2 import Freenect2, SyncMultiFrameListener
from pylibfreenect2 import FrameType, Registration, Frame

ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

ap.add_argument("-s", "--smoothing", type=int, default=20,
    help="Number of samples for moving average of nearest point")
ap.add_argument("-r", "--range", type=int, default=500,
    help="Range to clip from nearest object, in millimeters")
args = vars(ap.parse_args())

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

types = FrameType.Ir | FrameType.Depth
listener = SyncMultiFrameListener(types)

# Register listeners
device.setColorFrameListener(listener)
device.setIrAndDepthFrameListener(listener)

# Start streams
device.startStreams(rgb=False,depth=True)

# Initialize buffer for moving average of nearest value
nearest_buffer = np.empty(args["smoothing"])
nearest_buffer[:] = np.NaN

# Iterate acquiring frames
while True:
    frames = listener.waitForNewFrame()

    depth = frames["depth"].asarray(np.float32)

    # Flip invalid depth value (0) to maximum value, to clean up blown-out patches at infinity
    depth[depth == 0] = np.amax(depth)

    # Clip noise around nearest value by taking 20th lowest value
    nearest = np.partition(depth, 20, None)[19]
 
    # Determine nearest value by updating buffer and taking the average
    # Needed to combat flickering due to depth noise
    nearest_buffer[:-1] = nearest_buffer[1:]
    nearest_buffer[-1] = nearest
    nearest = np.average(nearest_buffer)

    # Apply clip from nearest
    depth = np.clip(depth, nearest, nearest + args["range"])
    depth -= nearest

    # Scale values to 0-1 for OpenCV and display
    cv2.imshow("Depth", depth / args["range"])

    listener.release(frames)

    key = cv2.waitKey(delay=1)
    if key == ord('q'):
        break

device.stop()
device.close()

sys.exit(0)
