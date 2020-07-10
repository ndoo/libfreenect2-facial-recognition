# libfreenect2-facial-recognition

Experiments in facial recognition by cross-referencing color and depth data
from 2nd generation Kinect sensors

### depth_auto_clip.py

This demo displays depth data while automatically clipping to between the
nearest point and 500mm from it. The distance can be overriden by passing the
`--range` (`-r`) argument.

Planar noise is not smoothed in the final output, but depth noise is smoothed
by taking the 20th lowest value, then taking a moving average of the last 20
samples. The number of samples buffered and averaged over can be overriden by
passing the `--samples` (`-s`) argument.

![Sample image from depth_auto_clip showing a Monstera Deliciosa being held in hand](https://github.com/ndoo/libfreenect2-facial-recognition/raw/master/depth_auto_clip.png)
