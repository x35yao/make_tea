import pyrealsense2 as rs
import numpy as np
from collections import namedtuple
from time import time
import cv2

# Has to be run with root priviledges for some reason

class RealSense2:
    def __init__(self, height=640, width=480, depth=True, color=True):
        self.width = width
        self.height= height
        self.depth = depth
        self.color = color
        self.running = False
        return

    def __enter__(self):
        try:
            self.config
        except AttributeError:
            self.pipeline = rs.pipeline()
            self.config = rs.config()

            self.varNames = []
            if self.depth:
                print('Configuring Depth Stream: {}x{}'.format(self.width, self.height))
                self.config.enable_stream(rs.stream.depth, self.height, self.width, rs.format.z16, 30)
                self.varNames.append('depth')
            if self.color:
                print('Configuring Color Stream: {}x{}'.format(self.width, self.height))
                self.config.enable_stream(rs.stream.color, self.height, self.width, rs.format.bgr8, 30)
                self.varNames.append('color')
            self.varNames.append('timestamp')
            # A place to store images
            self.Album = namedtuple('Album', self.varNames)

        if not self.running:
            print('Starting Pipeline')
            self.pipeline.start(self.config)
            self.running = True
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.running:
            print('Stopping Pipeline')
            self.pipeline.stop()
            self.running = False

    def __del__(self):
        if self.running:
            print('Stopping Pipeline')
            self.pipeline.stop()

    def startStream(self):
        self.__enter__()

    def closeStream(self):
        self.__exit__(None, None, None)

    def _getFrames(self):
        pics = []
        start = time()
        while True:
            flag=1
            frames = self.pipeline.wait_for_frames()

            if self.depth:
                depth_frame = frames.get_depth_frame()
                if not depth_frame:
                    flag*=0
                else:
                    depth_image = np.asanyarray(depth_frame.get_data())
                    pics.append(depth_image)

            if self.color:
                color_frame = frames.get_color_frame()
                if not color_frame:
                    flag*=0
                else:
                    color_image = np.asanyarray(color_frame.get_data())
                    pics.append(color_image)
            pics.append(time())
            if flag:
                return self.Album(*pics)
            elif (time() - start) > 1:
                raise TimeoutError('The Camera is taking longer than 1 sec')

    def takePicture(self, emptyBuffer=False):
        """
        Returns an Album namedtuple.
        This method starts a stream if needed, waits for the frames and builds the album
        to be returned. After the frames after returned, the stream is returned
        to its initial state.
        """
        initialState = self.running
        try:
            self.startStream()
            if emptyBuffer:
                for i in range(10):
                    self._getFrames()
            return self._getFrames()
        finally:
            if not initialState:
                self.closeStream()

    def videoStream(self):
        """
        Returns a generator object.
        This generator is to be used with the send method and accepts boolean
        values.
        If the value is True a StopIteration exception is raised.
        If false, the generator returns an Album named tuple with the images
        taken at that time.

        >>> cam = RealSense2()
        >>> reel = cam.videoStream()
        >>> album = reel.send(False)
        """
        try:
            self.startStream()
            while True:
                try:
                    quit = yield self._getFrames()
                    if quit:
                        break
                except TimeoutError:
                    print('Was there a missing frame?')
        finally:
            self.closeStream()


if __name__ == '__main__':
    with RealSense2() as cam:
        key = ''
        reel = cam.videoStream()
        # Start the generator
        reel.send(None)
        while True:
            try:
                key = cv2.waitKey(5)
                album = reel.send(key==113)
                cv2.imshow('RealSense2 Color', album.color)
                cv2.imshow('RealSense2 Depth', album.depth)
            except StopIteration:
                break
        cv2.destroyAllWindows()
