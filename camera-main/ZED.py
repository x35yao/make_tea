from collections import namedtuple
import pyzed.sl as sl
import numpy as np
from time import sleep, time

class ZEDCamera:

    def __init__(self, output=False, resolution='720', depth_mode='perf', fps=10, depth=True, color=True):
        self.depth = depth
        self.color = color
        self.fps = fps
        self.init = sl.InitParameters()
        self.output_path = output
        resolutions = {'720': sl.RESOLUTION.HD720,
                       '1080':sl.RESOLUTION.HD1080,
                       '2K'  :sl.RESOLUTION.HD2K}

        depthModes = {'perf': sl.DEPTH_MODE.PERFORMANCE,
                      'qual': sl.DEPTH_MODE.QUALITY,
                      'ultra': sl.DEPTH_MODE.ULTRA,}

        self.init.camera_resolution = resolutions[resolution]
        self.init.depth_mode = depthModes[depth_mode]
        self.init.camera_fps = self.fps
        self.cam = sl.Camera()
        return

    def _openCamera(self, totalAttempts=5):
        for attempt in range(totalAttempts):
            print('Opening ZED Camera...')
            status = self.cam.open(self.init)
            if status != sl.ERROR_CODE.SUCCESS:
                print('\nTry {} out of {}'.format(attempt+1,totalAttempts))
                print(repr(status))
                if attempt == (totalAttempts-1):
                    print('\n\n'+'-'*80)
                    print('Failed to open ZED')
                    print('Please Unplug the ZED and plug it back in!')
                    return False
            else:
                return True

    def __enter__(self):
        totalAttempts = 5
        for attempt in range(totalAttempts):
            if self._openCamera() == True:
                self.runtime = sl.RuntimeParameters()
                varNames = []
                if self.output_path:
                    self.record_param = sl.RecordingParameters(compression_mode=sl.SVO_COMPRESSION_MODE.H264, video_filename=self.output_path)
                    err = self.cam.enable_recording(self.record_param)
                if self.depth == True:
                    self.mat_depth = sl.Mat()
                    varNames.append('depth')
                if self.color == True:
                    self.mat_color = sl.Mat()
                    varNames.append('color')
                varNames.append('timestamp')
                self.Album = namedtuple('Album', varNames)
                return self
            else:
                sleep(5)
        raise IOError('Camera could not be opened, please try power cycling the ZED')

    def __exit__(self, exc_type, exc_value, traceback):
        print('Closing ZED...')
        self.cam.disable_recording()

    def startStream(self):
        self.__enter__()

    def closeStream(self):
        self.__exit__(None, None, None)

    def takePicture(self, emptyBuffer=False):
        """
        Returns an Album namedtuple.
        This method waits for a valid frame and builds the album
        to be returned.
        """
        start = time()
        status = self.cam.grab(self.runtime)
        while True:
            if status == sl.ERROR_CODE.SUCCESS:
                svo_image = sl.Mat()
                svo_depth = sl.Mat()
                self.cam.retrieve_image(svo_depth, sl.VIEW.DEPTH)
                self.cam.retrieve_image(svo_image, sl.VIEW.SIDE_BY_SIDE)
                color_image = np.asanyarray(svo_image.get_data())
                depth_image = np.asanyarray(svo_depth.get_data())
                return self.Album(depth_image, color_image, time())
            elif (time() - start) > 1:
                raise TimeoutError('The ZED is taking longer than 1 sec')
