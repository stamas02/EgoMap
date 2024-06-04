from egomap.video.video import get_video_info, prettify_video_info
import cv2


class FrameGenerator:
    def __init__(self, video_source, show_video_info=True, every_nth_frame = 1):
        """
        Init
        Parameters
        ----------
        video_source: str
            path to the video file you want to read.

        show_video_info: bool
            if true then video info is printed to the console

        every_nth_frame: int
            every nth frame will be yielded by the generator. 1 means that every single frame will be yielded.
        """

        video_file, frame_count, fps, length, height, width = get_video_info(video_source)
        if show_video_info:
            print(prettify_video_info(video_source, frame_count, fps, length, height, width))
        self._frame_count = frame_count
        self.fps = fps
        self.length=length
        self.resolution=(height, width)
        self._cap = cv2.VideoCapture(video_file)
        self._every_nth_frame = every_nth_frame
        if not self._cap.isOpened():
            raise ValueError("could not open video file: {0}".format(video_file))

    def __iter__(self):
        """
        reads frame from an opencv capture objects and yields until this object is not closed.

        Returns
        -------
        a nupy array representing an RGB frame
        """
        cnt = -1
        while (self._cap.isOpened()):
            ret, frame = self._cap.read()
            cnt += 1
            if cnt % self._every_nth_frame != 0:
                continue
            if frame is None:
                print("error")
                return
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            yield frame
        self._cap.release()
        raise StopIteration()

    def __len__(self):
        return self._frame_count // self._every_nth_frame


