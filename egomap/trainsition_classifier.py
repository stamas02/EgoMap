import cv2
import numpy as np
import argparse
import scipy.optimize as optimize
from egomap.video.frame_generator import FrameGenerator
from tqdm import tqdm


class CDCClassifier():
    """
    Cumulative Displacement Curves classifier used to classify transitional movement form anything else.
    The implementation is inspired by:

    @inproceedings{poleg2014temporal,
      title={Temporal segmentation of egocentric videos},
      author={Poleg, Yair and Arora, Chetan and Peleg, Shmuel},
      booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
      pages={2537--2544},
      year={2014}
    }
    """

    def __init__(self, grid,  smoothing_factor = 0.99, window = 0):
        # params for ShiTomasi corner detection
        # source: https://docs.opencv.org/3.1.0/d7/d8b/tutorial_py_lucas_kanade.html
        self.feature_params = dict(maxCorners=100,
                                   qualityLevel=0.3,
                                   minDistance=7,
                                   blockSize=7)
        self.lk_params = dict(winSize=(15, 15),
                              maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        self.smoothing_factor = smoothing_factor
        self.grid = grid
        self.CD = np.zeros((grid[0], grid[1],2)) #cumulative displacement curve.
        self.best_fit = np.zeros((grid[0], grid[1],1))
        self.prev_frame = None
        self.buffer = []
        self.window = window
        pass

    staticmethod
    def get_grid(frame, grid_height, grid_width):
        """
        Given an image, this function returns the grid blocks.

        Parameters
        ----------
        image : numpy array
            RGB or grayscale image.

        grid_height : int
            Height of the grid used to estimate optical flow. For example, 7 means that the image will be split into
            7 equal parts vertically.

        grid_width : int
            Width of the grid used to estimate optical flow. For example, 7 means that the image will be split into
            7 equal parts horizontally.

        Returns
        -------
        list of numpy arrays
            List of blocks as numpy arrays.

        Raises
        ------
        ValueError
            If the image cannot be divided into the given grid accurately.
        """
        h, w = frame.shape
        nrows = grid_height
        ncols = grid_width
        assert h % nrows == 0, "{} rows is not evenly divisble by {}".format(h, nrows)
        assert w % ncols == 0, "{} cols is not evenly divisble by {}".format(w, ncols)

        blocks = []
        for y in  range(0,h,h//nrows):
            for x in range(0, w, w // ncols):
                blocks.append(frame[y:y+h//nrows, x:x+h//ncols])

        return np.array(blocks)

    def _to_fit_exponential(self,data, a, b):
        result =  a*data[0,:]**2 + b*data[1,:]**2
        return result.flatten()

    def _to_fit_dexponential(self,data, a, b):
        dx = 2*a*data[0,:]
        dy = 2*b*data[1,:]
        grads = np.stack((dx,dy),axis=2)
        return grads.flatten()

    def _get_displacements(self, blocks_prev, blocks_current):
        """
        Give grid block extracted from the previous and the current frames
        this function calculates and returns a a displacement vector for each of these
        blocks.
        
        Parameters
        ----------
        blocks_prev: numpy array,
            array of blocks from a gray-scaled image. [n,h,w] where n is the number of blocks, 
            h and w are the height and weight of each block. 
        
        blocks_current: numpy array,
            array of blocks from a gray-scaled image. [n,h,w] where n is the number of blocks, 
            h and w are the height and weight of each block. 
            
        Returns
        -------
            A numpy array of 2 dimensional displacement vectors with size [n,2]
            where n is the number of blocks processed. Inf and Nan values are
            automatically set to 0
        """
        displacements = []
        for block_prev, block_current in zip(blocks_prev, blocks_current):
            p1 = cv2.goodFeaturesToTrack(block_prev, mask=None, **self.feature_params)
            if p1 is None:
                displacements.append(np.array([0,0]))
                continue
            p2, st, err = cv2.calcOpticalFlowPyrLK(block_prev, block_current, p1, None, **self.lk_params)
            good_p_1 = p1[st == 1].reshape(-1,2)
            good_p_2 = p2[st == 1].reshape(-1,2)
            displacements.append(np.mean(good_p_2-good_p_1, 0))
        return np.nan_to_num(np.array(displacements))

    def is_in_transition(self, frame, verbosity = 0):
        """
        Given a one channel greyscale image it classifies it to transitional or non-transitional frame.
        In order for this to work frames must be passed sequentially from a video file.

        Parameters
        ----------
        frame: numpy array,
            Must be one channel greyscale image

        Returns
        -------
            Classification results. False if not in transit else it is true.
        """

        if self.prev_frame is None:
            self.prev_frame = frame
            return False

        # Calculate displacements
        blocks_prev = CDCClassifier.get_grid(self.prev_frame, self.grid[0], self.grid[1])
        blocks_curr = CDCClassifier.get_grid(frame, self.grid[0], self.grid[1])
        displacements = self._get_displacements(blocks_prev, blocks_curr)
        displacements = displacements.reshape(self.CD.shape)
        #displacements = displacements * (displacements > 0.01)

        # update CD
        self.CD = self.CD*self.smoothing_factor+displacements*(1-self.smoothing_factor)

        # calculate the length of the vectors in CD
        CDnorm = np.sqrt(np.sum(self.CD**2,axis=2))

        # Create and centered grid to the minimum value of CDnorm
        center = np.array(np.unravel_index(np.argmin(CDnorm, axis=None), CDnorm.shape)).astype("float64")
        grid = np.array(np.meshgrid(range(0, self.CD.shape[1]), range(0, self.CD.shape[0]))).astype("float64")
        grid[0] -= center[1]
        grid[1] -= center[0]

        # fit curve to CDD
        params, pcov = optimize.curve_fit(self._to_fit_dexponential, grid, self.CD.flatten(),bounds=[[0.35,0.45],1])
        self.best_fit = self._to_fit_exponential(grid,params[0],params[1]).reshape(CDnorm.shape)
        perr = np.sum((self._to_fit_dexponential(grid,params[0],params[1])-self.CD.flatten())**2)
        if verbosity == 1:
            print("params: ax^2+by^2, a={0}, b={1}".format(params[0], params[1]))
            print("squared error: {0}".format(perr))

        self.prev_frame = frame
        self.buffer.append(perr < 150)
        if len(self.buffer) > self.window:
            self.buffer.pop(0)
        return np.mean(self.buffer) > 0.2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='CDCClassifier on vide')
    parser.add_argument('--video-file', type=str,
                        help='Path to the video file to be processed')
    parser.add_argument('--frame-width', type=int,
                        help='width of one frame')
    parser.add_argument('--frame-height', type=int,
                        help='height of one frame')
    parser.add_argument('--grid-width', type=int,
                        help='width of one grid')
    parser.add_argument('--grid-height', type=int,
                        help='height of one grid')

    return parser.parse_args()

def main():
    args = parse_args()

    fg = FrameGenerator(args.video_file, show_video_info=True)
    classifier = CDCClassifier((args.grid_height, args.grid_width), window=60)
    for frame in tqdm(fg,desc="playing video", unit="frame"):
        frame = cv2.resize(frame, (args.frame_width, args.frame_height))
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


        moving = classifier.is_in_transition(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), verbosity=1)
        print ("Moving: {0}".format(moving))
        nrows = args.grid_height
        ncols = args.grid_width
        #normCD = classifier.CD / np.expand_dims(np.sqrt(np.sum((classifier.CD ** 2), axis=2)), 2)


        # Draw vectors.
        CD = np.array(classifier.CD)
        CD *= 10
        for i, y in enumerate(range(0, args.frame_height, args.frame_height // nrows)):
            for j, x in enumerate(range(0, args.frame_width, args.frame_width // ncols)):
                vector_0 = ((x + (args.frame_width // ncols) // 2), (y + (args.frame_height // nrows) // 2))
                vector_1 = tuple((vector_0 + CD[i, j]).astype("int32"))
                cv2.line(bgr_frame, (vector_0[0], vector_0[1]), (vector_1[0], vector_1[1]), (255, 0, 0), 2)


        bestfit = classifier.best_fit/np.max(classifier.best_fit)
        bestfit = cv2.resize(bestfit, (args.frame_width, args.frame_height))

        cv2.imshow('bestfit', bestfit)
        cv2.imshow('video', bgr_frame)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()



