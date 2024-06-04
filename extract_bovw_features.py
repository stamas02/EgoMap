import argparse
import textwrap
import numpy as np
import os
import pickle
from tqdm import tqdm
from egomap.features import bovw

import sys

sys.modules['bovw'] = bovw




from egomap.video.frame_generator import FrameGenerator


def parseargs():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                     description=textwrap.dedent('''\ 
                                                             Given a video, this script extract a feature
                                                             for each frame in the video. 
                                                             Outputs a file containing the features.:
                                                                - <video_file>_features.pickle'''))

    parser.add_argument('--source-video', '-s', type=str,
                        help=textwrap.dedent(''' Path to the video file '''))
    parser.add_argument('--vocabulary-file', '-v',
                        type=str,
                        default="model/bowv.pickle",
                        help=textwrap.dedent('''vocabulary file obtained by running learn_vocabulary.py script'''))

    args = parser.parse_args()
    return args


################################
#           Script             #
################################
def extract_features(source_video, vocabulary_file):
    """
    Extract features for each stationary section in a given video.
    The result will be saved to a pickle file next to the input video file.
    The result file will have the name: <input_video_file_name>_features.piclke

    Parameters
    ----------
    source_video: str,
        path to the dataset.

    vocabulary_file: str
        Path to the trained bovw object created using 'learn_vocabulary.py'

    Returns
    -------
        None
    """
    filename = os.path.basename(os.path.splitext(source_video)[0])
    dirname = os.path.dirname(os.path.splitext(source_video)[0])
    output_file = os.path.join(dirname, filename + "_features.pickle")

    # load model file
    model = bovw.BOVW.load(vocabulary_file)

    # Ectract features
    features = []
    fg = FrameGenerator(source_video, show_video_info=True)
    for frame in tqdm(fg, desc="playing video", unit="frame"):
        f = np.array(model.extract_feature(frame), dtype=bool)
        features.append(f)

    # calcu late and save mean
    with open(output_file, 'wb') as handle:
        pickle.dump(features, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("[success] features where successfully extracted. "
          "Features where saved into {0}"
          .format(output_file), 'green')


##############################
#           Main             #
##############################

if __name__ == "__main__":

    # get args from user
    args = parseargs().__dict__

    # run main script
    extract_features(**args)
