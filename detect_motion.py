import numpy as np
import os
from tqdm import tqdm
from PIL import Image
from egomap.trainsition_classifier import CDCClassifier
from egomap.video.frame_generator import FrameGenerator
import cv2
from egomap.visualize_ import plot_function, html_table
import itertools
import argparse

from tabulate import tabulate
from tqdm import tqdm
import textwrap
import pickle
from egomap.video.video import get_video_info
#############################
#           CLI             #
#############################

GRID_RESOLUTION_X = 10
GRID_RESOLUTION_Y = 5
FRAME_RESOLUTION_X = 640
FRAME_RESOLUTION_Y = 480


def parseargs():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                     description=textwrap.dedent('''\
                                                                 Detects transitional motion given a videofile.'
                                                                 These files will be created in the destination folder:' 
                                                                    movement.pdf
                                                                    movement.npy
                                                                    movement.txt'''))
    parser.add_argument('--source-video', '-s', type=str, help="path to the videofile")
    parser.add_argument('--destination-folder', '-d', type=str, help="folder where the result is saved (multiple files will be created!)")
    
    
    args = parser.parse_args()
    return args 

def detect_motion(source_video, destination_folder):

    """
    This method runs through every frame in the video file and classifies each frame as stationary/moving (0/1).
    The result is saved to the save_to file.

    Parameters
    ----------
    source_video: str,
        path to the video file to be processed

    save_to: str,
         path to the file where the motion file is saved saved.

    Returns
    -------
        None
    """

    #########################################
    #       Movement classification         #
    #########################################
    transition_classifier = CDCClassifier([GRID_RESOLUTION_Y,GRID_RESOLUTION_X],
                                          smoothing_factor = 0.99,
                                          window = 60)

    fg = FrameGenerator(source_video, show_video_info=True)
    _, _, fps, _, frame_height, frame_width = get_video_info(source_video)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(os.path.join(destination_folder,"motion_viz.avi"), fourcc, int(fps), (frame_width, frame_height))
    movement = []
    for frame in tqdm(fg, desc="playing video", unit="frame"):
        cv_frame = frame
        frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY))
        frame = frame.resize([FRAME_RESOLUTION_X, FRAME_RESOLUTION_Y])
        is_moving = transition_classifier.is_in_transition(np.array(frame), verbosity=0)
        movement.append(is_moving)
        cv2.putText(cv_frame, "Moving:{0}".format(str(is_moving)), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
        cv_frame = cv2.cvtColor(cv_frame, cv2.COLOR_RGB2BGR)
        writer.write(cv_frame)
    #################################################
    #       Visualize movement as a function        #
    #################################################
    """
    plot_function(x=None,
                  y=movement,
                  size=600,
                  save_to=os.path.join(args.destination_folder, "movement"),
                  ylabel=movement,
                  xlabel="frames",
                  title="Detected movements during the video")
    """
    print ('[success] movement visual is saved to {}'.format(os.path.join(destination_folder, "movement.pdf")))

    #################################################
    #       Output movement stat as HTML table      #
    #################################################
    length_stat = [sum(list(g)) for b,g in itertools.groupby(movement) if b == 1]
    length_stat = length_stat if length_stat != [] else [0]
    movement_stat = [["Number of frames", len(movement)],
                     ["Number of frames with movement", sum(movement)],
                     ["Shortest movement (in frames)", min(length_stat)],
                     ["Longest movement (in frames)", max(length_stat)],
                     ["Average movement (in frames)", np.mean(length_stat)]]
    headers = ["Stat", "Values"]
    table = tabulate(movement_stat, headers, tablefmt="github")
    with open(os.path.join(args.destination_folder, "movement_stat.txt"), "w") as f:
        f.write(table)
    print(table)
    print ('[success] movement stat is saved to {}'.format(os.path.join(destination_folder, "movement_stat.txt")))


    with open(os.path.join(destination_folder, 'movement.pickle'), 'wb') as handle:
        pickle.dump(movement, handle, protocol=pickle.HIGHEST_PROTOCOL)


    print ('[success] movement data is saved to {}'.format(os.path.join(destination_folder, "movement.pickle")))

##############################
#           Main             #
##############################


if __name__ == "__main__":
    # Shine
    args = parseargs()
    
    detect_motion(**args.__dict__)
