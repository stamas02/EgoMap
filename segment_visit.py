import argparse
import textwrap
import os
import pickle
from tqdm import tqdm
import numpy as np

from egomap.video.video import get_video_info
from egomap.camera_motion import estimate_z_transition, get_optical_flow
from egomap.video.frame_generator import FrameGenerator


def parseargs():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                     description=textwrap.dedent('''\ 
                                                             Given a video, and a computed optical flow 
                                                             outputs a visit segmentation:
                                                                - <video_file>visit_segmentation_predicted.txt'''))

    parser.add_argument('--source-video', '-s', type=str,
                        help=textwrap.dedent(''' Path to the video file '''))
    parser.add_argument('--smooth', type=float, default=0.99,
                        help=textwrap.dedent('''temporal smoothing factor'''))
    parser.add_argument('--threshold', type=float, default=1.008,
                        help=textwrap.dedent('''threshold to differentiate between transition and stationary.'''))
    parser.add_argument('--min-visit-seconds', type=int, default=10,
                        help=textwrap.dedent('''threshold to differentiate between transition and stationary.'''))
    parser.add_argument('--min-transition-seconds', type=int, default=3,
                        help=textwrap.dedent('''threshold to differentiate between transition and stationary.'''))
    parser.add_argument('--grid-width', type=int, default=10,
                        help=textwrap.dedent('''Grid resolution for optical flow calculation (width). 
                                                E.g. 7 would split the image horizontally into 7 equal parts.'''))
    parser.add_argument('--grid-height', type=int, default=5,
                        help=textwrap.dedent('''Grid resolution for optical flow calculation (width). 
                                                E.g. 7 would split the image vertically into 7 equal parts.'''))

    args = parser.parse_args()
    return args


def compute_optical_flow(video_file, grid_width, grid_height):
    """
    Compute optical flow for each frame in a video.

    Parameters
    ----------
    video_file : str
        Path to the video file.

    grid_width : int
        Width of the grid used to estimate optical flow.

    grid_height : int
        Height of the grid used to estimate optical flow.

    Returns
    -------
    numpy array
        Array containing optical flow for each frame.
    """

    optical_flow = []
    frame_generator = FrameGenerator(video_file, show_video_info=True)
    num_frames = len(frame_generator)
    prev_frame = next(iter(frame_generator))

    for i, frame in enumerate(frame_generator, start=1):
        progress = i / num_frames * 100
        print("{:.2f}% ({}/{}) frames: Computing optical flow".format(progress, i, num_frames))

        flow = get_optical_flow(prev_frame, frame, grid_width, grid_height)
        optical_flow.append(flow)
        prev_frame = frame

    # Insert a zero optical flow at the beginning for consistency
    optical_flow.insert(0, optical_flow[0])

    return np.array(optical_flow)


def segment_visit(video_file, smooth, threshold, min_visit_seconds, min_transition_seconds, grid_width, grid_height,
                  output_file):
    _, frame_count, fps, _, _, _ = get_video_info(video_file)
    min_visit_frames = min_visit_seconds * fps
    min_transition_frames = min_transition_seconds * fps

    optical_flows = compute_optical_flow(video_file, grid_width, grid_height)
    num_frames = len(optical_flows)
    optical_flow_exponential_smooth = optical_flows[0]
    visit_segmentation = []
    for i, optical_flow in enumerate(optical_flows):
        progress = i / num_frames * 100
        print("{:.2f}% ({}/{}) frames: Estimating z transition".format(progress, i, num_frames))
        optical_flow_exponential_smooth = smooth * optical_flow_exponential_smooth + (1 - smooth) * optical_flow
        t, _ = estimate_z_transition(optical_flow_exponential_smooth)
        visit_segmentation.append(int(t < threshold))

    # Post processing: Remove single-frame visits and single-frame visit gaps
    visit_segmentation[1:-1][(visit_segmentation[:-2] == 0) & (visit_segmentation[2:] == 0)] = 0
    visit_segmentation[1:-1][(visit_segmentation[:-2] == 1) & (visit_segmentation[2:] == 1)] = 1


    # Post processing. Remove too short visits.
    visit_segmentation_diff = np.diff(np.array(visit_segmentation, dtype=np.float))
    visit_segments = np.reshape(np.where(visit_segmentation_diff != 0)[0] + 1, (-1, 2))
    for visit_segment in visit_segments:
        if visit_segment[1] - visit_segment[0] <= min_visit_frames:
            visit_segmentation[visit_segment[0]:visit_segment[1]] = 0

    # Post processing. Remove too short transitions.
    visit_segmentation_diff_predicted = np.diff(np.array(visit_segmentation_predicted, dtype=np.float))
    transition_segments_predicted = np.where(visit_segmentation_diff_predicted != 0)[0] + 1
    transition_segments_predicted = np.concatenate([[0], transition_segments_predicted, [frame_count]])
    transition_segments_predicted = np.reshape(transition_segments_predicted, (-1, 2))
    for transition_segment in transition_segments_predicted:
        if transition_segment[1] - transition_segment[0] <= min_transition_frames:
            visit_segmentation_predicted[transition_segment[0]:transition_segment[1]] = 1

    for transition_segment in transition_segments_true:
        visit_segmentation_ground_truth[transition_segment[0]:transition_segment[1]] = 1


##############################
#           Main             #
##############################

if __name__ == "__main__":
    # get args from user
    args = parseargs().__dict__

    # run main script
    segment_visit(**args)
