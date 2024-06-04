import argparse
import os.path
import textwrap
import numpy as np
import pickle
import pandas as pd
from tqdm import tqdm
from egomap.map.location import Location
from egomap.map.view import BOVWView
from egomap.map.map import EgoMap
import os

def parseargs():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                     description=textwrap.dedent('''Visualizes how a location is learned. 
                                                                    Outputs a video file:
                                                                        -loc_viz_wi.avi'''))

    parser.add_argument('--feature-file', '-f', type=str,
                        help="Features file in. (obtain by running extract_bow_features.py)")
    parser.add_argument('--movement-file', '-m', type=str,
                        help="Video segmentation file")
    parser.add_argument('--likelihood-threshold', type=float,
                        default=-6900.0,
                        help="threshold")
    parser.add_argument('--minimum-observation-count', '-m', type=float, default=3,
                        help="Minimum number of observation required to be "
                             "associated with the view not to delete the view")
    args = parser.parse_args()
    return args


def group_consecutive_indices(visits):
    """
    Group together indices of consecutive ones in a binary numpy array.

    Parameters:
    visits (np.ndarray): A 1D numpy array of binary values (0s and 1s).

    Returns:
    List[List[int]]: A list of lists where each sublist contains the indices
                     of consecutive ones in the input array.

    Example:
    #>>> visits = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1])
    #>>> group_consecutive_indices(visits)
    [[0, 1, 2, 3, 4], [9, 10, 11, 12], [16, 17, 18], [20, 21, 22]]
    """
    diffs = np.diff(visits, prepend=0, append=0)
    start_indices = np.where(diffs == 1)[0]
    end_indices = np.where(diffs == -1)[0]
    groups = [list(range(start, end)) for start, end in zip(start_indices, end_indices)]
    return groups


def update_map(map, features):
    predictions = []
    for feature in features:
        log_posterior = map.infer(feature)
        location = np.argmax(log_posterior)
        predictions.append(location)

    if location == 0:
        predictions = [len(log_posterior) if r == 0 else r for r in predictions]

    return predictions


def brush_together(a, b):
    min_len = min(len(a), len(b))
    c = [item for pair in zip(a[:min_len], b) for item in pair]
    c.extend(a[min_len:])
    return c


def map(feature_file, movement_file, likelihood_threshold, minimum_observation_count):
    with open(feature_file, 'rb') as handle:
        features = pickle.load(handle)

    with open(movement_file, 'rb') as handle:
        movement = pickle.load(handle)

    features = np.array(features)
    movement = np.array(movement)

    visit_segments_indexes = group_consecutive_indices(movement.astype(int))
    transition_segments_indexes = group_consecutive_indices(np.abs(movement.astype(int) - 1))

    prediction = []

    if movement[0] == False:
        segments_indexes = brush_together(visit_segments_indexes, transition_segments_indexes)
        segment_labels = [i % 2 for i in range(len(segments_indexes))]
    else:
        segments_indexes = brush_together(transition_segments_indexes, visit_segments_indexes)
        segment_labels = [np.abs(i % 2 - 1) for i in range(len(segments_indexes))]


    map = EgoMap(detector_true_negative_rate=0.68,
                 detector_true_positive_rate=0.68,
                 minimum_observation_count=minimum_observation_count,
                 new_location_likelihood_threshold=likelihood_threshold)

    for segment_indexes, segments_label in tqdm(zip(segments_indexes, segment_labels), unit="visit/transition"):
        if segments_label == 1:
            map.start_visit()
        else:
            map.start_transition()
        frame_wise_prediction = update_map(map, features[segment_indexes])
        prediction += list(frame_wise_prediction)

    df_result = pd.DataFrame(data={"prediction": prediction, "is_visit": movement})
    result_dir = os.path.dirname(feature_file)
    df_result.to_csv(os.path.join(result_dir, "result.csv"))
    with open(os.path.join(result_dir, "map.pkl"), 'wb') as file:
        pickle.dump(map, file)



if __name__ == "__main__":
    # get args from user
    args = parseargs().__dict__

    # run main script
    map(**args)
