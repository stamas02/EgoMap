import os
import pickle
import argparse
from egomap.features.bovw import BOVW
import textwrap


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                     description=textwrap.dedent('''\
                                         Learns a Bag-of-Visual-Words (BOVW) classifier from the given dataset.
                                         The following file will be created in the destination folder:
                                             bovw.pickle'''))
    parser.add_argument('--source', '-s',
                        type=str,
                        help=textwrap.dedent('''\
                            Path to the folder containing the training dataset.
                            The folder structure does not matter as long as it only contains image files.'''))

    parser.add_argument('--destination', '-d',
                        type=str,
                        help="Path to the folder where the result will be saved.")

    parser.add_argument('--method', '-m',
                        type=str,
                        choices=['SURF', 'SURF_e', 'SIFT'],
                        default="SURF_e",
                        help=textwrap.dedent('''\
                            Method used for feature extraction.
                            Supported methods are:
                                - SURF
                                - SURF_e [default]
                                - SIFT'''))

    parser.add_argument('--feature-cnt',
                        type=int,
                        default=10,
                        help=textwrap.dedent('''\
                            Total count of features extracted per image.'''))

    parser.add_argument('--dimensionality', "-dim",
                        type=int,
                        default=-1,
                        help=textwrap.dedent('''\
                            Dimensionality of the output feature.
                            If not specified, then sqrt(num_features) will be used.'''))

    args = parser.parse_args()
    return args

def learn_vocabulary(source, destination, method, feature_cnt, dimensionality):
    """
        Learns a vocabulary for Bag-of-Visual-Words (BOVW) and saves the trained object.

        Parameters
        ----------
        source : str
            Path to the folder containing the training images.

        destination : str
            Destination (including filename) to save the trained object using pickle.

        method : str
            Method used for feature extraction. Supported methods are ["SIFT", "SURF", "SURF_e"].
            Note that "SURF_e" stands for the extended version with 128-dimensional features.

        feature_cnt : int
            Number of features extracted per image

        dimensionality : int
            The dimensionality of descriptor for each image. If None, then sqrt(num_of_features) will be used.

        Returns
        -------
        None
        """
    # Load list of training files
    training_files = [os.path.join(root, file) for root, dirs, files in os.walk(source) for file in files]

    print("[Success] Training dataset is loaded with size: {0}".format(len(training_files)))

    # Start training
    my_bovw = BOVW(feature_extraction_method=method, n_clusters=dimensionality, feature_per_image=feature_cnt)
    my_bovw.train(training_files, verbosity=1)

    # Save trained vocabulary
    with open(os.path.join(destination, 'bovw.pickle'), 'wb') as handle:
        pickle.dump(my_bovw, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('[Success] Classifier is saved to {}'.format(os.path.join(destination, "bovw.pickle")))

##############################
#           Main             #
##############################

if __name__ == "__main__":
    # get args from user
    args = parse_args()

    # run main script
    learn_vocabulary(**args.__dict__) 
