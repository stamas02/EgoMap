import numpy as np
import cv2
import math
import os
import pqkmeans
from PIL import Image


class BOVW():
    """
    Bag of visual words implementation.

    This version uses PQk-means to calculate cluster centres.
        source: https://github.com/DwangoMediaVillage/pqkmeans
    """

    def __init__(self, feature_extraction_method="SURF", n_clusters=-1, feature_per_image=-1):
        """
        Initializes a BOVW object.

        Parameters
        ----------
        feature_extraction_method: str,
            Supported methods: "SURF", "SURF_e", "SIFT". Note that SURF_e refers to the extended 128 dim SURF

        n_clusters: int,
            The number of cluster centres used in clustering. This will be the dimensionality of the final descriptor.
            if -1 then the value is sqrt(num_features) where number of features is the number of SWIFT or SURF
            extracted from the set of training images.

        feature_per_image: int Oprional,
            Number of features per image. If -1 then all the features will be used. If an image does not have
            this number of features then all features will be used for that image.
            in other words the result is not guaranteed to be length of img_number*feature_per_image but it is
            guaranteed that img_number*feature_per_image is the upper bound.
        """

        self.model = None
        self.n_clusters = n_clusters
        self.method = feature_extraction_method
        self.dim = 64 if self.method == "SURF" else 128
        self.model = None
        self.feature_per_image = None if feature_per_image == -1 else feature_per_image
        self._encoder = pqkmeans.encoder.PQEncoder()
        pass

    def _get_extractor(self, method):
        """
        creates the feature extractor

        Parameters
        ----------
        method: str,
            Supported methods: "SURF", "SURF_e", "SIFT". Note that SURF_e refers to the extended 128 dim SURF

        Returns
        -------
            The cv2 object for feature extraction

        """
        if method == "SIFT":
            return cv2.xfeatures2d.SIFT_create()
        elif method == "SURF":
            return cv2.xfeatures2d.SURF_create()
        elif method == "SURF_e":
            tmp = cv2.xfeatures2d.SURF_create()
            tmp.setExtended(True)
            return tmp
        else:
            raise ValueError("Method is not supported {0}".format(method))

    def initialize(self, ):
        if self.n_clusters <= 0:
            raise ValueError("you must define the n_cluster argument to be > 0")

    def train(self, train_image_paths, verbosity=1):
        """
        Trains the BOVW vocabulary on the training images.

        Parameters
        ----------
        train_image_paths: List of str
            List of paths to individual training images.

        verbosity: int, optional
            Controls the verbosity of the training process.
            0 - No update messages and no progress bar will be displayed.
            1 - Progress messages will be displayed.

        Returns
        -------
        None
        """

        # EXTRACT ALL FEATURES FROM TRAINING IMAGES
        if (verbosity == 0):
            print("Extract {0} features from {1} images".format(self.method, len(train_image_paths)))

        train_features = np.ones((0, self.dim))
        feature_extractor = self._get_extractor(method=self.method)
        fails = []  # Store paths of images where feature extraction failed
        total_image_cnt = len(train_image_paths)
        for i, path in enumerate(train_image_paths):
            info_string = "{:.2f}% {}/{} Extracting features for image {}".format((i / total_image_cnt),
                                                                                  total_image_cnt, i, path)
            # Print status info
            if verbosity > 0:
                print(info_string)

            # Check if the file exists
            if not os.path.exists(path):
                raise ValueError('File does not exist {0}'.format(path))

            try:
                # Read the image and convert it to grayscale
                img = np.array(Image.open(path).convert('L'))
                # Extract features using the selected method
                kp, des = feature_extractor.detectAndCompute(img, None)
                # Randomly select a subset of features if there are too many
                if len(des) > self.feature_per_image:
                    tmp = np.random.choice(range(0, len(des)), self.feature_per_image)
                    des = des[tmp]
                # Extend the list of features
                train_features = np.concatenate((train_features, des))
            except Exception as e:
                # If feature extraction fails, record the failure
                fails.append([path, str(e)])

        # Report failures
        if len(fails) > 0:
            print("Could not use the following files:")
            for fail in fails:
                print("File: {0} \n Reason: {1}".format(fail[0], fail[1]))

        # Convert the list of features to a numpy array
        train_features = np.array(train_features)

        # Train KMeans on the extracted features
        if self.n_clusters == -1:
            self.n_clusters = int(math.sqrt(len(train_features)))

        if verbosity > 0:
            print("Train on {0} images using pqkmeans with k={1}".format(len(train_image_paths), self.n_clusters))

        tmp = np.random.choice(range(0, len(train_features)), 1000)
        self._encoder.fit(train_features[tmp])
        coded = self._encoder.transform(train_features)
        self.model = pqkmeans.clustering.PQKMeans(k=self.n_clusters, iteration=100, encoder=self._encoder)
        self.model.fit_predict(coded)
        if (verbosity > 0):
            print("Train Complete!")

    def extract_feature(self, image):
        """
        Extracts features using the trained model from an image.

        Parameters
        ----------
        image: numpy array
            the image used for feature extraction. MUST be gray scale or RGB format

        Returns
        -------
            The extracted feature.

        """
        return np.array(np.clip(self.get_descriptor(image), 0, 1))
