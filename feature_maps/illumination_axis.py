import logging
import numpy as np
import samples as sam

class IlluminationAxisDefinition:
    """
        This class was copied from @DeepHyperion tool

        Data structure that model one axis of the map. In general a map can have multiple axes, even if we visualize
        only a subset of them. On axis usually correspond to a feature to explore.

        For the moment we assume that each axis is equally split in `num_cells`
    """

    def __init__(self, feature_name, min_value, max_value, num_cells):
        self.logger = logging.getLogger('illumination_map.IlluminationAxisDefinition')
        self.logger.debug('Creating an instance of IlluminationAxisDefinition for feature %s', feature_name)

        self.feature_name = feature_name
        self.min_value = min_value
        self.max_value = max_value
        self.num_cells = num_cells
        # Definition of the inner map, values might fall outside it if less than min
        self.original_bins = np.linspace(min_value, max_value, num_cells)
        # Definition of the outer map
        # Include the default boundary conditions. Note that we do not add np.PINF, but the max value.
        # Check: https://stackoverflow.com/questions/4355132/numpy-digitize-returns-values-out-of-range
        self.bins = np.concatenate(([np.NINF], self.original_bins, [max_value + 0.001]))

    def get_bins_labels(self, is_outer_map=False):
        """
        Note that here we return explicitly the last bin
        Returns: All the bins plus the default

        """
        if is_outer_map:
            return np.concatenate(([np.NINF], self.original_bins, [np.PINF]))
        else:
            return self.original_bins

    def get_coordinate_for(self, sample: sam.Sample, is_outer_map=False):
        """
        Return the coordinate of this sample according to the definition of this axis. It triggers exception if the
            sample does not declare a field with the name of this axis, i.e., the sample lacks this feature

        Args:
            sample:

        Returns:
            an integer representing the coordinate of the sample in this dimension

        Raises:
            an exception is raised if the sample does not contain the feature
        """

        # TODO Check whether the sample has the feature
        value = sample.get_value(self.feature_name)

        if value < self.min_value:
            self.logger.warning("Sample %s has value %s below the min value %s for feature %s",
                                sample.id, value, self.min_value, self.feature_name)
        elif value > self.max_value:
            self.logger.warning("Sample %s has value %s above the max value %s for feature %s",
                                sample.id, value, self.max_value, self.feature_name)

        if is_outer_map:
            return np.digitize(value, self.bins, right=False)
        else:
            return np.digitize(value, self.original_bins, right=False)

    def is_outlier(self, sample):
        value = sample.get_value(self.feature_name)
        is_outlier = value < self.min_value or value > self.max_value #
        return is_outlier #
        # return value < self.min_value or value > self.max_value

    def to_dict(self):
        the_dict = {
            "name": self.feature_name,
            "min-value": self.min_value,
            "max-value": self.max_value,
            "num-cells": self.num_cells
        }
        return the_dict