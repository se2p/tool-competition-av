
import logging
import statistics
import itertools

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from itertools import combinations

# https://stackoverflow.com/questions/34630772/removing-duplicate-elements-by-their-attributes-in-python
def deduplicate(items):
    """
    Make sure that items with the same ID are removed
    TODO This might be also possible to refine
    Args:
        items:

    Returns:

    """
    seen = set()
    for item in items:
        if not item.id in seen:
            seen.add(item.id)
            yield item
        else:
            logging.debug("Removing duplicated sample %s", item.id)

def manhattan(coords_ind1, coords_ind2):
    return abs(coords_ind1[0] - coords_ind2[0]) + abs(coords_ind1[1] - coords_ind2[1])

def drop_outliers_for(feature, samples):
    """
    Return a list of samples that have min/max. There might be a more pythonic way of doing it using a filter
    """
    return [s for s in samples if not feature.is_outlier(s)]

class IlluminationMap:
    """
            Data structure that represent a map. The map is defined in terms of its axes
        """

    def __init__(self, axes: list, samples: set, drop_outliers=False):
        """
        Note that axes are positional, the first [0] is x, the second[1] is y, the third [2] is z, etc.
        Args:
            axes:
        """
        self.logger = logging.getLogger('illumination_map.IlluminationMapDefinition')

        assert len(axes) > 1, "Cannot build a map with only one feature"

        # Definition of the axes
        self.axes = axes

        # Hide samples that fall ouside the maps as defined by the axes
        self.drop_outliers = drop_outliers

        # Remove duplicated samples: samples with same ID but different attributes (e.g. basepath)
        all_samples = list(deduplicate(samples))
        # Split samples in to valid, invalid, outliers
        self.samples = [sample for sample in all_samples if sample.is_valid]
        # We keep them around no matter what, so we can count them, see them, etc.
        self.invalid_samples = [sample for sample in all_samples if not sample.is_valid]

    def _avg_sparseness_from_map(self, map_data):

        # If there are no samples, we cannot compute it
        if np.count_nonzero(map_data) == 0:
            return np.NaN

        # Iterate over all the non empty cells and compute the distance to all the others non empty cells
        avg_sparseness = 0

        # https://numpy.org/doc/stable/reference/arrays.nditer.html
        # This should iterate over all the elements
        it = np.nditer(map_data, flags=['multi_index'])
        samples = []
        for a in it:
            # print("Considering index ", it.multi_index)
            samples.append(it.multi_index)

        # TODO This can be python-ized
        # Total number of samples

        # k = # observations to compute the mean
        k = 0
        for (sample1, sample2) in combinations(samples, 2):

            if map_data[sample1] == 0 or map_data[sample2] == 0:
                continue

            # Compute distance
            distance = manhattan(sample1, sample2)

            # Increment number of observations
            k += 1

            # Update the avg distance
            # See https://math.stackexchange.com/questions/106700/incremental-averageing

            # print("Considering:", sample1, sample2)
            # print("K", k)
            # print("AVG ", avg_sparseness, end=" ")
            avg_sparseness = avg_sparseness + (distance - avg_sparseness) / k
            # print("AVG ", avg_sparseness)

        return avg_sparseness

    def _avg_max_distance_between_filled_cells_from_map(self, map_data):
        """
        Alternative fomulation for sparseness: for each cell consider only the distance to the farthest cell

        Args:
            map_data:

        Returns:

        """

        # If there are no samples, we cannot compute it
        if np.count_nonzero(map_data) == 0:
            return np.NaN

        # Iterate over all the non empty cells and compute the distance to all the others non empty cells
        avg_sparseness = 0

        # https://numpy.org/doc/stable/reference/arrays.nditer.html
        # This should iterate over all the elements
        it = np.nditer(map_data, flags=['multi_index'])
        samples = []
        for a in it:
            # print("Considering index ", it.multi_index)
            samples.append(it.multi_index)

        # TODO This can be python-ized
        # Total number of samples

        last_sample = None

        max_distances_starting_from_sample = {}

        for (sample1, sample2) in combinations(samples, 2):
            # Combinations do not have repetition, so everytime that sample1 changes, we need to "recompute the
            # max distance". We use sample1 as index to store the max distances starting from a sample

            # Do not consider empty cells
            if map_data[sample1] == 0 or map_data[sample2] == 0:
                continue

            # Compute distance between cells
            distance = manhattan(sample1, sample2)

            if sample1 in max_distances_starting_from_sample.keys():
                max_distances_starting_from_sample[sample1] = max(max_distances_starting_from_sample[sample1], distance)
            else:
                max_distances_starting_from_sample[sample1] = distance

        # Compute the average
        if len(max_distances_starting_from_sample) < 1:
            return 0.0
        else:
            return np.mean([list(max_distances_starting_from_sample.values())])

    # @deprecated("Use np.size instead")
    # def _total_cells(self, features):
    #     return int(np.prod([a.num_cells for a in features]))

    def _total_misbehaviors(self, samples: set) -> int:
        return len([s for s in samples if s.is_misbehavior()])

    def _total_samples(self, samples: set) -> int:
        return len(samples)

    def _mapped_misbehaviors_from_map(self, misbehavior_data):
        """
        Args:
            misbehaviour_data a matrix that contains the count of misbheaviors per cell
        Returns:
            the count of cells for which at least one sample is a misbehavior.

        """
        return np.count_nonzero(misbehavior_data > 0)

    def _filled_cells_from_map(self, coverage_data):
        """
        Returns:
            the count of cells covered by at least one sample.
        """
        # https://note.nkmk.me/en/python-numpy-count/
        return np.count_nonzero(coverage_data > 0)

    def _relative_density_of_mapped_misbehaviors_from_map(self, coverage_data, misbehavior_data):
        """
        Returns:
            the density of misbehaviors in a map computed w.r.t. the amount of filled cells in the map
        """
        filled_cells = self._filled_cells_from_map(coverage_data)
        return self._mapped_misbehaviors_from_map(misbehavior_data) / filled_cells if filled_cells > 0 else np.NaN

    def _density_of_mapped_misbehavior_from_maps(self, misbehavior_data):
        """
        Returns:
            the density of misbehaviors in a map computed
        """
        return self._mapped_misbehaviors_from_map(misbehavior_data) / misbehavior_data.size

    def _density_of_covered_cells_from_map(self, coverage_data):
        """
        Returns:
            the density of covered cell in the map. This is basically the coverage of the map
        """
        return self._filled_cells_from_map(coverage_data) / coverage_data.size

    def _count_collisions(self, map_data):
        """
        Returns:
            the overall count of cells with a collision, i.e., where there's more than one sample
        """
        return np.count_nonzero(map_data > 1)

    def _collisions_ratio(self, map_data):
        """
        Returns:
            the amount of collisions in the map (when two or more samples hit the same cell).
        """
        filled_cells = self._filled_cells_from_map(map_data)
        total_samples = np.sum(map_data)
        return (total_samples - filled_cells) / filled_cells if filled_cells > 0 else np.NaN

    def _get_tool(self, samples):
        # TODO Assume that all the samples belong to the same tool
        # TODO trigger exception otherwise
        if len(samples) > 0:
            for sample in samples:
                return sample.tool
        return None

    def _get_run_id(self, samples):
        # TODO Assume that all the samples belong to the same run_id
        # TODO trigger exception otherwise
        if len(samples) > 0:
            for sample in samples:
                return sample.run
        return None

    def compute_statistics(self, tags=[], feature_selector=None, sample_selector=None):
        """
        Compute the statistics for this map optionally selecting the samples according to the give selector and a
        subset of the features. Otherwise one report for each feature combination will be generated.

        The selector for example, can tell whether or not consider the samples generated in a given interval of
        time.

        Args:
            tags: list of tags, includig the "within X mins"
            feature_selector: a function to select the features
            sample_selector: a function to select the samples. For example, only the samples collected in the first
            15 minutes of run

        Returns:

        """
        filtered_samples = self.samples
        self.logger.debug("Valid samples: %s", len(filtered_samples))

        filtered_invalid_samples = self.invalid_samples
        self.logger.debug("Invalid samples: %s", len(filtered_invalid_samples))

        if sample_selector is not None:
            filtered_samples = sample_selector(self.samples)
            self.logger.debug("Filtered valid samples: %s", len(filtered_samples))

            filtered_invalid_samples = sample_selector(self.invalid_samples)
            self.logger.debug("Filtered invalid samples: %s", len(filtered_invalid_samples))

        filtered_features = self.axes
        if feature_selector is not None:
            filtered_features = feature_selector(self.axes)

        assert len(filtered_features) > 1, "Cannot compute statistics with less than two features"

        report = {}
        # Meta data
        report["Tool"] = self._get_tool(filtered_samples)
        report["Run ID"] = self._get_run_id(filtered_samples)

        report["Tags"] = tags

        # Totals
        report["Total Samples"] = self._total_samples(filtered_samples) + self._total_samples(filtered_invalid_samples)
        report["Valid Samples"] = self._total_samples(filtered_samples)
        report["Invalid Samples"] = self._total_samples(filtered_invalid_samples)
        report["Total Misbehaviors"] = self._total_misbehaviors(filtered_samples)
        report["MisbehaviorPerSample"] = report["Total Misbehaviors"] / report["Total Samples"]

        # Per Feature Statistics
        report["Features"] = {}
        for feature in filtered_features:
            report["Features"][feature.feature_name] = {}
            report["Features"][feature.feature_name]["meta"] = feature.to_dict()
            report["Features"][feature.feature_name]["stats"] = {}

            feature_raw_data = [sample.get_value(feature.feature_name) for sample in filtered_samples]

            report["Features"][feature.feature_name]["stats"]["mean"] = np.NaN
            report["Features"][feature.feature_name]["stats"]["stdev"] = np.NaN
            report["Features"][feature.feature_name]["stats"]["median"] = np.NaN

            if len(feature_raw_data) > 2:
                report["Features"][feature.feature_name]["stats"]["mean"] = statistics.mean(feature_raw_data)
                report["Features"][feature.feature_name]["stats"]["stdev"] = statistics.stdev(feature_raw_data)
                report["Features"][feature.feature_name]["stats"]["median"] = statistics.median(feature_raw_data)
            elif len(feature_raw_data) == 1:
                report["Features"][feature.feature_name]["stats"]["mean"] = feature_raw_data[0]
                report["Features"][feature.feature_name]["stats"]["stdev"] = 0.0
                report["Features"][feature.feature_name]["stats"]["median"] = feature_raw_data[0]

        # Create one report for each pair of selected features
        report["Reports"] = []

        # We filter outliers

        total_samples_in_the_map = filtered_samples

        for feature1, feature2 in itertools.combinations(filtered_features, 2):

            selected_features = [feature1, feature2]

            # make sure we reset this across maps
            filtered_samples = total_samples_in_the_map

            # Remove samples that are outliers for this map
            if self.drop_outliers:
                filtered_samples = drop_outliers_for(feature1, filtered_samples)
                filtered_samples = drop_outliers_for(feature2, filtered_samples)

            # Build the map data: For the moment forget about outer maps, those are mostly for visualization!
            coverage_data, misbehavior_data, _, _ = self._compute_maps_data(feature1, feature2, filtered_samples)

            # Compute statistics over the map data
            map_report = {
                # Meta data
                'Features': [feature.feature_name for feature in selected_features],
                # Counts
                'Sample Count': len(total_samples_in_the_map),
                'Outlier Count': len(total_samples_in_the_map) - len(filtered_samples),
                'Total Cells': coverage_data.size,
                'Filled Cells': self._filled_cells_from_map(coverage_data),
                'Mapped Misbehaviors': self._mapped_misbehaviors_from_map(misbehavior_data),
                # Density
                'Misbehavior Relative Density': self._relative_density_of_mapped_misbehaviors_from_map(coverage_data,
                                                                                                       misbehavior_data),
                'Misbehavior Density': self._density_of_mapped_misbehavior_from_maps(misbehavior_data),
                'Filled Cells Density': self._density_of_covered_cells_from_map(coverage_data),
                'Collisions': self._count_collisions(coverage_data),
                'Misbehavior Collisions': self._count_collisions(misbehavior_data),
                'Collision Ratio': self._collisions_ratio(coverage_data),
                'Misbehavior Collision Ratio': self._collisions_ratio(misbehavior_data),
                # Sparseness
                'Coverage Sparseness': self._avg_max_distance_between_filled_cells_from_map(coverage_data),
                'Misbehavior Sparseness': self._avg_max_distance_between_filled_cells_from_map(misbehavior_data),
                # The follwing two only for retro-compability
                'Avg Sample Distance': self._avg_sparseness_from_map(coverage_data),
                'Avg Misbehavior Distance': self._avg_sparseness_from_map(misbehavior_data)
            }

            report["Reports"].append(map_report)

        return report

    def _compute_maps_data(self, feature1, feature2, samples):
        """
        Create the raw data for the map by placing the samples on the map and counting for each cell how many samples
        are there and how many misbehaviors
        Args:
            feature1:
            feature2:
            samples:

        Returns:
            coverage_map, misbehavior_map
            coverage_outer_map, misbehavior_outer_map
        """
        # TODO Refactor:

        # Reshape the data as ndimensional array. But account for the lower and upper bins.
        coverage_data = np.zeros(shape=(feature1.num_cells, feature2.num_cells), dtype=int)
        misbehaviour_data = np.zeros(shape=(feature1.num_cells, feature2.num_cells), dtype=int)

        coverage_outer_data = np.zeros(shape=(feature1.num_cells + 2, feature2.num_cells + 2), dtype=int)
        misbehaviour_outer_data = np.zeros(shape=(feature1.num_cells + 2, feature2.num_cells + 2), dtype=int)

        for sample in samples:

            # Coordinates reason in terms of bins 1, 2, 3, while data is 0-indexed
            x_coord = feature1.get_coordinate_for(sample, is_outer_map=False) - 1
            y_coord = feature2.get_coordinate_for(sample, is_outer_map=False) - 1

            # Increment the coverage cell
            coverage_data[x_coord, y_coord] += 1

            # Increment the misbehaviour cell
            if sample.is_misbehavior():
                misbehaviour_data[x_coord, y_coord] += 1

            # Outer Maps
            x_coord = feature1.get_coordinate_for(sample, is_outer_map=True) - 1
            y_coord = feature2.get_coordinate_for(sample, is_outer_map=True) - 1

            # Increment the coverage cell
            coverage_outer_data[x_coord, y_coord] += 1

            # Increment the misbehaviour cell
            if sample.is_misbehavior():
                misbehaviour_outer_data[x_coord, y_coord] += 1

        return coverage_data, misbehaviour_data, coverage_outer_data, misbehaviour_outer_data

    def visualize_probability(self, tags=None, feature_selector=None, sample_selector=None):
        """
            Visualize the probability of finding a misbehavior in a give cell, computed as the total of misbehavior over
            the total samples in each cell. This is defined only for cells that have samples in them. Also store
            the probability data so they can be post-processed (e.g., average across run/configuration)
        """
        # Prepare the data by selecting samples and features

        filtered_samples = self.samples
        self.logger.debug("All samples: %s", len(filtered_samples))
        if sample_selector is not None:
            filtered_samples = sample_selector(self.samples)
            self.logger.debug("Filtered samples: %s", len(filtered_samples))

        filtered_features = self.axes
        if feature_selector is not None:
            filtered_features = feature_selector(self.axes)

        figures = []
        # Might be redundant if we store also misbehaviour_maps and coverage_maps
        probability_maps = []
        # To compute confidence intervals and possibly other metrics on the map
        misbehaviour_maps = []
        coverage_maps = []

        total_samples_in_the_map = filtered_samples

        # Create one visualization for each pair of self.axes selected in order
        for feature1, feature2 in itertools.combinations(filtered_features, 2):

            # Make sure we reset this for each feature combination
            filtered_samples = total_samples_in_the_map
            # Remove samples that are outliers for this map
            if self.drop_outliers:
                filtered_samples = drop_outliers_for(feature1, filtered_samples)
                filtered_samples = drop_outliers_for(feature2, filtered_samples)

            coverage_data, misbehaviour_data, _, _ = self._compute_maps_data(feature1, feature2, filtered_samples)

            # figure
            fig, ax = plt.subplots(figsize=(8, 8))

            cmap = sns.cubehelix_palette(dark=0.1, light=0.9, as_cmap=True)
            # Cells have a value between 0.0 and 1.0 since they represent probabilities

            # Set the color for the under the limit to be white (0.0) so empty cells are not visualized
            # cmap.set_under('0.0')
            # Plot NaN in white
            cmap.set_bad(color='white')

            # Coverage data might be zero, so this produces Nan. We convert that to 0.0
            # probability_data = np.nan_to_num(misbehaviour_data / coverage_data)
            raw_probability_data = misbehaviour_data / coverage_data

            # For some weird reason the data in the heatmap are shown with the first dimension on the y and the
            # second on the x. So we transpose
            probability_data = np.transpose(raw_probability_data)

            sns.heatmap(probability_data, vmin=0.0, vmax=1.0, square=True, cmap=cmap)

            xtickslabel = [round(the_bin, 1) for the_bin in feature1.get_bins_labels(is_outer_map=False)]
            ytickslabel = [round(the_bin, 1) for the_bin in feature2.get_bins_labels(is_outer_map=False)]
            #
            ax.set_xticklabels(xtickslabel)
            plt.xticks(rotation=45)
            ax.set_yticklabels(ytickslabel)
            plt.yticks(rotation=0)

            tool_name = str(self._get_tool(filtered_samples))
            run_id = str(self._get_run_id(filtered_samples)).zfill(3)

            title_tokens = ["Mishbehavior Probability", "\n"]
            title_tokens.extend(["Tool:", tool_name, "--", "Run ID:", run_id])

            if tags is not None and len(tags) > 0:
                title_tokens.extend(["\n", "Tags:"])
                title_tokens.extend([str(t) for t in tags])

            the_title = " ".join(title_tokens)

            fig.suptitle(the_title, fontsize=16)

            # Plot small values of y below.
            # We need this to have the y axis start from zero at the bottom
            ax.invert_yaxis()

            # axis labels
            plt.xlabel(feature1.feature_name)
            plt.ylabel(feature2.feature_name)

            # Include data to store the file with same prefix

            # Add the store_to attribute to the figure and maps object
            setattr(fig, "store_to",
                    "-".join(["probability", tool_name, run_id, feature1.feature_name, feature2.feature_name]))
            figures.append(fig)

            probability_maps.append({
                "data": raw_probability_data,
                "store_to": "-".join(["probability", tool_name, run_id, feature1.feature_name, feature2.feature_name])
            })

            misbehaviour_maps.append({
                "data": misbehaviour_data,
                "store_to": "-".join(["misbehaviour", tool_name, run_id, feature1.feature_name, feature2.feature_name])
            })

            coverage_maps.append({
                "data": coverage_data,
                "store_to": "-".join(["coverage", tool_name, run_id, feature1.feature_name, feature2.feature_name])
            })

        return figures, probability_maps, misbehaviour_maps, coverage_maps

    def visualize(self, tags=None, feature_selector=None, sample_selector=None):
        """
            Visualize the samples and the features on a map. The map cells contains the number of samples for each
            cell, so empty cells (0) are white, cells with few elements have a light color, while cells with more
            elements have darker color. This gives an intuition on the distribution of the misbheaviors and the
            collisions

        Args:
            tags: List of tags to the title so we can easily identify the maps
            feature_selector:
            sample_selector:

        Returns:


        """

        filtered_samples = self.samples
        self.logger.debug("All samples: %s", len(filtered_samples))
        if sample_selector is not None:
            filtered_samples = sample_selector(self.samples)
            self.logger.debug("Filtered samples: %s", len(filtered_samples))

        filtered_features = self.axes
        if feature_selector is not None:
            filtered_features = feature_selector(self.axes)

        figures = []

        total_samples_in_the_map = filtered_samples

        # Create one visualization for each pair of self.axes selected in order
        for feature1, feature2 in itertools.combinations(filtered_features, 2):

            # Make sure we reset this for each feature combination
            filtered_samples = total_samples_in_the_map
            # Remove samples that are outliers for this map
            if self.drop_outliers:
                filtered_samples = drop_outliers_for(feature1, filtered_samples)
                filtered_samples = drop_outliers_for(feature2, filtered_samples)

            # TODO For the moment, since filtered_Samples might be different we need to rebuild this every time
            coverage_data, misbehaviour_data, _, _ = self._compute_maps_data(feature1, feature2, filtered_samples)

            # figure
            fig, ax = plt.subplots(figsize=(8, 8))

            cmap = sns.cubehelix_palette(dark=0.5, light=0.9, as_cmap=True)
            # Set the color for the under the limit to be white (so they are not visualized)
            cmap.set_under('1.0')

            # For some weird reason the data in the heatmap are shown with the first dimension on the y and the
            # second on the x. So we transpose
            coverage_data = np.transpose(coverage_data)

            sns.heatmap(coverage_data, vmin=1, vmax=20, square=True, cmap=cmap)

            # Plot misbehaviors - Iterate over all the elements of the array to get their coordinates:
            it = np.nditer(misbehaviour_data, flags=['multi_index'])
            for v in it:
                # Plot only misbehaviors
                if v > 0:
                    alpha = 0.1 * v if v <= 10 else 1.0
                    (x, y) = it.multi_index
                    # Plot as scattered plot. the +0.5 ensures that the marker in centered in the cell
                    plt.scatter(x + 0.5, y + 0.5, color="black", alpha=alpha, s=50)

            xtickslabel = [round(the_bin, 1) for the_bin in feature1.get_bins_labels(is_outer_map=False)]
            ytickslabel = [round(the_bin, 1) for the_bin in feature2.get_bins_labels(is_outer_map=False)]
            #
            ax.set_xticklabels(xtickslabel)
            plt.xticks(rotation=45)
            ax.set_yticklabels(ytickslabel)
            plt.yticks(rotation=0)

            tool_name = str(self._get_tool(filtered_samples))
            run_id = str(self._get_run_id(filtered_samples)).zfill(3)

            title_tokens = ["Collisions and Mishbehaviors", "\n"]
            title_tokens.extend(["Tool:", tool_name, "--", "Run ID:", run_id])

            if tags is not None and len(tags) > 0:
                title_tokens.extend(["\n", "Tags:"])
                title_tokens.extend([str(t) for t in tags])

            the_title = " ".join(title_tokens)

            fig.suptitle(the_title, fontsize=16)

            # Plot small values of y below.
            # We need this to have the y axis start from zero at the bottom
            ax.invert_yaxis()

            # axis labels
            plt.xlabel(feature1.feature_name)
            plt.ylabel(feature2.feature_name)

            # Add the store_to attribute to the figure object
            store_to = "-".join(
                ["collision", "misbehavior", tool_name, run_id, feature1.feature_name, feature2.feature_name])
            setattr(fig, "store_to", store_to)

            figures.append(fig)

        return figures