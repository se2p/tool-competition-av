# from illumination_map import IlluminationMap
# from illumination_axis import IlluminationAxisDefinition
import illumination_axis
import illumination_map
import samples as sam
from self_driving.simulation_data import SimulationDataRecord
# from samples import BeamNGSample

import dh_utils
import logging as log
import matplotlib.pyplot as plt
import numpy as np

import json
import glob
import time
import os


def create_feature_maps(ctx, visualize, drop_outliers, tag, run_folder):
    '''
    Creates 2D maps for all the features on axis.

    Parameters:
        - visualise: True if is needed to make a visual map
        - drop_outliers: drops all samples which feature values are out of min and max if the value is True
        - run_folder: the folder to save results
    '''
    # creates map axes
    map_axis = create_illumination_map_axis()

    # creates samples to put on the map
    samples = create_samples()

    # creates the ilumination map
    the_map = illumination_map.IlluminationMap(map_axis, set(samples), drop_outliers=drop_outliers)

    # creates tags
    tags = []
    for t in tag:
        # TODO Leave Tags Case Sensitive
        t = str(t)
        if t not in tags:
            if ctx.obj['show-progress']:
                print("Tagging using %s" % t)
            tags.append(t)

    # # creates at_minutes
    at_minutes = []
    # for e in at:
    #     if e not in at_minutes:
    #         if ctx.obj['show-progress']:
    #             print("Generating Maps for samples at %s minutes" % e)
    #         at_minutes.append(e)

    # Note the None identify the "use all the samples" setting. By default is always the last one
    at_minutes.append(None)

    report = the_map.compute_statistics(tags=[], sample_selector=None)

    # TODO what is at_minutes?
    for e in at_minutes:

        select_samples = None
        _tags = tags[:]
        if e is not None:
            if ctx.obj['show-progress']:
                print("Selecting samples within ", e, "minutes")
            # Create the sample selector
            select_samples = select_samples_by_elapsed_time(e)

            # Add the minutes ot the tag, create a copy so we do not break the given set of tags
            _tags.append("".join([str(e).zfill(2), "min"]))
        else:
            if ctx.obj['show-progress']:
                print("Selecting all the samples")

        report = the_map.compute_statistics(tags =_tags, sample_selector = select_samples)

        # Show this if debug is enabled
        log.debug(json.dumps(report, indent=4))

        # Store the report as json
        _store_report_to_folder(report, _tags, run_folder)

        # Create coverage and probability figures
        figures = the_map.visualize(tags=_tags, sample_selector=select_samples)
        # store figures without showing them
        _store_figures_to_folder(figures, _tags, run_folder)

        figures, probability_maps, misbehaviour_maps, coverage_maps\
            = the_map.visualize_probability(tags=_tags, sample_selector=select_samples)
        # store the outputs
        _store_figures_to_folder(figures, _tags, run_folder)
        # store maps
        ##_store_maps_to_folder(probability_maps, _tags, run_folder)
        ##_store_maps_to_folder(misbehaviour_maps, _tags, run_folder)
        _store_maps_to_folder(coverage_maps, _tags, run_folder)

        # Visualize Everything at the end
    if visualize:
        plt.show()
    else:
        # Close all the figures if open
        for figure in figures:
            plt.close(figure)

def create_illumination_map_axis():
    '''
    creates axis for the map
    '''

    # creates array of features names
    features = []
    features.append('MinRadius')
    features.append('DirectionCoverage')
    features.append('MeanLateralPosition')
    features.append('SegmentCount')
    features.append('SDSteeringAngle')
    features.append('Curvature')

    # creates matrix of features, max, min, int = 5
    array = create_array_of_features_for_axis(features)

    # creates the axis
    map_features = []
    for f in array:
        # if ctx.obj['show-progress']:
        #     print("Using feature %s" % f[0])
        map_features.append(illumination_axis.IlluminationAxisDefinition(f[0], f[1], f[2], f[3]))
    return map_features

def create_array_of_features_for_axis(names_of_features):
    array_of_features = []
    for feature in names_of_features:
        min_value = find_min_value_of_feature(feature)
        max_value = find_max_value_of_feature(feature)
        feature_array = []
        feature_array.append(feature)
        feature_array.append(min_value)
        feature_array.append(max_value)
        feature_array.append(5) # TODO how to put a value that equal the number of cells
        array_of_features.append(feature_array)
    return array_of_features

def find_min_value_of_feature(feature_name):
    # TODO to prove that list of dictionaries is not None
    list_of_dictionaries = create_array_of_computed_features_dictionaries_of_all_tests()
    min_value = float('inf')
    for dictionary in list_of_dictionaries:
        if (dictionary[feature_name] < min_value):
            min_value = dictionary[feature_name]
    return min_value

def find_max_value_of_feature(feature_name):
    list_of_dictionaries = create_array_of_computed_features_dictionaries_of_all_tests()
    max_value = float('-inf')
    for dictionary in list_of_dictionaries:
        if (dictionary[feature_name] > max_value):
            max_value = dictionary[feature_name]
    return max_value

def create_array_of_computed_features_dictionaries_of_all_tests():
    '''
    creates a list of Dictionaries, that contains fitness function and all features
    of all 'results/sample_test_generators.*/test.*.json' files
    '''
    list_of_dictionaries_of_computed_features = []
    path_to_results = "../../tool-competition-av/results/sample_test_generators.*/test.*.json"

    # returns paths to test.0001.json files
    list_of_paths_to_json_files = glob.glob(path_to_results)
    for path in list_of_paths_to_json_files:
        dictionary_of_features = compute_features_for_the_simulation(path)
        # if the simulation data is not valid
        if (dictionary_of_features != None):
            list_of_dictionaries_of_computed_features.append(compute_features_for_the_simulation(path))

    return list_of_dictionaries_of_computed_features

def compute_features_for_the_simulation(folder):
    """
    creates a dictionary with feature values for simulations of the test in the folder 'folder'.
     - key 'MinRadius'
     - key 'DirectionCoverage'
     - key 'MeanLateralPosition'
     - key 'SegmentCount'
     - key 'SDSteeringAngle'
     - key 'Curvature'
      and the fitness value of this test
      - key 'FitnessFunction'
      """

    # creates an empty list of the featuresValues
    feature_values_dict = {}

    # fetches simulation data from json file
    log.info("Parsing an collecting the data for calculating features and fitness functions...")
    test_data = load_test_data_from_json_file(folder)
    if (test_data['is_valid']):
        test_data = load_test_data_from_json_file(folder)
        interpolated_points = load_the_interpolated_points_from_json_file(folder)
        states = load_the_simulation_records_data_from_json(folder)

        # calculates the value of 'MinRadius' feature
        min_Radius_Value = dh_utils.min_radius(interpolated_points, )

        feature_values_dict['MinRadius'] = min_Radius_Value

        # calculates the value of 'DirectionCoverage' feature
        direction_Coverage = dh_utils.direction_coverage(interpolated_points, )

        feature_values_dict['DirectionCoverage'] = direction_Coverage

        # calculates the value of 'MeanLateralPosition' feature
        mean_Lateral_Position = dh_utils.mean_lateral_position(states)

        feature_values_dict['MeanLateralPosition'] = mean_Lateral_Position

        # calculates the value of 'SegmentCount' feature
        segment_count_value = dh_utils.segment_count(interpolated_points)

        feature_values_dict['SegmentCount'] = segment_count_value

        # calculates the value of 'SDSteeringAngle' feature
        sds_steering_angle_value = dh_utils.sd_steering(states)

        feature_values_dict['SDSteeringAngle'] = sds_steering_angle_value

        # calculates the value of 'Curvature' feature
        curvature_value = dh_utils.curvature(interpolated_points, )

        feature_values_dict['Curvature'] = curvature_value

        # calculates fitness function for the test
        # the value is 1 if the test has failed and 0 if the test has passed
        fitness_function_value = dh_utils.fitness_function(test_data)
        feature_values_dict['FitnessFunction'] = fitness_function_value

        # adds test_data to the dictionary
        feature_values_dict['TestData'] = test_data

        # adds interpolated_points to the dictionary
        feature_values_dict['InterpolatedPoints'] = interpolated_points

        # adds states to the dictionary
        feature_values_dict['States'] = states

        return feature_values_dict

def load_test_data_from_json_file(path_to_json):
    '''
    loads all the test data from the json file
    path_to_json is the path + the name of the json file
    '''
    # path_json = "./results/" + folder + "/" + param
    log.info("loading the test's data from the file '" + path_to_json + "'...")
    with open(path_to_json, 'r', encoding='utf-8') as f:
        obj = json.load(f)
        return obj

def load_the_interpolated_points_from_json_file(path_to_json):
    '''
    loads interpolation points (sample_points) from the json file
    path_to_json is the path + the name of the json file
    '''
    # path_json = "./results/" + folder + "/" + param
    log.info("loading the interpolated points from the file '" + path_to_json + "'...")
    with open(path_to_json, 'r', encoding='utf-8') as f:
        obj = json.load(f)
        return obj['interpolated_points']

def load_the_simulation_records_data_from_json(path_to_json):
    '''
    loads simulation_data_record from the json file
    path_to_json is the path + the name of the json file
    '''
    # path_json = "./results/" + folder + "/" + param
    log.info("Loading the simulation data records from the file '" + path_to_json + "'...")
    with open(path_to_json, 'r') as f:
        obj = json.loads(f.read())

    states = [SimulationDataRecord(*r) for r in obj["execution_data"]]
    return states

def create_samples():
    '''
    creates BeamNGSamples to use by creating the IlluminationMap
    '''
    log.info("Creating samples for the feature map...")
    samples = []


    list_of_dictionaries = create_array_of_computed_features_dictionaries_of_all_tests()
    basepath = ""

    for dictionary in list_of_dictionaries:
        samples.append(sam.BeamNGSample(basepath, dictionary['TestData']['id'],
                                    dictionary['TestData']['is_valid'],
                                    dictionary['InterpolatedPoints'],
                                    dictionary['States'],
                                    dictionary['TestData'],
                                    dictionary['MinRadius'],
                                    dictionary['DirectionCoverage'],
                                    dictionary['MeanLateralPosition'],
                                    dictionary['SegmentCount'],
                                    dictionary['SDSteeringAngle'],
                                    dictionary['Curvature'],
                                    dictionary['FitnessFunction']))

    return samples

def select_samples_by_elapsed_time(max_elapsed_minutes, min_elapsed_minutes=0):
    """
    Consider the samples only if their elapsed time is between min and max. Min is defaulted to 0.0.
    Everything is expressed in minutes

    Args:
        max_elapsed_minutes:
        min_elapsed_minutes

    Returns:

    """

    def _f(samples):
        # Convert minutes in seconds. Note we need to account for minutes that are over the hour, but also that times
        # must all have the same year (gmtime and strptime start from 1970 and 1900)
        elapsed_time_min = time.strptime(time.strftime('%H:%M:%S', time.gmtime(min_elapsed_minutes * 60)), "%H:%M:%S")
        elapsed_time_max = time.strptime(time.strftime('%H:%M:%S', time.gmtime(max_elapsed_minutes * 60)), "%H:%M:%S")

        log.debug("CONSIDERING ONLY SAMPLES BETWEEN \n\t%s\n\t%s ", elapsed_time_min, elapsed_time_max)

        return [sample for sample in samples if
                elapsed_time_min <= time.strptime(sample.elapsed, "%H:%M:%S.%f") <= elapsed_time_max]

    return _f

def _store_report_to_folder(report, tags, run_folder):
    """
    Store the content of the report dict as json in the given run_folder as stats.json

    Args:
        report:
        run_folder:

    Returns:

    """
    # Basic format
    file_name_tokens = [str(report["Tool"]), str(report["Run ID"]).zfill(3)]

    # Add tags if any
    # if tags is not None:
    #     file_name_tokens.extend([str(t) for t in tags])

    # This is the actual file name
    file_name_tokens.append("stats")

    # Add File extension
    report_file_name = "-".join(file_name_tokens) + "." + "json"

    report_file = os.path.join(run_folder, report_file_name)

    log.debug("Storing report %s to file %s ", id, report_file)
    with open(report_file, 'w') as output_file:
        output_file.writelines(json.dumps(report, indent=4))

def _store_figures_to_folder(figures, tags, run_folder):
    # figures, report
    # "Tool": "DLFuzz",
    #     "Run ID": "1",
    #     "Total Samples": 1098,
    file_format = 'pdf'

    for figure in figures:
        file_name_tokens = [figure.store_to]

        # Add tags if any
        if tags is not None:
            file_name_tokens.extend([str(t) for t in tags])

        # Add File extension
        figure_file_name = "-".join(file_name_tokens) + "." + file_format

        figure_file = os.path.join(run_folder, figure_file_name)
        log.debug("Storing figure to file %s ", figure_file)
        figure.savefig(figure_file, format=file_format)

def _store_maps_to_folder(maps, tags, run_folder):
    file_format = 'npy'
    # np.save('test3.npy', a)
    for the_map in maps:
        file_name_tokens = [the_map["store_to"]]

        # Add tags if any
        if tags is not None:
            file_name_tokens.extend([str(t) for t in tags])

        # Add File extension
        map_file_name = "-".join(file_name_tokens) + "." + file_format

        map_file = os.path.join(run_folder, map_file_name)
        log.debug("Storing map %s to file %s ", id, map_file)
        # Finally store it in  platform-independent numpy format
        np.save(map_file, the_map["data"])

# creates a folder in the system
def create_folder(folder_name, path):
    _fold_name = folder_name.replace(':', ' ').replace('.', ' ')
    _path = path
    if not (os.path.exists(_path + "/" + _fold_name)):
        os.chdir(_path)
        os.mkdir(_fold_name)
    else:
        print("the folder with the name '" + folder_name + "' is already exist in the system")

if __name__ == '__main__':
    # when current file run as executable
    # result = load_the_interpolated_points_from_json_file("../tool-competition-av/results/sample_test_generators.one_test_generator_OneTestGenerator_1619774479881/test.0001.json")
    # print(result)
    # result2 = load_the_simulation_records_data_from_json("../tool-competition-av/results/sample_test_generators.one_test_generator_OneTestGenerator_1619774479881/test.0001.json")
    # print(result2)
    # list = create_list_of_paths()
    # test_data = load_test_data_from_json_file(list[0])
    # print(test_data['is_valid'])
    # print(list)
    ##interpolated_points = load_the_interpolated_points_from_json_file(list[1])  # ['interpolated_points']
    # print(interpolated_points)

    # states = load_the_simulation_records_data_from_json(list[1])
    # print(states)
    # print("Steering of the first State: " + str(states[1].steering))
    # print("Oob_distance of the first State: " + str(states[1].oob_distance))
    # #compute_features_for_the_simulation("")

    # min_Radius_Value = min_radius(interpolated_points, )
    # direction_Coverage = direction_coverage(interpolated_points, )
    # mean_Lateral_Position = mean_lateral_position(states)
    # segment_count_value = segment_count(interpolated_points)
    # sds_steering_angle_value = sd_steering(states)
    # curvature_value = curvature(interpolated_points, )
    # fitness_function_value = fitness_function(test_data)
    # print(direction_Coverage)

    # dictionary_of_features = compute_features_for_the_simulation(list[1])
    # print(dictionary_of_features)
    # array_of_tests_dict_with_features = create_array_of_computed_features_dictionaries_of_all_tests()
    # print(array_of_tests_dict_with_features)

    max_MinRadius = find_max_value_of_feature('MinRadius')
    min_MinRadius = find_min_value_of_feature('MinRadius')
    print(max_MinRadius)
    print(min_MinRadius)
    features = []
    features.append('MinRadius')
    features.append('DirectionCoverage')
    features.append('MeanLateralPosition')
    features.append('SegmentCount')
    features.append('SDSteeringAngle')
    features.append('Curvature')

    array = create_array_of_features_for_axis(features)
    # print(array)
    #
    samples = create_samples()
    sample1 = samples[1].to_dict()
    # print(samples[1].to_dict())
    #
    #
    class Ctx:
        obj = {"show-progress": True}


    ctx = Ctx()

    tag = []
    tag.append("test")
    # tag.append(str(datetime.datetime.now().time()))
    # print(tag)


    create_folder("map", "../../tool-competition-av/results/")
    # if not os.path.exists(os.path.dirname("../tool-competition-av/results/map/")):
    #     dir_name = os.path.dirname(filename)
    #     os.makedirs(dir_name)


    # Create the feature map ctx, visualize, drop_outliers, tag, at, run_folder
    create_feature_maps(ctx, True, True, tag, "../../tool-competition-av/results/map")
    print(samples[1].to_dict())

    # generate()