DATA_FOLDER = "Data"

# DONT_INCLUDE_GROUNDWATER_AS_FEATURE = False

import numpy as np, pandas as pd
from sklearn.preprocessing import RobustScaler


# =============================================================================
#### OBJECT TO IMPORT AND PROCESS DATA
# =============================================================================

class Data_preprocessor:

    _instance = None
    BASE_PATH = DATA_FOLDER
    BASE_DATA_PATH = f'{DATA_FOLDER}/GW_phase2_Data/'

    def __new__(cls):
        if not cls._instance:
            cls._instance = super(Data_preprocessor, cls).__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if not hasattr(self, 'initialized'):  # To prevent reinitialization
            self.well_list = pd.read_csv(f'{Data_preprocessor.BASE_PATH}/ListOfWellsGermany.txt',sep=' ')
            self.feature_scaler = None
            self.gwl_scaler = None
            self.global_settings = None
            self.num_features = None
            self.nearest_wells_list = None
            self.merged_well_data = None
            self.normalized_x_y_merged_well_data = None
            self.initialized = True  # Set initialized to True after the first init


    def euclidean_distance(self, point1, point2):
        return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


    def k_nearest_neighbours(self, global_settings):
        x_target_well = self.well_list['X_Coord_UTM_32N'][global_settings['target_well_no']]
        y_target_well = self.well_list['Y_Coord_UTM_32N'][global_settings['target_well_no']]

        target_well_coords =[x_target_well, y_target_well]

        distance = [(global_settings['target_well_no'], 0)]
        for curr_well_no in range(len(self.well_list)):
            curr_well_coords = [self.well_list['X_Coord_UTM_32N'][curr_well_no], self.well_list['Y_Coord_UTM_32N'][curr_well_no]]
            if curr_well_no == global_settings['target_well_no']:
                continue
            else:
                distance.append((curr_well_no, self.euclidean_distance(target_well_coords, curr_well_coords)))

        distance.sort(key=lambda x: x[1])

        return distance[:global_settings['k_nearest_wells']+1]


    def to_supervised(self, data, global_settings):
        # Data here is a tuple containing X and y
        data_X = data[0]
        data_y = data[1]
        X, Y = [], []

        # Step over the entire history one time step at a time
        for i in range(len(data_X)):
            # Find the end of this pattern
            end_idx = i + global_settings['seq_length']
            # Check if we are beyond the dataset
            if end_idx + global_settings['prediction_length'] > len(data_X):
                break
            # Gather input and output parts of the pattern
            seq_x = data_X[i:end_idx, : ]
            seq_y = data_y[end_idx:end_idx + global_settings['prediction_length']]

            X.append(seq_x)
            Y.append(seq_y)

        return np.array(X), np.array(Y)


    def split_data(self, well_number, global_settings):
        ##Obtaining the well id
        id = self.well_list['ID'][well_number]

        # List of all columns (WARNING : may be outdated info)
        # temperature
        # precipitation
        # snow water equivalent
        # evapotranspiration
        # canopy_instant
        # storm_runoff
        # soil_moisture
        # GWL

        ##Obtaining the data from directory
        data = pd.read_csv(Data_preprocessor.BASE_DATA_PATH + id + '_data.csv')
        #### data = data.drop(columns = ['precipitation'])
        if not global_settings['extra_features_enabled']:
            data = data.drop(columns = [    'snow water equivalent',
                                            'evapotranspiration',
                                            'canopy_instant',
                                            'storm_runoff',
                                            'soil_moisture'
                                        ]
                            ) #
        data['Date'] = pd.to_datetime(data['Date'])
        data = data.set_index('Date')

        no_of_samples       = len(data)
        train_size          = 1 - (global_settings['val_size'] + global_settings['test_size'])
        train_plus_val_size = 1 - (global_settings['test_size'])

        # data = data_.copy()

        TrainingData =  data[                                         : round(train_size * no_of_samples) ]

        StopData     =  data[ round(train_size * no_of_samples)+1     : round(train_plus_val_size * no_of_samples) ]
        StopData_ext =  data[ (round(train_size * no_of_samples)+1
                                -global_settings["seq_length"])       : round(train_plus_val_size * no_of_samples) ]
                            #extend data according to delays/sequence length

        OptData      =  StopData.copy(deep = True)
        OptData_ext  =  StopData_ext.copy(deep = True)

        TestData     =  data[ round(train_plus_val_size * no_of_samples)+1   :      ]
        TestData_ext =  data[ (round(train_plus_val_size * no_of_samples)+1
                                -global_settings["seq_length"])              :      ]
                            #extend data according to delays/sequence length

        return (    TrainingData,
                    StopData, StopData_ext,
                    OptData, OptData_ext,
                    TestData, TestData_ext
                )


    def normalise_well_data(self, global_settings):
        well_data = []   # For storing normal data
        normalized_well_data = []   # For storing transformed data

        # Here we are fetching all the splits of all the wells in the list, i.e. the target well and its K nearest wells
        for index,_ in self.nearest_wells_list:
            list_of_split = self.split_data(index, global_settings)
            well_data.append(list_of_split)

        # Then we are concatenating all the training data matrices into a single giant training data matrix
        # This is SOLELY used to fit our scalers only and nothing else
        total_train_data = np.empty((0,0))
        for row in range(len(well_data)):
            if total_train_data.shape[0] == 0:
                total_train_data = well_data[row][0]
            else:
                total_train_data = np.concatenate((total_train_data, well_data[row][0]),axis=0)

        # Defining Normalizers for data
        self.feature_scaler = RobustScaler()
        self.gwl_scaler     = RobustScaler()
        self.num_features = total_train_data.shape[1] - (not global_settings['include_GWL_as_feature'])

        # Controling statement for groundwater inclusion as a FEATURE
        self.feature_scaler.fit(np.array(total_train_data)[:,:self.num_features])
        self.gwl_scaler.fit(np.array(total_train_data)[:,-1].reshape(-1,1))

        for list_of_split in well_data:
            normalised_single_well_data = []
            for data in list_of_split:
                # Controling statement for groundwater inclusion as a FEATURE
                X_transform = self.feature_scaler.transform(np.array(data)[:,:self.num_features])
                y_transform = self.gwl_scaler.transform(np.array(data)[:,-1].reshape(-1,1))
                y_transform = y_transform.reshape(-1,)
                normalised_single_well_data.append((X_transform,y_transform))

            normalized_well_data.append(normalised_single_well_data)

        return well_data, normalized_well_data


    def create_data_seq(self, global_settings):

        well_data, normalized_well_data = self.normalise_well_data(global_settings)

        ##Creating the Sequential Data for Model
        normalized_x_y_well_data = []
        for well_no in range(len(normalized_well_data)):
            normalized_x_y_single_well_data = []
            for split_no in range(len(normalized_well_data[0])):
                X,y = self.to_supervised(normalized_well_data[well_no][split_no], global_settings)
                normalized_x_y_single_well_data.append((X,y))

            normalized_x_y_well_data.append(normalized_x_y_single_well_data)

        ## Merging Seq length data  for model
        merged_well_data = []

        for split_no in range(len(well_data[0])):
            merged_data_current_split = np.empty((0,0,0))

            for well_no in range(len(well_data)):
                if merged_data_current_split.shape[0] == 0:  # Check if parent_X is empty
                    merged_data_current_split = well_data[well_no][split_no]
                else:
                    merged_data_current_split = pd.concat([merged_data_current_split,
                                                                well_data[well_no][split_no][0]], axis=0)

            merged_well_data.append(merged_data_current_split)

        ## Merging Seq length data  for model
        normalized_x_y_merged_well_data = []

        for split_no in range(len(normalized_x_y_well_data[0])):
            merged_X_data_current_split = np.empty((0,0,0))
            merged_y_data_current_split = np.empty((0,0))

            for well_no in range(len(normalized_x_y_well_data)):
                if merged_X_data_current_split.shape[0] == 0:  # Check if parent_X is empty
                    merged_X_data_current_split = normalized_x_y_well_data[well_no][split_no][0]
                    merged_y_data_current_split = normalized_x_y_well_data[well_no][split_no][1]
                else:
                    merged_X_data_current_split = np.concatenate((merged_X_data_current_split,
                                                                normalized_x_y_well_data[well_no][split_no][0]), axis=0)
                    merged_y_data_current_split = np.concatenate((merged_y_data_current_split,
                                                                normalized_x_y_well_data[well_no][split_no][1]), axis=0)

            normalized_x_y_merged_well_data.append((merged_X_data_current_split, merged_y_data_current_split))

        # normalized_well_data := This isn't return as its usage is only to created X,y samples only
        return merged_well_data, normalized_x_y_merged_well_data


    def prepare_data(self, global_settings):
        self.global_settings = global_settings
        self.global_settings.update({'num_features': self.num_features})
        self.nearest_wells_list = self.k_nearest_neighbours(global_settings)
        self.merged_well_data, self.normalized_x_y_merged_well_data = self.create_data_seq(global_settings)


# REDUNDANT FUNCTIONS : CURRENTLY NOT BEING USED

def load_GW_and_HYRAS_Data(i):
    well_list_file = fr"{DATA_FOLDER}/ListOfWellsGermany.txt"

    #load a list of all sites
    well_list = pd.read_csv(well_list_file, sep = ' ', decimal = '.')
    Well_ID = well_list.ID[i]

    #load and merge the data
    GWData = pd.read_csv(fr"{DATA_FOLDER}/GWData/{Well_ID}_GW-Data.csv",
                        parse_dates=['Date'],index_col=0, dayfirst = True,
                        decimal = '.', sep=',')

    # HYRASData = pd.read_csv(fr"{DATA_FOLDER}/HYRAS/{Well_ID}_weeklyData_HYRAS.csv",
    #                         parse_dates=['Date'],index_col=0, dayfirst = True,
    #                         decimal = '.', sep=',')

    HYRASData = pd.read_csv(fr"{DATA_FOLDER}/GW_phase2_Data/{Well_ID}_data.csv",
                            parse_dates=['Date'],index_col=0, dayfirst = True,
                            decimal = '.', sep=',').drop(columns = ['GWL'])

    HYRASData=HYRASData.reset_index()
    HYRASData['Date']=pd.to_datetime(HYRASData['Date'])
    HYRASData.set_index('Date', inplace = True)

    data = pd.merge(GWData, HYRASData, how='inner', left_index = True, right_index = True)

    return data, Well_ID

# =============================================================================
#### UTILITY FUNCTIONS
# =============================================================================

def split_data(data_, global_settings):
    no_of_samples = len(data)
    train_size = 1 - (global_settings['val_size'] + global_settings['test_size'])
    train_plus_val_size = 1 - (global_settings['test_size'])

    data = data_.copy()

    TrainingData =  data[                                                  : round(train_size * no_of_samples) ]

    StopData     =  data[ round(train_size * no_of_samples)+1              : round(train_plus_val_size * no_of_samples) ]
    StopData_ext =  data[ (round(train_size * no_of_samples)+1
                                -global_settings["seq_length"])            : round(train_plus_val_size * no_of_samples) ] #extend data according to dealys/sequence length

    OptData      =  StopData.copy(deep = True)
    OptData_ext  =  StopData_ext.copy(deep = True)

    TestData     =  data[ round(train_plus_val_size * no_of_samples)+1     :                            ]
    TestData_ext =  data[ (round(train_plus_val_size * no_of_samples)+1
                                -global_settings["seq_length"])            :                            ] #extend data according to dealys/sequence length

    return (    TrainingData,
                StopData, StopData_ext,
                OptData, OptData_ext,
                TestData, TestData_ext
            )


def to_supervised(data, global_settings):
    no_of_samples = len(data)

    X, Y = list(), list()
    # step over the entire history one time step at a time
    for i in range(no_of_samples):
        # find the end of this pattern
        end_idx = i + global_settings["seq_length"]
        # check if we are beyond the data
        if end_idx >= no_of_samples:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = data[i:end_idx, (not global_settings['include_GWL_as_feature']) : ], data[end_idx, 0]
        X.append(seq_x)
        Y.append(seq_y)
    return np.array(X), np.array(Y)