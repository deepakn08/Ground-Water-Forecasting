# RESULT_FOLDER = "well_results"
# RESULT_FOLDER = "well_results_new_lstm"
BASE_PROJECT = ''
RESULT_FOLDER = f"RESULTS_{BASE_PROJECT}"

import numpy as np, pandas as pd
from bayes_opt.logger import JSONLogger
from scipy import stats
# from sklearn.preprocessing import RobustScaler
from uncertainties import unumpy

import datetime, copy, gc
import pprint

from lstm_model import *
from data_preparation import *

class Sweep:
    # A singleton class, used to transfer and use global variables across python scripts without any hassle
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Sweep, cls).__new__(cls)
            cls._instance.optimizer = None
            cls._instance.time_total_process = None
            cls._instance.time_single_well = None
        return cls._instance

PREFIX_DENSE_SIZE_PARAM = 'dense'
POSTFIX_PHRASE_DENSE_SIZE_PARAM = '_size'

# =============================================================================
#### FUNCTIONS TO PERFORM BAYES OPTIMIZATION
# =============================================================================
## IMPORTANT NOTE_
## IF A FUNCTION DOESN'T USES "MCDropout" LAYER THEN WITHIN THE 2ND FUNCTION, WITHIN THE  1ST FOR LOOP THAT YOU SEE
## INSIDE THIS LOOP IN THE 4TH LAST LINE CHANGE THE LAST PARAMETER THAT IS BEING PASSED, TO 1

def bayesOpt_function_wrapper(_global_settings):
    # global optimizer, time_total_process, time_single_config

    def bayesOpt_function(**kwargs):
        bayes_sweep = Sweep()
        optimizer = bayes_sweep.optimizer
        time_total_process = bayes_sweep.time_total_process
        time_single_well = bayes_sweep.time_single_well

        # tunable_params = {k: int(v) for k, v in kwargs.items()}
        tunable_params = {}
        for key, value in kwargs.items():

            if key in ['bidirectional', 'state_management']:
                if -0.5 <= value < 0.5:
                    value = False
                elif 0.5 <= value <= 1.5:
                    value = True
                tunable_params.update({key: value})

            elif key == 'use_state_layernorms':
                if -0.5 <= value < 0.5:
                    value = 0
                elif 0.5 <= value < 1.5:
                    value = 1
                elif 1.5 <= value <= 2.5:
                    value = 2
                tunable_params.update({key: value})

            else:
                tunable_params.update({key: int(value)})

        global_settings = copy.deepcopy(_global_settings)
        global_settings.update(tunable_params)

        start_pharse = PREFIX_DENSE_SIZE_PARAM
        end_phrase = POSTFIX_PHRASE_DENSE_SIZE_PARAM
        dense_size_dict = {    key: tunable_params[key]
                                    for key in tunable_params
                                        if (key.startswith(start_pharse) and key.endswith(end_phrase))
                            }
        dense_units_list = []
        # Loop through the keys in order and collect values
        for i in range(1, len(dense_size_dict) + 1):
            key = f'{start_pharse}{i}{end_phrase}'
            if key in dense_size_dict:
                dense_units_list.append(dense_size_dict[key])

        global_settings.update({'dense_units_list': dense_units_list})

        data_manager = Data_preprocessor()
        data_manager.prepare_data(global_settings)
        global_settings.update({'num_features': data_manager.num_features})
        scaler_gwl = data_manager.gwl_scaler
        TrainingData, StopData, _, _, _, _, _ = data_manager.merged_well_data

        (NormalizedTrainingData_xy,
        _, NormalizedStopData_ext_xy,
        _, NormalizedOptData_ext_xy, _, _) = data_manager.normalized_x_y_merged_well_data

        X_train, Y_train = NormalizedTrainingData_xy
        X_stop, Y_stop = NormalizedStopData_ext_xy
        X_opt, Y_opt = NormalizedOptData_ext_xy


        #build and train model with idifferent initializations
        inimax = global_settings['seeds_per_config'] #3
        optresults_members = np.zeros((len(X_opt), global_settings['prediction_length'], inimax))

        if not global_settings['disable_wandb']:
            global_settings.update({
                                    'use_wandb'  : True,
                                    'runs_group' : f'config_{len(optimizer.res)+1}'
                                    })

        for ini in range(inimax):
            if not global_settings['disable_wandb']:
                global_settings.update({'run_name': f'config_{len(optimizer.res)+1}__run_{ini}'})

            print("(target_well_no:{}) BayesOpt-Iteration {} - ini-Ensemblemember {}".format(
                            global_settings['target_well_no'], len(optimizer.res)+1, ini+1))

            model, history = gwmodel(ini,global_settings,X_train, Y_train, X_stop, Y_stop)
            opt_sim_n = model.predict(X_opt)

            del model
            gc.collect()

            opt_sim = scaler_gwl.inverse_transform(opt_sim_n)
            optresults_members[:, :, ini] = opt_sim.reshape(-1, global_settings['prediction_length'])

        opt_sim_median = np.median(optresults_members,axis = 2)

        sim = np.asarray(opt_sim_median.reshape(-1, global_settings['prediction_length']))
        obs = np.asarray(scaler_gwl.inverse_transform(Y_opt.reshape(-1, global_settings['prediction_length'])))
        err = sim-obs

        meanTrainingGWL = np.mean(np.asarray(TrainingData['GWL']))
        meanStopGWL = np.mean(np.asarray(StopData['GWL']))
        err_nash = obs - np.mean([meanTrainingGWL, meanStopGWL])

        r2_values_per_timestep = []
        for timestep in range(global_settings['prediction_length']):
            r = stats.linregress(sim[:,timestep], obs[:,timestep])
            r2_values_per_timestep.append(r.rvalue ** 2)

        mean_r2_value = np.mean(r2_values_per_timestep)

        nse_per_timestep = 1 - ( np.sum(err ** 2, axis = 0) / np.sum(err_nash ** 2, axis = 0) )
        mean_nse = np.mean(nse_per_timestep)

        print("total elapsed time = {}".format(datetime.datetime.now()-time_total_process))
        print("(target_well_no = {}) elapsed time = {}".format(global_settings['target_well_no'],datetime.datetime.now()-time_single_well))

        return mean_nse + mean_r2_value #NSE+R²: (max = 2)

    return bayesOpt_function


def simulate_testset(Well_ID, tuned_params, global_settings):

    global_settings.update(tuned_params)

    start_pharse = PREFIX_DENSE_SIZE_PARAM
    end_phrase = POSTFIX_PHRASE_DENSE_SIZE_PARAM
    dense_size_dict = {    key: tuned_params[key]
                                for key in tuned_params
                                    if (key.startswith(start_pharse) and key.endswith(end_phrase))
                        }
    dense_units_list = []
    # Loop through the keys in order and collect values
    for i in range(1, len(dense_size_dict) + 1):
        key = f'{start_pharse}{i}{end_phrase}'
        if key in dense_size_dict:
            dense_units_list.append(dense_size_dict[key])

    global_settings.update({'dense_units_list': dense_units_list})

    data_manager = Data_preprocessor()
    data_manager.prepare_data(global_settings)
    global_settings.update({'num_features': data_manager.num_features})
    scaler_gwl = data_manager.gwl_scaler
    TrainingData, StopData, _, _, _, TestData, _ = data_manager.merged_well_data
    data = pd.concat([TrainingData, StopData, TestData], axis = 0)

    (NormalizedTrainingData_xy,
    _, NormalizedStopData_ext_xy,
    _, _, _, NormalizedTestData_ext_xy) = data_manager.normalized_x_y_merged_well_data

    X_train, Y_train = NormalizedTrainingData_xy
    X_stop, Y_stop = NormalizedStopData_ext_xy
    X_test, Y_test = NormalizedTestData_ext_xy


    #build and train model with different initializations
    inimax = global_settings['seeds_for_best_config'] #7
    sim_members = np.zeros((len(X_test), global_settings['prediction_length'], inimax))
    sim_members[:] = np.nan

    sim_std = np.zeros((len(X_test), global_settings['prediction_length'], inimax))
    sim_std[:] = np.nan

    target_well_no = global_settings['target_well_no']
    pathResults = global_settings['pathResults']

    if not global_settings['disable_wandb']:
        global_settings.update({'use_wandb': False})

    f = open(fr'{pathResults}/traininghistory_{Well_ID}.txt', "w")
    for ini in range(inimax):
        model,history = gwmodel(ini,global_settings,X_train, Y_train, X_stop, Y_stop)

        loss = np.zeros((1, 100))
        loss[:,:] = np.nan
        loss[0,0:np.shape(history.history['loss'])[0]] = history.history['loss']

        val_loss = np.zeros((1, 100))
        val_loss[:,:] = np.nan
        val_loss[0,0:np.shape(history.history['val_loss'])[0]] = history.history['val_loss']

        print('loss', file = f)
        print(loss.tolist(), file = f)
        print('val_loss', file = f)
        print(val_loss.tolist(), file = f)

        #make prediction 100 times for each ini
        y_pred_distribution = predict_distribution(X_test, model, global_settings['predict_repetitions'])

        del model
        gc.collect()

        sim = scaler_gwl.inverse_transform(y_pred_distribution.reshape(-1, global_settings['prediction_length']))
        sim = sim.reshape(-1, global_settings['prediction_length'], global_settings['predict_repetitions'])
        sim_members[:, :, ini], sim_std[:, :, ini]= sim.mean(axis=2), sim.std(axis=2)

        print(f"{ini}th sim_test trial, on best model, for well {target_well_no} done")

    f.close()
    sim_members_uncertainty = unumpy.uarray(sim_members,1.96*sim_std) #1.96 because of sigma rule for 95% confidence
    sim_mean = np.nanmedian(sim_members,axis = 2)
    sim_mean_uncertainty = np.sum(sim_members_uncertainty,axis = 2)/inimax


    # get scores
    sim = np.asarray(sim_mean)
    obs = np.asarray(scaler_gwl.inverse_transform(Y_test))
    err = sim-obs
    err_rel = (sim-obs)/(np.max(data['GWL'])-np.min(data['GWL']))
    err_nash = obs - np.mean(np.asarray(
                                        data['GWL'].iloc[0:round(0.9 * len(data))]
                                        ))

    nse_per_timestep = 1 - ( np.sum(err ** 2, axis = 0) / np.sum(err_nash ** 2, axis = 0) )
    NSE = np.mean(nse_per_timestep)

    r2_per_timestep = []
    for timestep in range(global_settings['prediction_length']):
            r = stats.linregress(sim[:,timestep], obs[:,timestep])
            r2_per_timestep.append(r.rvalue ** 2)
    R2 = np.mean(r2_per_timestep)

    rmse_per_timestep = np.sqrt(np.mean(err ** 2, axis = 0))
    RMSE =  np.mean(rmse_per_timestep)

    rrmse_per_timestep = np.sqrt(np.mean(err_rel ** 2, axis = 0)) * 100
    rRMSE = np.mean(rrmse_per_timestep)

    bias_per_timestep = np.mean(err, axis = 0)
    Bias = np.mean(bias_per_timestep)

    rbias_per_timestep = np.mean(err_rel, axis = 0) * 100
    rBias = np.mean(rbias_per_timestep)

    scores = pd.DataFrame(np.array([[NSE, R2, RMSE, rRMSE, Bias, rBias]]),
                columns=['NSE','R2','RMSE','rRMSE','Bias','rBias'])
    # sim1=sim # These seem IMPORTANT don't remove them further discretion is need
    # obs1=obs

    scores_per_timestep = {} #np.zeros((6, global_settings['prediction_length']))
    scores_per_timestep['NSE'] = np.array(nse_per_timestep)
    scores_per_timestep['R2'] = np.array(r2_per_timestep)
    scores_per_timestep['RMSE'] = np.array(rmse_per_timestep)
    scores_per_timestep['rRMSE'] = np.array(rrmse_per_timestep)
    scores_per_timestep['Bias'] = np.array(bias_per_timestep)
    scores_per_timestep['rBias'] = np.array(rbias_per_timestep)
    scores_per_timestep = pd.DataFrame(scores_per_timestep, columns=['NSE','R2','RMSE','rRMSE','Bias','rBias'])

    print('Overall performance \n--------------------------------\n')
    print(scores)
    print('\n\nTimestep wise performance \n--------------------------------\n')
    print(scores_per_timestep)

    f = open(fr'{pathResults}/final_performance_report_{Well_ID}.txt', "w")
    print('Overall performance \n--------------------------------\n', file = f)
    print(scores, file = f)
    print('\n\nTimestep wise performance \n--------------------------------\n', file = f)
    print(scores_per_timestep, file = f)
    f.close()

    # INCORRECT calculations here
    errors = np.zeros((inimax,6))
    errors[:] = np.nan
    for i in range(inimax):
        # sim = np.asarray(sim_members[:,:,i].reshape(-1,1))
        # err = sim-obs
        # err_rel = (sim-obs)/(np.max(data['GWL'])-np.min(data['GWL']))
        errors[i,0] = NSE
        errors[i,1] = R2
        errors[i,2] = RMSE
        errors[i,3] = rRMSE
        errors[i,4] = Bias
        errors[i,5] = rBias


    return (scores, scores_per_timestep,
            TestData, sim, obs, # Originally it was sim1 and obs1
            inimax, sim_members, Well_ID, errors,
            sim_members_uncertainty, sim_mean_uncertainty)


class newJSONLogger(JSONLogger) :
    def __init__(self, path):
        self._path=None
        super(JSONLogger, self).__init__()
        self._path = path if path[-5:] == ".json" else path + ".json"

