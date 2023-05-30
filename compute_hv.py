import torch
from botorch.utils.multi_objective.box_decompositions.dominated import (
    DominatedPartitioning,
)
from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd
import argparse
import pathlib

qnehvi_success_citeseq = ['upbeat-sweep-100',
 'ethereal-sweep-97',
 'cool-sweep-94',
 'pleasant-sweep-93',
 'trim-sweep-90',
 'grateful-sweep-89',
 'youthful-sweep-85',
 'genial-sweep-83',
 'vocal-sweep-78',
 'apricot-sweep-75',
 'quiet-sweep-74',
 'skilled-sweep-73',
 'avid-sweep-67',
 'dry-sweep-66',
 'devoted-sweep-63',
 'playful-sweep-56',
 'dulcet-sweep-52',
 'cool-sweep-49',
 'sandy-sweep-48',
 'ruby-sweep-47',
 'still-sweep-46',
 'kind-sweep-43',
 'vivid-sweep-41',
 'lucky-sweep-37',
 'devout-sweep-35',
 'magic-sweep-27',
 'lively-sweep-26',
 'peachy-sweep-21',
 'youthful-sweep-20',
 'pious-sweep-18',
 'tough-sweep-16',
 'expert-sweep-11',
 'true-sweep-4',
 'crimson-sweep-5',
 'giddy-sweep-3',
 'pretty-sweep-1']

qparego_success_citeseq = ['different-sweep-98',
 'bumbling-sweep-93',
 'true-sweep-92',
 'comfy-sweep-90',
 'hopeful-sweep-89',
 'eternal-sweep-86',
 'dauntless-sweep-84',
 'gentle-sweep-81',
 'proud-sweep-74',
 'jumping-sweep-73',
 'glowing-sweep-67',
 'pleasant-sweep-66',
 'major-sweep-65',
 'stellar-sweep-57',
 'wild-sweep-52',
 'frosty-sweep-49',
 'crisp-sweep-48',
 'divine-sweep-45',
 'royal-sweep-42',
 'rich-sweep-41',
 'golden-sweep-40',
 'dry-sweep-38',
 'scarlet-sweep-37',
 'likely-sweep-35',
 'sweet-sweep-30',
 'cool-sweep-29',
 'peachy-sweep-26',
 'glad-sweep-23',
 'sunny-sweep-20',
 'leafy-sweep-15',
 'silvery-sweep-16',
 'laced-sweep-12',
 'radiant-sweep-10',
 'clear-sweep-9',
 'rural-sweep-7',
 'still-sweep-4',
 'fearless-sweep-3',
 'true-sweep-1']

qnehvi_success_imc = ['brisk-sweep-96',
 'pretty-sweep-95',
 'avid-sweep-94',
 'winter-sweep-93',
 'radiant-sweep-87',
 'usual-sweep-86',
 'apricot-sweep-85',
 'jumping-sweep-84',
 'silver-sweep-83',
 'trim-sweep-81',
 'hopeful-sweep-78',
 'spring-sweep-77',
 'graceful-sweep-76',
 'wild-sweep-72',
 'apricot-sweep-71',
 'decent-sweep-69',
 'jolly-sweep-68',
 'morning-sweep-65',
 'stellar-sweep-63',
 'fearless-sweep-62',
 'twilight-sweep-61',
 'cosmic-sweep-60',
 'eager-sweep-56',
 'brisk-sweep-53',
 'jolly-sweep-48',
 'decent-sweep-46',
 'classic-sweep-45',
 'dry-sweep-41',
 'zesty-sweep-39',
 'serene-sweep-37',
 'warm-sweep-36',
 'dulcet-sweep-34',
 'dauntless-sweep-33',
 'chocolate-sweep-27',
 'brisk-sweep-26',
 'scarlet-sweep-25',
 'celestial-sweep-24',
 'trim-sweep-22',
 'radiant-sweep-21',
 'rose-sweep-19',
 'zany-sweep-17',
 'light-sweep-14',
 'genial-sweep-13',
 'dry-sweep-12',
 'fiery-sweep-11',
 'polished-sweep-7',
 'polished-sweep-6',
 'helpful-sweep-4',
 'deep-sweep-1']

qparego_success_imc = ['clear-sweep-95',
 'effortless-sweep-88',
 'comic-sweep-87',
 'lucky-sweep-84',
 'serene-sweep-81',
 'graceful-sweep-76',
 'good-sweep-72',
 'leafy-sweep-71',
 'vocal-sweep-70',
 'flowing-sweep-61',
 'dauntless-sweep-55',
 'dutiful-sweep-50',
 'confused-sweep-39',
 'electric-sweep-37',
 'charmed-sweep-26',
 'denim-sweep-17',
 'comic-sweep-11',
 'zesty-sweep-8']

tkwargs = {
        "dtype": torch.float32,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }

def make_hv_df(array, method):
    hvs_df = pd.DataFrame(array.cpu())
    hvs_df['run ID'] = range(array.shape[0])
    hvs_df['method'] = method
    hvs_df = pd.melt(hvs_df, id_vars=['run ID', 'method'])
    hvs_df.columns = ['run ID', 'method', 'Acquisition step', 'Hypervolume']
    return hvs_df

def compute_hv(acquisitions, num_init_points, ref_point):
    
    hv_array = torch.zeros((acquisitions.shape[0], acquisitions.shape[1] - num_init_points), **tkwargs)
    
    for run in range(acquisitions.shape[0]):
        for i in range(acquisitions.shape[1] - num_init_points):
            # compute hypervolume
            if type(acquisitions) is torch.Tensor:
                bd = DominatedPartitioning(ref_point=ref_point, 
                                           Y=acquisitions[run,:num_init_points+i+1,:])
            else:
                bd = DominatedPartitioning(ref_point=ref_point, 
                                           Y=torch.tensor(acquisitions, dtype=torch.float32)[run,:num_init_points+i+1,:])
            bd.to(device=tkwargs["device"])
            volume = bd.compute_hypervolume().item()
            hv_array[run, i] = volume
        if (run % 10) == 0:
            print(f"Run {run}")
    
    return hv_array

def export_acquisitions(path, saved_files, total_iter, num_tasks, method, experiment):
    
    if experiment == 'citeseq':
        qnehvi_success = qnehvi_success_citeseq
        qparego_success = qparego_success_citeseq
    elif experiment == 'imc':
        qnehvi_success = qnehvi_success_imc
        qparego_success = qparego_success_imc
    else:
        print("Invalid experiment")

    acquisitions = torch.zeros((len(saved_files), total_iter, num_tasks), **tkwargs)
    unsaved_runs = []

    if method == "botorch":
        acquisitions = torch.zeros((len(qnehvi_success), total_iter, num_tasks), **tkwargs)

        for i,run in enumerate(qnehvi_success):
            if run+'_output_dict.pt' in saved_files:
                tmp = torch.load(path+"/"+run+'_output_dict.pt')
                acquisitions[i,:,:] = tmp['train_y_original/'+method]
            else:
                print(f"{run} is not saved")    
                assert False

    elif method == "qparego":
        acquisitions = torch.zeros((len(qparego_success), total_iter, num_tasks), **tkwargs)

        for i,run in enumerate(qparego_success):
            if run+'_output_dict.pt' in saved_files:
                tmp = torch.load(path+'/'+run+'_output_dict.pt')
                acquisitions[i,:,:] = tmp['train_y_original/'+method]
            else:
                print(f"{run} is not saved")
                unsaved_runs.append(run)
        assert len(unsaved_runs) == 1, "Warning! Deleting runs will be wrong"
        unsaved_ind = [i for i in range(len(qparego_success)) if qparego_success[i] == unsaved_runs[0]][0]
        acquisitions = torch.cat((acquisitions[:unsaved_ind,:,:], acquisitions[unsaved_ind+1:,:,:]), axis=0)

    else:
        for i,run in enumerate(saved_files):
            tmp = torch.load(path+'/'+run)
            acquisitions[i,:,:] = tmp['train_y_original/'+method]

    return acquisitions

def scale_data(data, y_min, y_max):
    data = (data - y_min) / (y_max - y_min)
    return data

def scale_acquisitions(array, mins, maxes):
    if type(array) is torch.Tensor:
        scaled_array = torch.zeros_like(array, **tkwargs)
    else:
        scaled_array = torch.zeros_like(torch.tensor(array), **tkwargs)

    for obj in range(array.shape[2]):
        if type(array) is torch.Tensor:
            scaled_array[:,:,obj] = scale_data(array[:,:,obj], mins[obj], maxes[obj])
        else:
            scaled_array[:,:,obj] = torch.tensor(scale_data(array[:,:,obj], mins[obj], maxes[obj]))
    
    return scaled_array

def define_settings(experiment):
    citeseq_paths = ["manatee-run-dicts/citeseq-random-state-sweeps/p5q4rjku",
                     "manatee-run-dicts/citeseq-random-state-sweeps/gkqg9hiv_x9a5qwif",
                     "manatee-run-dicts/citeseq-random-state-sweeps/55nq05yl",
                     "manatee-run-dicts/citeseq-random-state-sweeps/v2qh0wsn_87fm2qv4",
                     "manatee-new/pipeline/citeseq_usemo-w9iqo4ar", 
                     "manatee-run-dicts/citeseq-random-state-sweeps/73eys3zm",
                     "manatee-run-dicts/citeseq-random-state-sweeps/4t7umvjt"]
    
    imc_paths = ["manatee-new/pipeline/final-imc-sweep/msa-ra-rs",
                "manatee-new/pipeline/final-imc-sweep/msa-ra-rs",
                "manatee-new/pipeline/final-imc-sweep/msa-ra-rs",
                "manatee-new/pipeline/final-imc-sweep/mas",
                "manatee-new/pipeline/imc_usemo-dew50r9d",
                "manatee-run-dicts/final-imc-sweep/qnehvi",
                "manatee-run-dicts/final-imc-sweep/qparego"]

    citeseq_num_tasks = 9
    citeseq_total_iter = 5+36
    imc_num_tasks = 7
    imc_total_iter = 5+35

    db_lower_bound = -18.588226318359375
    cal_upper_bound = 304.9784240722656
    citeseq_ref_point = torch.tensor([-1.,   # sil
                                  0.,    # cal
                                  db_lower_bound, #db
                                  -1., 
                                  -1., 
                                  -1., 
                                  -1., 
                                  -1.,
                                  -1.], **tkwargs)

    imc_ref_point = torch.tensor([-1.], **tkwargs).repeat(imc_num_tasks)

    citeseq_mins =  [-1., 0., db_lower_bound, -1., -1., -1., -1., -1., -1.]
    citeseq_maxes = [1., cal_upper_bound, 0., 1., 1., 1., 1., 1., 1.]
    imc_mins =  [-1., -1., -1., -1., -1., -1., -1.]
    imc_maxes = [1., 1., 1., 1., 1., 1., 1.]


    if experiment == "citeseq":
        paths = citeseq_paths
        total_iter = citeseq_total_iter
        num_tasks = citeseq_num_tasks
        ref_point = citeseq_ref_point
        mins = citeseq_mins
        maxes = citeseq_maxes
    elif experiment == 'imc':
        paths = imc_paths
        total_iter = imc_total_iter
        num_tasks = imc_num_tasks
        ref_point = imc_ref_point
        mins = imc_mins
        maxes = imc_maxes
    else:
        assert False

    settings = {}
    settings["paths"] = paths
    settings["total_iter"] = total_iter
    settings["num_tasks"] = num_tasks
    settings["ref_point"] = ref_point
    settings["mins"] = mins
    settings["maxes"] = maxes

    return settings

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute HV for experiment of choice.')
    parser.add_argument('--experiment', type=str, 
                        help='Type of experiment. Options: imc, citeseq')
    args = parser.parse_args()

    print(f"Using {tkwargs['device']}")

    global_path = "/home/campbell/aselega/Projects/"
        
    methods = ["manatee", "random prob", "random loc", "manatee", "usemo", "botorch", "qparego"]
    method_labels = ["M-SA", "RS", "RA", "M-AS", "USeMO", "qNEHVI", "qNParEGO"]
    num_init_points = 5
    
    settings = define_settings(args.experiment)
    
    for i, method in enumerate(methods):
        mypath = global_path + settings["paths"][i]
        print(f"Currently in directory {settings['paths'][i]}")
        files = [f for f in listdir(mypath) if isfile(join(mypath, f)) and f.endswith(".pt")]
    
        # Export acquisitions from .pt file
        acq_array = export_acquisitions(mypath, files, settings["total_iter"], settings["num_tasks"], method, args.experiment)

        # Scale acquisitions to [0,1]
        scaled_array = scale_acquisitions(acq_array, settings["mins"], settings["maxes"])

        # Compute HV
        hvs_array = compute_hv(scaled_array, num_init_points, settings["ref_point"])

        print(f"Processed {method_labels[i]}.")

        hvs_df = make_hv_df(hvs_array, method_labels[i])
        dir_path = pathlib.Path(f"{global_path}/manatee-run-dicts/hypervolumes/scaled-{args.experiment}/")
        dir_path.mkdir(parents=True, exist_ok=True)
        filename = method_labels[i] + "_hvs.csv"
        filepath = dir_path / f"{filename}"        
        hvs_df.to_csv(filepath)
        print(f"Written {filename}.")
