import os
import sys
import shutil

from matplotlib.pyplot import disconnect
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../SGMANet'))
import visual
import data_loader
import eval

import run_base_line
import run_ants_affine
import run_sync
import run_niftyreg
import run_convexAdam
import run_model
import run_random_affine
import argparse
import pandas as pd
import datetime as dt

import numpy as np

import json
import vessel_dataloader


supported_method = {
                        "baseline" : run_base_line.baseline_reg,
                        "random_affine" : run_random_affine.random_affine_reg,
                        "ants_affine" : run_ants_affine.ants_reg,
                        "sync" : run_sync.sync_reg,
                        "niftyreg" : run_niftyreg.nifty_reg,
                        "convex_adam" : run_convexAdam.convexadam_reg,
                        "voxelmorph" : run_model.model_reg,
                        "voxelmorph-diff" : run_model.model_reg,
                        "transmorph-diff" : run_model.model_reg,
                        "mambamorph-diff" : run_model.model_reg,
                        "voxelmorph-diff-dice" : run_model.model_reg,
                        "voxelmorph-diff-dice-boundary" : run_model.model_reg,
                        "voxelmorph-diff-dice-boundary-unoverlop": run_model.model_reg,
                        "voxelmorph-diff-dice-boundary-unoverlop-smooth": run_model.model_reg,
                        "transmorph" : run_model.model_reg,
                        "affine_mi" : run_model.model_reg,
                        "affine_struct" : run_model.model_reg,
                        "sgmaconv": run_model.model_reg,
                        "sgmatrans_0": run_model.model_reg,
                        "sgmatrans_1": run_model.model_reg,
                        "sgmatrans_2": run_model.model_reg,
                        "vmamba-diff": run_model.model_reg,
                        "vmamba-diff-dice": run_model.model_reg,
                        "vmamba-diff-dice-boundary": run_model.model_reg,
                        "vmamba-diff-dice-boundary-smooth": run_model.model_reg,
                        }


def setup_data(config_data, data_type="train", dataset_vessel=False):
    run_intra_registration = True if config_data["exp"] == 'intra' else False

    this_dir_path = os.path.dirname(__file__)
    json_path = os.path.join(this_dir_path, config_data["path"])


    affine_mat_dir = os.path.join(os.path.dirname(__file__), "../", config_data["path_affine_mat"]) if config_data["path_affine_mat"] else None

    if data_type=="train" and not config_data["train_use_mat"]:
        affine_mat_dir = None

    if data_type=="valid" and not config_data["valid_use_mat"]:
        affine_mat_dir = None

    if data_type=="test" and not config_data["test_use_mat"]:
        affine_mat_dir = None

    if not dataset_vessel:
        dataset = data_loader.RegDataSet(json_path, 
                                     intra_patient=run_intra_registration, 
                                     affine_mat=affine_mat_dir, 
                                     type=data_type)
    else:
        dataset = vessel_dataloader.vessel_dataset(json_path,
                                                    intra_patient=run_intra_registration,
                                                    type=data_type)        

    return dataset

def save_results(result, moving_image_name, fix_image_name, result_dir, fix_image_path):
    if "moved_image" in result:
        image_name = f"{moving_image_name}_{fix_image_name}_image.nii.gz"
        path_to_moved_image = os.path.join(result_dir, image_name)
        visual.save_image(result["moved_image"], path_to_moved_image, fix_image_path)

    if "moved_label" in result:
        moved_label_name = f"{moving_image_name}_{fix_image_name}_label.nii.gz"
        path_to_moved_label = os.path.join(result_dir, moved_label_name)
        visual.save_image(result["moved_label"], path_to_moved_label, fix_image_path)

    # if "moved_portalvein" in result:
    #     portalvein_name = f"{moving_image_name}_{fix_image_name}_portalvein.nii.gz"
    #     path_to_moved_portalvein = os.path.join(result_dir, portalvein_name)
    #     visual.save_image(result["moved_portalvein"], path_to_moved_portalvein, fix_image_path)

    # if "moved_venacava" in result:
    #     venacava_name = f"{moving_image_name}_{fix_image_name}_venacava.nii.gz"
    #     path_to_moved_venacava = os.path.join(result_dir, venacava_name)
    #     visual.save_image(result["moved_venacava"], path_to_moved_venacava, fix_image_path)

    if "moved_portalvein" in result and "moved_venacava" in result:
        vessel_name = f"{moving_image_name}_{fix_image_name}_vessel.nii.gz"
        path_to_moved_vessel = os.path.join(result_dir, vessel_name)
        portalvein_tensor = eval.label_to_tensor(result["moved_portalvein"])
        venacava_tensor = eval.label_to_tensor(result["moved_venacava"])

        if torch.is_floating_point(portalvein_tensor):
            portalvein_tensor = (portalvein_tensor > 0.5).to(torch.uint8)
        
        if torch.is_floating_point(venacava_tensor):
            venacava_tensor = (venacava_tensor > 0.5).to(torch.uint8)

        vessel_tensor = torch.zeros_like(portalvein_tensor, dtype=torch.uint8)
        vessel_tensor[portalvein_tensor == 1] = 1
        vessel_tensor[venacava_tensor == 1] = 2

        visual.save_image(vessel_tensor, path_to_moved_vessel, fix_image_path)

    if "svf" in result:
        svf_name = f"{moving_image_name}_{fix_image_name}_svf.nii.gz"
        path_to_svf = os.path.join(result_dir, svf_name)
        visual.save_deformation(result["svf"], path_to_svf, fix_image_path)


def save_affine_matrix(mat_dir, moving_image_path, fix_image_path, result):
        if not os.path.exists(mat_dir):
            os.makedirs(mat_dir) 

        _, moving_image_name = os.path.split(moving_image_path)
        _, fix_image_name = os.path.split(fix_image_path)
        moving_image_name = moving_image_name.split('.')[0]
        fix_image_name = fix_image_name.split('.')[0]

        mat_name = fix_image_name + '_' + moving_image_name + '.mat'
        mat_path = os.path.join(mat_dir, mat_name)

        if type(result["affine_matrix"]) == str:
            shutil.copy(result['fwdtransforms'], mat_path)
        elif type(result["affine_matrix"]) == torch.Tensor:
            torch.save(result["affine_matrix"], mat_path)
        else:
            raise ValueError("Unsupported format!")
        
def process_result(result, fix_label, moving_image_name, fix_image_name):
    record_node = {}
    record_node["moving_image"] = moving_image_name
    record_node["fix_image"] = fix_image_name

    # calculate dice score
    fixed_onehot = fix_label.unsqueeze(0)
    moved_onehot = eval.label_to_onehot(result["moved_label"]).unsqueeze(0)
    dsc = eval.dsc_one_hot(fixed_onehot, moved_onehot).numpy()
    record_node["dsc_0"] = dsc[0]
    record_node["dsc_1"] = dsc[1]
    record_node["dsc_2"] = dsc[2]
    record_node["dsc_3"] = dsc[3]
    record_node["avg_dsc"] = np.mean(dsc)

    result_string = f'move: {record_node["moving_image"]}, fix: {record_node["fix_image"]}, dsc: {record_node["avg_dsc"]: .2f}'

    # calculate jicobal
    if result["svf"] is not None:
        svf_tensor = eval.disp_def_to_tensor(result["svf"])
        jcob = eval.jacobian_determinant_pytorch(svf_tensor)
        record_node["jcob"] = jcob
        result_string = result_string + f', jcob: {record_node["jcob"]: .6f}'

    if "run_time" in result:
        record_node["run_time"] = result["run_time"]
        result_string += f', run_time: {record_node["run_time"]: .6f}'

    return record_node, result_string

def process_result_vessel(result, fix_label, moving_label, moving_image_name, fix_image_name, fix_image_path):
    record_node = {}
    record_node["moving_image"] = moving_image_name
    record_node["fix_image"] = fix_image_name

    # calculate dice score
    liver_label_fixed = torch.from_numpy(fix_label[0]).unsqueeze(0).unsqueeze(0)
    liver_label_moved = eval.label_to_tensor(result["moved_label"]).unsqueeze(0).unsqueeze(0)

    # Check if the liver_label_moved is of type uint8
    if torch.is_floating_point(liver_label_moved):
        raise TypeError("liver_label_moved must be of type uint8")
    
    
    if torch.is_floating_point(liver_label_fixed):
        raise TypeError("liver_label_fixed must be of type uint8")

    dsc = eval.dsc_one_hot(liver_label_fixed, liver_label_moved).numpy()
    record_node["liver_dsc"] = dsc[0]

    ravd = eval.compute_ravd(liver_label_fixed, liver_label_moved)
    record_node["liver_ravd"] = ravd

    assd, _, mssd = eval.get_surface_metrics(liver_label_fixed, liver_label_moved, fix_image_path)
    record_node["liver_assd"] = assd
    record_node["liver_mssd"] = mssd

    portalvein_label_fixed = torch.from_numpy(fix_label[1]).unsqueeze(0).unsqueeze(0)
    portalvein_label_moved = eval.label_to_tensor(result["moved_portalvein"]).unsqueeze(0).unsqueeze(0)
    venacava_label_fixed = torch.from_numpy(fix_label[2]).unsqueeze(0).unsqueeze(0)
    venacava_label_moved = eval.label_to_tensor(result["moved_venacava"]).unsqueeze(0).unsqueeze(0)

    # Threshold and convert to uint8
    if torch.is_floating_point(portalvein_label_moved):
        portalvein_label_moved = (portalvein_label_moved > 0.5).to(torch.uint8)
    
    if torch.is_floating_point(venacava_label_moved):
        venacava_label_moved = (venacava_label_moved > 0.5).to(torch.uint8)

    portalvein_dsc = eval.dsc_one_hot(portalvein_label_fixed, portalvein_label_moved).numpy()
    record_node["portalvein_dsc"] = portalvein_dsc[0]   
    
    venacava_dsc = eval.dsc_one_hot(venacava_label_fixed, venacava_label_moved).numpy()
    record_node["venacava_dsc"] = venacava_dsc[0]

    vessel_fixed = portalvein_label_fixed + venacava_label_fixed
    vessel_moved = portalvein_label_moved + venacava_label_moved
    vessel_fixed[vessel_fixed > 1] = 1
    vessel_moved[vessel_moved > 1] = 1
    vessel_dsc = eval.dsc_one_hot(vessel_fixed, vessel_moved).numpy()
    record_node["vessel_dsc"] = vessel_dsc[0]

    overlap = eval.dsc_one_hot(portalvein_label_moved, venacava_label_moved).numpy()
    record_node["overlap"] = overlap[0]

    disconnected_moving_portalvein = eval.count_disconnected_regions(moving_label[1])
    disconnected_moving_venacava = eval.count_disconnected_regions(moving_label[2])
    disconnected_moved_portalvein = eval.count_disconnected_regions(portalvein_label_moved)
    disconnected_moved_venacava = eval.count_disconnected_regions(venacava_label_moved)
    record_node["disconnection_portalvein"] = disconnected_moved_portalvein - disconnected_moving_portalvein
    record_node["disconnection_venacava"] = disconnected_moved_venacava - disconnected_moving_venacava

    record_node["disconnection_vessel"] = record_node["disconnection_portalvein"] + record_node["disconnection_venacava"]

    # calculate jicobal
    if result["svf"] is not None:
        svf_tensor = eval.disp_def_to_tensor(result["svf"])
        jcob = eval.jacobian_determinant_pytorch(svf_tensor)
        record_node["jcob"] = jcob

    if "run_time" in result:
        record_node["run_time"] = result["run_time"]

    # Print the record_node key-value pairs in one line
    result_string = "Record Node: " + ", ".join([f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}" for key, value in record_node.items()])

    return record_node, result_string

if __name__ == "__main__":

    # Get dataset config 
    with open(os.path.join(os.path.dirname(__file__), "../SGMANet/config/dataset.json"), "r") as f:
        config_dataset_all = json.load(f)

    # Parser input
    parser = argparse.ArgumentParser(description = 'Run compare test')

    parser.add_argument('-m', '--method', dest='method',  choices=supported_method.keys(), help='method to be used')
    parser.add_argument('-d', '--data', dest='dataset', choices=config_dataset_all.keys(), help="dataset to use")
    parser.add_argument('-v', '--visual', dest='visual', action='store_true', help="save the moved image and label for visualization")
    parser.add_argument('-t', '--data_type', dest='data_type', default="test", help="data_type: train, valid, test")
    parser.add_argument('-p', '--model_path', dest='model_path', default=None, help="path to the saved model")
    parser.add_argument('-n', '--mat', dest='mat_path', default=None, help="path to save the linear transformation matrix")
    parser.add_argument('-i', '--index', dest='index', default=None, type=int, help="index to draw")


    args = parser.parse_args()

    # Set up the data loader
    config_dataset = config_dataset_all[args.dataset]
    dataset = setup_data(config_dataset, args.data_type, dataset_vessel=(args.dataset in ["3dircab", "khp"]))


    #create dir to save model data
    now = dt.datetime.now()
    result_dir_data = f"{args.dataset}"
    result_dir_method = f"{args.method}"
    result_dir_time = f"{now:%Y-%m-%d-%H-%M-%S}"
    result_dir = os.path.join(os.path.dirname(__file__), "./results", result_dir_data, result_dir_method, result_dir_time)

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)      

    #save config
    config = vars(args)
    config["dataset_param"] = config_dataset
 

    # setup method
    method = supported_method[args.method]  
    method = method(config)

    if hasattr(method,  "model_path"):
        config["model_path"] = method.model_path

    # Run test
    record_note_list = []

    for index, (moving, fix) in enumerate(dataset):

        if args.index != None:
            if index != args.index:
                continue
            
        # Record the path
        moving_path = moving[-1]
        fix_path = fix[-1]

        moving_image_path = moving_path["image"]
        fix_image_path = fix_path["image"]

        if args.dataset in ["3dircab"]:
            moving_image_name = moving_image_path.split('/')[-2]
            fix_image_name = fix_image_path.split('/')[-2]
        elif args.dataset in ["khp"]:
            moving_image_name = moving_image_path.split('/')[-1].split('_')[1]
            fix_image_name = fix_image_path.split('/')[-1].split('_')[1]
        else:
            moving_image_name = moving_image_path.split('/')[-1]
            fix_image_name = fix_image_path.split('/')[-1]    

        result = method.registration(moving, fix)

        # Save the result for visualization
        if args.visual:
            save_results(result, moving_image_name, fix_image_name, result_dir, fix_image_path)

        if args.mat_path and args.method in ["ants_affine", "affine_mi", "affine_mine", "affine_struct"]:
            save_affine_matrix(args.mat_path, moving_image_path, fix_image_path, result)

        # evaluation
        if args.dataset in ["3dircab", "khp"]:
            record_node, result_string = process_result_vessel(result, fix[2], moving[2], moving_image_name, fix_image_name, fix_image_path)
        else:
            record_node, result_string = process_result(result, fix[2], moving_image_name, fix_image_name)
        
        record_note_list.append(record_node)
        print(result_string)


    df = pd.DataFrame.from_records(record_note_list)
    summary = df.describe()
    summary = summary.loc[['mean', 'std']]

    # save config
    with open(os.path.join(result_dir, "config.json"), 'w') as f:
        json.dump(config, f, indent = 4)   

    # save to excel
    file_name = f"result_{now:%Y_%m_%d_%H_%M_%S}.xlsx"
    file_name = os.path.join(result_dir, file_name)

    with pd.ExcelWriter(file_name) as writer:  
        df.to_excel(writer, sheet_name='data', index=False, float_format='%.6f')
        summary.to_excel(writer, sheet_name='summary', float_format='%.6f')