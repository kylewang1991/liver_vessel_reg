import os
import sys
import sys
import glob
from natsort import natsorted
import json
import argparse
import datetime as dt
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import train

import losses
import data_loader
import vessel_dataloader
import networks

from torchinfo import summary


    
def setup_data(config_data, random_transform_type='Rigid', data_type="train", dataset_vessel=False):
    run_intra_registration = True if config_data["param"]["exp"] == 'intra' else False

    this_dir_path = os.path.dirname(__file__)
    json_path = os.path.join(this_dir_path, config_data["param"]["path"])

    # if_shuffle = True if data_type=="train" else False

    affine_mat_dir = os.path.join(os.path.dirname(__file__), "../", config_data["param"]["path_affine_mat"]) if config_data["param"]["path_affine_mat"] else None

    if data_type=="train" and not config_data["param"]["train_use_mat"]:
        affine_mat_dir = None

    if data_type=="valid" and not config_data["param"]["valid_use_mat"]:
        affine_mat_dir = None

    if data_type=="test" and not config_data["param"]["test_use_mat"]:
        affine_mat_dir = None

    if not dataset_vessel:
        dataset = data_loader.RegDataSet(json_path, 
                                         intra_patient=run_intra_registration, 
                                         affine_mat=affine_mat_dir, 
                                         type=data_type,
                                         use_distance=config_data["use_distance"],
                                         use_weight=config_data["use_weight"],
                                         roi_index=config_data["roi_index"],
                                         distance_inner=config_data["distance_inner"],
                                         distance_outer=config_data["distance_outer"],
                                         segma = config_data["segma"],
                                         do_random_affine_data_enhancement=config_data["param"]["random_affine_enhancement"],
                                         random_type=random_transform_type)
        collate_func = data_loader.collate
    else:
        dataset = vessel_dataloader.vessel_dataset(json_path,
                                                    intra_patient=run_intra_registration,
                                                    type=data_type,
                                                    segma=config_data["segma"],
                                                    use_distance=config_data["use_distance"],
                                                    use_weight=config_data["use_weight"])
        collate_func = vessel_dataloader.collate
    
    generator = DataLoader(dataset, batch_size=config_data["param"]["batch_size"], 
                           shuffle=True, num_workers=4, collate_fn=collate_func, 
                           pin_memory=True)
    return generator

def setup_loss(config, loss_type, inshape):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if loss_type == 'ncc':
        loss = losses.NCC(inshape, device, config["use_weight"])
    elif loss_type == 'mse':
        loss = losses.MSE(use_weight=config["use_weight"])
    elif loss_type == 'mind':
        loss = losses.mind_ssc(device, use_weight=config["use_weight"])  
    elif loss_type == "l2":
        loss = losses.Grad3d(use_weight=config["use_weight"], penalty='l2')
    elif loss_type == "l1":
        loss = losses.Grad3d(use_weight=config["use_weight"], penalty='l1')      
    elif loss_type == "dice":
        loss = losses.Dice(inshape)
    elif loss_type == "boundary_loss":
        loss = losses.boundary_loss(inshape)
    elif loss_type == "dice_boundary_comb":
        loss = losses.dice_boundary_comb(inshape, epoch_start=config["start_epoch"], total_epoch=config["total_epoch"])
    elif loss_type == "liver_vessel":
        loss = losses.liver_vessel_comb(inshape, epoch_start=config["start_epoch"], total_epoch=config["total_epoch"], 
                                        liver_boundary_weight=config["liver_boundary_weight"], vessel_dice_weight=config["vessel_dice_weight"],
                                        vessel_unoverlap_weight=config["vessel_unoverlap_weight"])
    elif loss_type == "mi":
        loss = losses.MutualInformation(device, use_weight=config["use_weight"], num_bin=config["num_bin"])
    elif loss_type == "mine":
        loss = None
    else:
        assert(False)

    return loss

def setup_config(config_yaml, dataset_selected):
    with open(config_yaml, 'r') as f:
        config_custom = json.load(f)

    with open(os.path.join(os.path.dirname(__file__), "./config/dataset.json"), 'r') as f:
        default_config_data = json.load(f)

    with open(os.path.join(os.path.dirname(__file__), "./config/model.json"), 'r') as f:
        default_config_model = json.load(f)

    with open(os.path.join(os.path.dirname(__file__), "./config/loss.json"), "r") as f:
        default_config_loss = json.load(f)


    # dataset config
    data_type = dataset_selected
    data_param = default_config_data[dataset_selected]

    for type in default_config_model.keys():
        if "inshape" in default_config_model[type].keys():
            default_config_model[type]["inshape"] = default_config_data[dataset_selected]["image_shape"]
        if "in_chans" in default_config_model[type].keys():
            default_config_model[type]["in_chans"] = default_config_data[dataset_selected]["channel"] * 2


    #model config
    model_type = config_custom["model"]["type"]
    model_param = config_custom["model"]["param"]

    for type in default_config_loss.keys():
        if "int_downsize" in default_config_loss[type].keys():
            default_config_loss[type]["int_downsize"] = config_custom["model"]["int_downsize"]

    for type in default_config_loss.keys():
        if "total_epoch" in default_config_loss[type].keys():
            default_config_loss[type]["total_epoch"] = config_custom["train"]["epochs"]

    # loss config
    intensity_loss_type = config_custom["intensity_loss"]["type"]
    intensity_loss_param = config_custom["intensity_loss"]["param"]

    deformation_loss_type = config_custom["deformation_loss"]["type"]
    deformation_loss_param = config_custom["deformation_loss"]["param"]

    struct_loss_type = config_custom["struct_loss"]["type"]
    struct_loss_param = config_custom["struct_loss"]["param"]  

    config_custom["data"]["type"] = data_type
    config_custom["data"]["param"] = data_param
    config_custom["model"]["param"] = default_config_model[model_type] | model_param
    config_custom["intensity_loss"]["param"] = default_config_loss[intensity_loss_type] | intensity_loss_param
    config_custom["deformation_loss"]["param"] = default_config_loss[deformation_loss_type] | deformation_loss_param
    config_custom["struct_loss"]["param"]  = default_config_loss[struct_loss_type] | struct_loss_param

    return config_custom

def setup_model_dir(config_dict):
    now = dt.datetime.now()
    
    # data_type = config_dict["data"]["type"]
    # model_type = config_dict["model"]["type"]
    # intensity_loss_type = config_dict["intensity_loss"]["type"]
    # deformation_loss_type = config_dict["deformation_loss"]["type"]
    # struct_loss_type = config_dict["struct_loss"]["type"]

    model_sub_dir_dataset = config_dict["data"]["type"]
    model_sub_dir_exp = config_dict["experiment"]
    model_sub_dir_date = f"{now:%Y-%m-%d-%H-%M-%S}"
    model_dir = os.path.join(os.path.dirname(__file__), "./models", model_sub_dir_dataset, model_sub_dir_exp, model_sub_dir_date)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)  

    return model_dir

def setup_mine(path_to_saved_data, total_epoch, checkpoint=None):

    # Set up mine model, optimizer, schduler       
    model_mine = losses.MINE()
    optimizer_mine = torch.optim.Adam(model_mine.parameters(), lr=1e-4)
    scheduler_mine = torch.optim.lr_scheduler.MultiStepLR(optimizer_mine, milestones=[int(total_epoch/2)], gamma=10)

    print("Use MINE...")

    if checkpoint is None:
        
        print(f'init mine data from {path_to_saved_data}')

        path_to_saved_data = os.path.join(os.path.dirname(__file__), "../", path_to_saved_data)
        saved_data = torch.load(path_to_saved_data)
        model_mine.load_state_dict(saved_data)
    else:
        print("load mine data from checkpoint")
        model_mine.load_state_dict(checkpoint['mine_state'])
        optimizer_mine.load_state_dict(checkpoint['mine_optimizer_state'])
        scheduler_mine.load_state_dict(checkpoint['mine_scheduler_state'])

    return model_mine, optimizer_mine, scheduler_mine



if __name__ == "__main__":

    # Get dataset config 
    with open(os.path.join(os.path.dirname(__file__), "../SGMANet/config/dataset.json"), "r") as f:
        config_dataset_all = json.load(f)

    # Parser input argument
    parser = argparse.ArgumentParser(description = 'Run train')

    parser.add_argument('-c', '--config', dest='config_path',  help='path to the config json')
    parser.add_argument('-d', '--data', dest='dataset', choices=config_dataset_all.keys(),  help="dataset to use")
    parser.add_argument('-l', '--load', dest='load', default=None, help="path to load the model")

    args = parser.parse_args()


    if args.load is None:

        # read config from user setting
        config_custom = setup_config(args.config_path, args.dataset)

        #create dir to save model data
        model_dir = setup_model_dir(config_custom)

        # save config
        with open(os.path.join(model_dir, "config.json"), 'w') as f:
            json.dump(config_custom, f, indent = 4)
    else:
        model_dir = args.load

        config_yaml = os.path.join(model_dir, "config.json")
        with open(config_yaml, 'r') as f:
            config_custom = json.load(f)

    # setup train data
    train_data_loader = setup_data(config_custom["data"], 
                                   random_transform_type=config_custom["model"]["transform_type"],
                                   data_type="train",
                                   dataset_vessel= (config_custom["data"]["type"] in ["3dircab", "khp"]))
    eval_data_loader = setup_data(config_custom["data"], 
                                  random_transform_type=config_custom["model"]["transform_type"],
                                  data_type="valid",
                                  dataset_vessel= (config_custom["data"]["type"] in ["3dircab", "khp"]))

    #setup model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if config_custom["model"]["transform_type"] in ["Rigid", "Similarity", "Affine"]:
        model = networks.sgmanet_affine(config_custom["model"]["param"], 
                                config_custom["data"]["param"]["image_shape"],
                                config_custom["model"]["transform_type"]).to(device)
    else:
        model = networks.sgmanet(config_custom["model"]["param"], 
                                config_custom["data"]["param"]["image_shape"], 
                                config_custom["model"]["int_steps"],
                                config_custom["model"]["int_downsize"]).to(device)

    # Print model information
    input_size_single = (config_custom["data"]["param"]["batch_size"], config_custom["data"]["param"]["channel"], *config_custom["data"]["param"]["image_shape"])
    print(summary(model, input_size=[input_size_single, input_size_single, input_size_single, input_size_single]))

    # set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config_custom["train"]["lr"])
    milestones = config_custom["train"]["milestones"] if "milestones" in config_custom["train"] else []
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    if args.load is not None:
        checkpoint_dir = os.path.join(args.load, "checkpoint/")
        model_lists = natsorted(glob.glob(checkpoint_dir + '*'))
        print(f"load from: {model_lists[-1]}")

        checkpoint = torch.load(model_lists[-1], map_location=torch.device(device))

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint['epoch'] + 1
    else:
        epoch = 0

    for loss_type in ["intensity_loss", "deformation_loss", "struct_loss"]:
        if config_custom[loss_type]["type"] == "dice_boundary_comb":
            config_custom[loss_type]['param']['start_epoch'] = epoch

    # setup loss
    if config_custom["intensity_loss"]["weight"] != 0:
        intensity_loss = setup_loss(config_custom["intensity_loss"]["param"], config_custom["intensity_loss"]["type"], config_custom["data"]["param"]["image_shape"])
    else:
        intensity_loss = None
    
    if config_custom["deformation_loss"]["weight"] != 0:
        deformation_loss = setup_loss(config_custom["deformation_loss"]["param"], config_custom["deformation_loss"]["type"], config_custom["data"]["param"]["image_shape"])
    else:
        deformation_loss = None

    if config_custom["struct_loss"]["weight"] != 0:
        struct_loss = setup_loss(config_custom["struct_loss"]["param"], config_custom["struct_loss"]["type"], config_custom["data"]["param"]["image_shape"])
    else:
        struct_loss = None

    # Setup MINE model if it is used
    optimizer_mine = None
    scheduler_mine = None
    model_mine = None

    if config_custom["intensity_loss"]["type"] == "mine" and config_custom["intensity_loss"]["weight"] != 0:
        model_mine, optimizer_mine, scheduler_mine = setup_mine(config_custom["intensity_loss"]["param"]["path_to_saved_param"], 
                                                                config_custom["train"]["epochs"],
                                                                checkpoint if args.load is not None else None)

        model_mine.to(device)

        intensity_loss = model_mine.loss    

    # Train
    train.reg_train(config_custom["train"], 
                    train_data_loader, 
                    eval_data_loader, 
                    model, 
                    optimizer,
                    scheduler, 
                    intensity_loss, config_custom["intensity_loss"]["weight"],
                    deformation_loss,  config_custom["deformation_loss"]["weight"],
                    struct_loss, config_custom["struct_loss"]["weight"],
                    model_dir, 
                    device, 
                    epoch,
                    model_mine,
                    optimizer_mine,
                    scheduler_mine)
    