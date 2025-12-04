#!/usr/bin/env python

import os
import time
import numpy as np
import  torch
import visual
import eval

import sys
import glob
from natsort import natsorted

from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt

from tqdm import tqdm
import json

class Logger(object):
    def __init__(self, save_dir):
        self.terminal = sys.stdout
        self.log = open(os.path.join(save_dir,"logfile.log"), "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def save_checkpoint(dir, checkpoint_dict, max_model_num=8):
    path = os.path.join(dir, f'{checkpoint_dict["epoch"]:04d}_chechpoint.pth.tar')

    torch.save(checkpoint_dict, path)
    
    dir = os.path.join(dir, "*")

    model_lists = natsorted(glob.glob(dir))
    while len(model_lists) > max_model_num:
        os.remove(model_lists[0])
        model_lists = natsorted(glob.glob(dir + '*'))

def comput_fig(img):
    img = img.detach().cpu().numpy()[0, 0, [10, 20, 30, 40, 50, 60]]
    fig = plt.figure(figsize=(12,12), dpi=180)
    for i in range(img.shape[0]):
        plt.subplot(2, 3, i + 1)
        plt.axis('off')
        plt.imshow(img[i, :, :], cmap='gray')
    fig.subplots_adjust(wspace=0, hspace=0)
    return fig


# def mk_grid_img(grid_step, line_thickness=1, grid_sz=(160, 192, 224)):
#     grid_img = np.zeros(grid_sz)
#     for j in range(0, grid_img.shape[1], grid_step):
#         grid_img[:, j+line_thickness-1, :] = 1
#     for i in range(0, grid_img.shape[2], grid_step):
#         grid_img[:, :, i+line_thickness-1] = 1
#     grid_img = grid_img[None, None, ...]
#     grid_img = torch.from_numpy(grid_img).cuda()
#     return grid_img

class best_recoder():
    def __init__(self, path_to_best, max_save_records =5):
        self.max_save_records = max_save_records

        self.path_to_best = path_to_best
        self.best_json_path = os.path.join(path_to_best, 'best.json')

        if os.path.exists(self.best_json_path):
            with open(self.best_json_path, 'r') as file:
                record_dict = json.load(file)

            # New format
            if "best_list" in record_dict:
                self.record = record_dict["best_list"]
                self.threshold = record_dict["threshold"]
            else:
                self.record = [{"dsc": record_dict["best_dsc"], 
                                "jacob": record_dict["best_jacob"], 
                                "score": record_dict["best_dsc"] - record_dict["best_jacob"], 
                                "epoch": record_dict["epoch"]}]
                self.threshold = 0

            print(f"load saved best from: {self.best_json_path}, content: {self.record}")
        else:
            self.record = []
            self.threshold = 0

    def update_record(self, new_node):
        self.record.append(new_node)

        if len(self.record) > self.max_save_records:

            self.record.sort(key=lambda n: n["score"])

            delete_node = self.record.pop(0)

            file_name = f'best_{delete_node["epoch"]:04d}_{int(delete_node["score"]*10000):04d}.pth'
            os.remove(os.path.join(self.path_to_best, file_name))

            self.threshold = self.record[0]["score"]

    def save_best(self, dsc, jacob, model, epoch):

        score = dsc - jacob

        if score > self.threshold:
            node = {"dsc": dsc, "jacob": jacob, "score": score, "epoch": epoch}

            self.update_record(node)

            with open(self.best_json_path, 'w') as file:
                json.dump({"best_list" : self.record, "threshold" : self.threshold}, file, indent = 4)

            file_name = f'best_{node["epoch"]:04d}_{int(node["score"]*10000):04d}.pth'
            torch.save(model.state_dict(), os.path.join(self.path_to_best, file_name))

            print("best update!")



def reg_train(config, data_train, data_eval, model, optimizer, scheduler, intensity_loss, intensity_weight, 
              deformation_loss, deformation_weight,  
              struct_loss, struct_weight,
              log_dir, device, init_epoch, 
              model_mine, optimizer_mine, scheduler_mine):

    # enabling cudnn determinism appears to speed up training by a lot
    torch.backends.cudnn.deterministic = True

    # torch.autograd.set_detect_anomaly(True)

    # prepare the model for training and send to device
    model.to(device)

    # Create the folder to variouse information
    path_to_checkpoint = os.path.join(log_dir, "checkpoint")
    if not os.path.exists(path_to_checkpoint):
        os.makedirs(path_to_checkpoint)  

    path_to_best = os.path.join(log_dir, "best")
    if not os.path.exists(path_to_best):
        os.makedirs(path_to_best)  

    path_to_log = os.path.join(log_dir, "log")
    if not os.path.exists(path_to_log):
        os.makedirs(path_to_log)  

    #Use this to update the best model
    best_manager = best_recoder(path_to_best)

    # Set up logger
    sys.stdout = Logger(path_to_log)
    writer = SummaryWriter(path_to_log)

    # training loops
    for epoch in range(init_epoch, config["epochs"]):

        # save model checkpoint
        if epoch % 10 == 0 and epoch != 0:
            checkpoint_dict = {
                                'epoch': epoch,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'scheduler_state_dict' : scheduler.state_dict()
                            }

            if model_mine is not None:
                 checkpoint_dict["mine_state"] = model_mine.state_dict()
                 checkpoint_dict["mine_optimizer_state"] = optimizer_mine.state_dict()
                 checkpoint_dict["mine_scheduler_state"] = scheduler_mine.state_dict()             
            

            save_checkpoint(path_to_checkpoint, checkpoint_dict)

        epoch_loss = []
        epoch_total_loss = []

        epoch_eval_dsc = []
        epoch_eval_jacob = []
        # epoch_eval_jacob_pytorch = []

        # epoch_time_info = []

        model.train()
        if model_mine is not None: model_mine.train()

        # Run train
        for step, (moving_image, fix_image, moving_label, fix_label, fix_weight, fix_distance) in enumerate(tqdm(data_train, total=min(config["train_steps_per_epoch"], len(data_train)))):

            if step >= config["train_steps_per_epoch"]:
                break

            # # Recode the start time
            # start_time = time.perf_counter()

            # Load image to device
            moving_image = moving_image.to(device)
            fix_image = fix_image.to(device)

            # run inputs through the model to produce a warped image and flow field
            result_dict = model(moving_image, fix_image, moving_label, fix_label, need_moved_label=(struct_weight != 0), need_deformation_filed=False)

            # # Recode this time point
            # model_end_time = time.perf_counter()

            # Loss list
            loss_list = []
            loss = torch.zeros(1).to(device)

            # intensity loss
            if intensity_weight != 0 and ("moved_image" in result_dict):

                if model_mine is None:
                    fixed_image_single_channel = fix_image[:, (0,), ...]
                    intensity_loss_value = intensity_loss(fixed_image_single_channel, result_dict["moved_image"], fix_weight)
                else:
                    fixed_image_single_channel = fix_image[:, (0,), ...]
                    mi_value = model_mine(fixed_image_single_channel, result_dict["moved_image"])
                    intensity_loss_value = -torch.mean(mi_value)

                loss += intensity_weight * intensity_loss_value
                loss_list.append(intensity_loss_value.item())

            # #Recode the time point
            # intensity_end_time = time.perf_counter()

            # deformation loss
            if deformation_weight != 0 and ("displacement_filed" in result_dict):
                deformation_loss_value = deformation_loss(None, result_dict["displacement_filed"], fix_weight)
                loss += deformation_weight * deformation_loss_value
                loss_list.append(deformation_loss_value.item())

            # #Recode the time point
            # deformation_end_time = time.perf_counter()

            # struct loss
            if struct_weight != 0 and ("moved_label" in result_dict):
                struct_loss_value = struct_loss(fix_label, result_dict["moved_label"], fix_distance)
                loss += struct_weight * struct_loss_value
                loss_list.append(struct_loss_value.item())

            # #Recode the time point
            # struct_end_time = time.perf_counter()

            epoch_loss.append(loss_list)
            epoch_total_loss.append(loss.item())

            # backpropagate and optimize
            optimizer.zero_grad()
            if optimizer_mine is not None: optimizer_mine.zero_grad()
            loss.backward()
            optimizer.step()
            if optimizer_mine is not None: optimizer_mine.step()

            # # Recode the end time
            # end_time = time.perf_counter()

            # # print time info
            # epoch_time_info.append((end_time - start_time, model_end_time - start_time, intensity_end_time - model_end_time, deformation_end_time - intensity_end_time, struct_end_time - deformation_end_time))

        #Run eval
        with torch.no_grad():
            model.eval()
            for step, (moving_image, fix_image, moving_label, fix_label, _, _) in enumerate(tqdm(data_eval, total=min(config["eval_steps_per_epoch"], len(data_eval)))):
                
                if step >= config["eval_steps_per_epoch"]:
                    break
                
                # Load image to device
                moving_image = moving_image.to(device)
                fix_image = fix_image.to(device)

                # run inputs through the model to produce a warped image and flow field
                eval_result_dict = model(moving_image, fix_image, moving_label, fix_label, need_moved_label=True, need_deformation_filed=True)

                # calculate metric

                #Dsc
                if "moved_label" in eval_result_dict:
                    eval_dsc = eval.dsc_one_hot(fix_label, eval_result_dict["moved_label"])
                    eval_dsc = torch.mean(eval_dsc).item()
                else:
                    eval_dsc = 0

                #jacob
                if "deformation_filed" in eval_result_dict:
                    jacob = eval.jacobian_determinant(eval_result_dict["deformation_filed"])
                else:
                    jacob = 0    

                epoch_eval_dsc.append(eval_dsc)
                epoch_eval_jacob.append(jacob)

        scheduler.step()
        if scheduler_mine is not None: scheduler_mine.step()

        avg_item_losses = np.mean(epoch_loss, axis=0)
        avg_total_loss = np.mean(epoch_total_loss)
        
        # print epoch info
        
        #Print train result
        epoch_info = 'Epoch %d/%d' % (epoch + 1, config["epochs"])
        losses_info = ', '.join(['%.4e' % f for f in avg_item_losses])
        loss_info = 'loss: %.4e  (%s)' % (avg_total_loss, losses_info)
        lr_info = 'lr: %.4e' % scheduler.get_last_lr()[0]
        print(' - '.join((epoch_info, loss_info, lr_info)), flush=True)

        # Record train result
        writer.add_scalar('train/total_loss', avg_total_loss, epoch)
        writer.add_scalar('train/intensity_loss', avg_item_losses[0], epoch)
        if len(avg_item_losses) > 1 : writer.add_scalar('train/deformation_loss', avg_item_losses[1], epoch) 
        if len(avg_item_losses) > 2 : writer.add_scalar('train/struct_loss', avg_item_losses[2], epoch)

        # Print evaluation result
        avg_dsc = np.mean(epoch_eval_dsc)
        # avg_jacob = np.mean(epoch_eval_jacob)
        avg_jacob = np.mean(epoch_eval_jacob)
        print("eval:")
        print(f"avg_dsc: {avg_dsc: .4f}")
        # print(f"avg_jacob: {avg_jacob: .4f}")
        print(f"avg_jacob_pytorch: {avg_jacob: .4f}")

        # # Average and print the time info
        # epoch_time_info = np.array(epoch_time_info)
        # average_time_info = np.mean(epoch_time_info, axis=0)
        # print(f"total_time: {average_time_info[0]}, \
        #       model_time: {average_time_info[1]}, \
        #       intensity_time: {average_time_info[2]}, \
        #       deformation_time: {average_time_info[3]}, \
        #       struct_time: {average_time_info[4]}")

        # Record evaluation result
        writer.add_scalar('eval/dsc', avg_dsc, epoch)
        writer.add_scalar('eval/jacob', avg_jacob, epoch)

        #Record figure
        plt.switch_backend('agg')

        fixed_fig = comput_fig(fix_image)
        writer.add_figure('fix/image', fixed_fig, epoch)
        plt.close(fixed_fig)

        if "moved_image" in eval_result_dict:
            moved_fig = comput_fig(eval_result_dict["moved_image"])
            writer.add_figure('moved/image', moved_fig, epoch)
            plt.close(moved_fig)

        fixed_label_fig = comput_fig(fix_label)
        writer.add_figure('fix/label', fixed_label_fig, epoch)
        plt.close(fixed_label_fig)

        if "moved_label" in eval_result_dict:
            moved_label_fig = comput_fig(eval_result_dict["moved_label"])
            writer.add_figure('moved/label', moved_label_fig, epoch)
            plt.close(moved_label_fig)

        # Save best
        best_manager.save_best(avg_dsc, avg_jacob, model, epoch)

    writer.close()



