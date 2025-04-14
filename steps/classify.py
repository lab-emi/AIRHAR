import models as model
from project import Project
from utils.util import count_net_params
import os
import torch



def main(proj: Project):
    ###########################################################################################################
    # Initialization
    ###########################################################################################################
    # Set Accelerator Device
    proj.set_device()

    # Build Dataloaders
    train_loader, test_loader = proj.build_dataloaders()

    ###########################################################################################################
    # Network Settings
    ###########################################################################################################
    # Instantiate Model


    # Load Pretrained PA Model
    # path_pa_model = os.path.join('save', proj.dataset_name, 'train_pa', pa_model_id + '.pt')
    # net_pa.load_state_dict(torch.load(path_pa_model))

    # Instantiate DPD Model
    net_classifier = model.CoreModel(
                          hidden_size=proj.Classification_hidden_size,
                          num_layers=proj.Classification_num_layers,
                          backbone_type=proj.Classification_backbone,
                          dim=proj.dim,
                          dt_rank=proj.dt_rank,
                          d_state=proj.d_state,
                          image_height=proj.image_height,
                          image_width=proj.frame_length,
                          num_classes=proj.num_classes,
                          channels=proj.channels,
                          dropout=proj.dropout,
                          channel_confusion_layer=proj.channel_confusion_layer,
                          channel_confusion_out_channels=proj.channel_confusion_out_channels,
                          time_downsample_factor=proj.time_downsample_factor,
                          optional_avg_pool=proj.optional_avg_pool)
    net_classifier_params = count_net_params(net_classifier)
    print("::: Number of DPD Model Parameters: ", net_classifier_params)
    classifier_model_id = proj.gen_classifier_model_id(net_classifier_params)
    # classifier_model_id = os.path.join('save', proj.dataset_name, 'classify', classifier_model_id + '.pt')
    # net_classifier.load_state_dict(torch.load(classifier_model_id))
    # for param in net_classifier.parameters():
    #     param.requires_grad = False

    net_cas = net_classifier



    # Move the network to the proper device
    net_cas = net_cas.to(proj.device)

    ###########################################################################################################
    # Logger, Loss and Optimizer Settings
    ###########################################################################################################
    # Build Logger
    proj.build_logger(model_id=classifier_model_id)

    # Select Loss function
    criterion = proj.build_criterion()

    # Create Optimizer and Learning Rate Scheduler
    optimizer, lr_scheduler = proj.build_optimizer(net=net_cas)

    ###########################################################################################################
    # Training
    ###########################################################################################################
    proj.train(net=net_cas,
               criterion=criterion,
               optimizer=optimizer,
               lr_scheduler=lr_scheduler,
               train_loader=train_loader,
               val_loader=test_loader,
               test_loader=test_loader,
               best_model_metric='NMSE')