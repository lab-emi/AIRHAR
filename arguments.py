__author__ = "Yizhuo Wu, Chang Gao"
__license__ = "Apache-2.0 License"
__email__ = "yizhuo.wu@tudelft.nl, chang.gao@tudelft.nl"

import argparse


def get_arguments():
    # Process Arguments
    parser = argparse.ArgumentParser(description='Train a GRU network.')
    # Data Prepare : bin to numpy + train_val_test split
    parser.add_argument('--dataset_name', default='Alabma', help='Micro doppler signature data root')
    parser.add_argument('--num_classes', default=12, type=int, help='Number of classes.')

    # Micro doppler signature settings
    parser.add_argument('--channels', default=1, type=int, help='Number of input channels.')
    parser.add_argument('--image_height', default=224, type=int, help='Micro doppler signature height (doppler).')
    parser.add_argument('--frame_length', default=224, type=int, help='Image segments width. (time)')
    parser.add_argument('--stride', default=1, type=int, help='Stride.')
    parser.add_argument('--continuous', default=False, type=bool, help='Whether the data is continuous.')
    # Dataset & Log
    parser.add_argument('--log_precision', default=8, type=int, help='Number of decimals in the log files.')

    # Training Process
    parser.add_argument('--step', default='classify', choices=['classify'], help='Step to run.')
    parser.add_argument('--eval_val', default=1, type=int, help='Whether evaluate val set during training.')
    parser.add_argument('--eval_test', default=1, type=int, help='Whether evaluate test set during training.')
    parser.add_argument('--accelerator', default='cuda', choices=["cpu", "cuda", "mps"], help='Accelerator types.')
    parser.add_argument('--devices', default=0, type=int, help='Which accelerator to train on.')
    parser.add_argument('--re_level', default='soft', choices=['soft', 'hard'], help='Level of reproducibility.')

    # General Hyperparameters
    parser.add_argument('--seed', default=0, type=int, help='Global random number seed.')
    parser.add_argument('--loss_type', default='CrossEntropy', choices=['l1', 'l2', 'CrossEntropy', 'WeightedCrossEntropy'], help='Type of loss function.')
    parser.add_argument('--opt_type', default='adamw', choices=['sgd', 'adam', 'adamw', 'adabound', 'rmsprop'], help='Type of optimizer.')
    parser.add_argument('--batch_size', default=16, type=int, help='Batch size for training.')
    parser.add_argument('--batch_size_eval', default=137, type=int, help='Batch size for evaluation.')
    parser.add_argument('--n_epochs', default=100, type=int, help='Number of epochs to train for.')
    parser.add_argument('--lr_schedule', default=1, type=int, help='Whether enable learning rate scheduling')
    parser.add_argument('--lr', default=5e-3, type=float, help='Learning rate')
    parser.add_argument('--lr_end', default=1e-6, type=float, help='Learning rate')
    parser.add_argument('--decay_factor', default=0.5, type=float, help='Learning rate')
    parser.add_argument('--patience', default=15, type=float, help='Learning rate')
    parser.add_argument('--grad_clip_val', default=200, type=float, help='Gradient clipping.')

    # Classification Model Settings
    parser.add_argument('--Classification_backbone', default='radmamba',
                        choices=['vgg','resnet','bilstm', 'cnnlstm', 'cnngru', 'radmamba'], help='DPD model Recurrent layer type')
    parser.add_argument('--Classification_hidden_size', default=64, type=int, help='Hidden size of DPD backbone.')
    parser.add_argument('--Classification_num_layers', default=1, type=int, help='Number of layers of the DPD backbone.')


    # RadMamba Settings
    parser.add_argument('--channel_confusion_layer', default=1, type=int, help='Number of channel confusion layers.')
    parser.add_argument('--channel_confusion_out_channels', default=3, type=int, help='Kernel size of the channel confusion layer.')
    parser.add_argument('--time_downsample_factor', default=2, type=int, help='Downsample factor of the channel confusion layer.')
    parser.add_argument('--optional_avg_pool', default=False, type=bool, help='Whether to use optional avg pool.')
    parser.add_argument('--dim', default=80, type=int, help='Dimension of the transformer model.')
    parser.add_argument('--dt_rank', default=0, type=int, help='Rank of the dynamic routing matrix.')
    parser.add_argument('--d_state', default=4, type=int, help='Dimension of the state vector.')
    parser.add_argument('--dropout', default=0, type=float, help='Dropout rate.')

    return parser.parse_args()