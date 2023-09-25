import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F
import torch.nn as nn
import torch.optim as optim
from advertorch.attacks import GradientSignAttack
from advertorch.context import ctx_noparamgrad_and_eval

import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import json
from vast import architectures, tools, losses

import pathlib





def command_line_options():
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='This is the main training script for all MNIST experiments. \
                    Where applicable roman letters are used as Known Unknowns. \
                    During training model with best performance on validation set in the no_of_epochs is used.'
    )

    parser.add_argument("--approach", "-a", default="SoftMax", choices=['SoftMax', 'BG', 'entropic', 'objectosphere'])
    parser.add_argument("--arch", default='LeNet', choices=['LeNet', 'LeNet_plus_plus'])
    parser.add_argument('--second_loss_weight', "-w", help='Loss weight for Objectosphere loss', type=float, default=0.0001)
    parser.add_argument('--Minimum_Knowns_Magnitude', "-m", help='Minimum Possible Magnitude for the Knowns', type=float,
                        default=50.)
    parser.add_argument("--solver", dest="solver", default='sgd',choices=['sgd','adam'])
    parser.add_argument("--lr", "-l", dest="lr", default=0.01, type=float)
    parser.add_argument('--batch_size', "-b", help='Batch_Size', action="store", dest="Batch_Size", type=int, default=128)
    parser.add_argument("--no_of_epochs", "-e", dest="no_of_epochs", type=int, default=50)
    parser.add_argument("--pertubation", "-p", dest="pertubation", type=str, default="FGSD", choices=['Noise', 'None', 'FGSD'])
    parser.add_argument("--pertub_eps", "-pe", dest="pertub_eps", type=int, default=0.1)
    parser.add_argument("--dataset_root", "-d", default ="/tmp", help="Select the directory where datasets are stored.")
    

    return parser.parse_args()


class Dataset(torch.utils.data.dataset.Dataset):
    """A split dataset for our experiments. It uses MNIST as known samples and EMNIST letters as unknowns.
    Particularly, the first 13 letters will be used as known unknowns (for training and validation), and the last 13 letters will serve as unknown unknowns (for testing only).
    The MNIST test set is used both in the validation and test split of this dataset.

    For the test set, you should consider to leave the parameters `include_unknown` and `BG` at their respective defaults -- this might make things easier.

    Parameters:

    dataset_root: Where to find/download the data to.

    which_set: Which split of the dataset to use; can be 'train' , 'test' or 'validation' (anything besides 'train' and 'test' will be the validation set)

    include_unknown: Include unknown samples at all (might not be required in some cases, such as training with plain softmax)

    BG: Set this to True when training softmax with background class. This way, unknown samples will get class label 10. If False (the default), unknown samples will get label -1.
    """
    def __init__(self, dataset_root, which_set="train", include_unknown=False, BG=False):
        self.mnist = torchvision.datasets.MNIST(
            root=dataset_root,
            train=which_set == "train",
            download=True,
            transform=transforms.ToTensor()
        )
        self.letters = torchvision.datasets.EMNIST(
            root=dataset_root,
            train=which_set == "train",
            download=True,
            split='letters',
            transform=transforms.ToTensor()
        )
        self.which_set = which_set
        targets = list() if not include_unknown else list(range(13)) if which_set != "test" else list(range(13,26))
        self.letter_indexes = [i for i, t in enumerate(self.letters.targets) if t in targets]
        self.BG = BG
        
    def __getitem__(self, index):
        if index < len(self.mnist):
            if(index%10000==0):
                print("using numbers")
            return self.mnist[index]     
        else:  
            if(index%10000==0):
                print("using letters")             
            return torch.transpose(self.letters[self.letter_indexes[index - len(self.mnist)]][0],1,2), 10 if self.BG else -1
            # return self.letters[self.letter_indexes[index - len(self.mnist)]][0], 10 if self.BG else -1

    def __len__(self):  
        return len(self.mnist) + len(self.letter_indexes)
        



def get_loss_functions(args):
    """Returns the loss function and the data for training and validation"""
    if args.approach == "SoftMax":
        return dict(
                    first_loss_func=nn.CrossEntropyLoss(reduction='none'),
                    second_loss_func=lambda arg1, arg2, arg3=None, arg4=None: torch.tensor(0.),
                    dir_name = "Softmax",
                    training_data = Dataset(args.dataset_root, include_unknown=False),
                    val_data = Dataset(args.dataset_root, which_set="val", include_unknown=False),
                )
    elif args.approach =="BG":
        return dict(
                    first_loss_func=nn.CrossEntropyLoss(reduction='none'),
                    second_loss_func=lambda arg1, arg2, arg3=None, arg4=None: torch.tensor(0.),
                    dir_name = "BGSoftmax",
                    training_data = Dataset(args.dataset_root, BG=True, include_unknown=False),
                    val_data = Dataset(args.dataset_root, which_set="val", BG=True, include_unknown=True)
                )
    
    elif args.approach == "entropic":
        return dict(
                    first_loss_func=losses.entropic_openset_loss(),
                    second_loss_func=lambda arg1, arg2, arg3=None, arg4=None: torch.tensor(0.),
                    dir_name = "Cross",
                    training_data=Dataset(args.dataset_root, include_unknown=False),
                    val_data = Dataset(args.dataset_root, which_set="val", include_unknown=True)
                )
    
    elif args.approach == "objectosphere":
        return dict(
                    first_loss_func=losses.entropic_openset_loss(),
                    second_loss_func=losses.objectoSphere_loss(args.Batch_Size,knownsMinimumMag=args.Minimum_Knowns_Magnitude),
                    dir_name = "ObjectoSphere",
                    training_data=Dataset(args.dataset_root),
                    val_data = Dataset(args.dataset_root, which_set="val")
                )


def train(args):

    epsilon = args.pertub_eps
    pertubation = args.pertubation


    torch.manual_seed(0)
    start_time = datetime.now()
    # get training data and loss function(s)
    first_loss_func,second_loss_func,dir_name,training_data,validation_data = list(zip(*get_loss_functions(args).items()))[-1]

    dir_name = dir_name + str(args.no_of_epochs) + pertubation + str(epsilon)
    results_dir = pathlib.Path(f"{args.arch}/{dir_name}")

    save_dir = f"{results_dir}/{dir_name}.model"

    results_dir.mkdir(parents=True, exist_ok=True)

    # instantiate network and data loader
    net = architectures.__dict__[args.arch](use_BG=args.approach == "BG")
    net = tools.device(net)
    train_data_loader = torch.utils.data.DataLoader(
        training_data,
        batch_size=args.Batch_Size,
        shuffle=True,
        num_workers=5,
        pin_memory=True
    )
    val_data_loader = torch.utils.data.DataLoader(
        validation_data,
        batch_size=args.Batch_Size,
        pin_memory=True
    )

    if args.solver == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=args.lr)
    elif args.solver == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)

    logs_dir = results_dir/'Logs'
    writer = SummaryWriter(logs_dir)
    model_update = 0
    last_epoch = 0
    update_epoch = []
    # train network
    for epoch in range(1, args.no_of_epochs + 1, 1):  # loop over the dataset multiple times
        loss_history = []
        train_accuracy = torch.zeros(2, dtype=int)
        train_magnitude = torch.zeros(2, dtype=float)
        train_confidence = torch.zeros(2, dtype=float)
        net.train()
        for x, y in train_data_loader:
            x = tools.device(x)
            y = tools.device(y)
            
            optimizer.zero_grad()
            logits, features = net(x)    

            if epoch > 0 and (pertubation =="FGSM" or pertubation == "Noise"):
                temp_max = torch.max(torch.nn.functional.softmax(logits , dim=1), axis=1)
                samples_to_pertub = (temp_max.indices == y) & (temp_max.values >= 0.9)
                pertb_x = x[samples_to_pertub]
                pertb_y = y[samples_to_pertub]
                if len(pertb_x) > 0:
                    if pertubation == "FGSM":
                        adversary = GradientSignAttack(net, first_loss_func,  eps=epsilon)
                        with ctx_noparamgrad_and_eval(net):
                            adv_untargeted = adversary.perturb(pertb_x, pertb_y)                    
                    elif pertubation == "Noise":
                        adv_untargeted = torch.clamp((pertb_x+tools.device(epsilon*(torch.rand(28,28)-0.5))),0,1)
                    else:
                        adv_untargeted = torch.Tensor()
                        print("No Perturbation")
                    if args.approach == "SoftMax":
                        pertb_y = pertb_y                     
                    elif args.approach == "BG":
                        pertb_y = torch.ones_like(pertb_y)*10
                    elif args.approach == "entropic":
                        pertb_y = torch.ones_like(pertb_y)*-1
                                          
                    pertb_logits, pertb_features = net(adv_untargeted)
      
                    x = torch.cat((x, pertb_x),0)
                    y = torch.cat((y, pertb_y),0)
                    logits = torch.cat((logits, pertb_logits),0)
                    features = torch.cat((features, pertb_features),0)
            # metrics on training set
            train_accuracy += losses.accuracy(logits, y)
            train_confidence += losses.confidence(logits, y)
            if args.approach not in ("SoftMax", "BG"):
                train_magnitude += losses.sphere(features, y, args.Minimum_Knowns_Magnitude if args.approach in args.approach == "objectosphere" else None)


            # first loss is always computed, second loss only for some loss functions
            loss = first_loss_func(logits, y) + args.second_loss_weight * second_loss_func(features, y)
            
            

            loss_history.extend(loss.tolist())
            loss.mean().backward()
            optimizer.step()
            

        # metrics on validation set
        with torch.no_grad():
            val_loss = torch.zeros(2, dtype=float)
            val_accuracy = torch.zeros(2, dtype=int)
            val_magnitude = torch.zeros(2, dtype=float)
            val_confidence = torch.zeros(2, dtype=float)
            net.eval()
            for x,y in val_data_loader:
                x = tools.device(x)
                y = tools.device(y)
                outputs = net(x)

                loss = first_loss_func(outputs[0], y) + args.second_loss_weight * second_loss_func(outputs[1], y)
                val_loss += torch.tensor((torch.sum(loss), len(loss)))
                val_accuracy += losses.accuracy(outputs[0], y)
                val_confidence += losses.confidence(outputs[0], y)
                if args.approach not in ("SoftMax", "BG"):
                    val_magnitude += losses.sphere(outputs[1], y, args.Minimum_Knowns_Magnitude if args.approach == "objectosphere" else None)

        # log statistics
        epoch_running_loss = torch.mean(torch.tensor(loss_history))
        writer.add_scalar('Loss/train', epoch_running_loss, epoch)
        writer.add_scalar('Loss/val', val_loss[0] / val_loss[1], epoch)
        writer.add_scalar('Acc/train', float(train_accuracy[0]) / float(train_accuracy[1]), epoch)
        writer.add_scalar('Acc/val', float(val_accuracy[0]) / float(val_accuracy[1]), epoch)
        writer.add_scalar('Conf/train', float(train_confidence[0]) / float(train_confidence[1]), epoch)
        writer.add_scalar('Conf/val', float(val_confidence[0]) / float(val_confidence[1]), epoch)
        writer.add_scalar('Mag/train', train_magnitude[0] / train_magnitude[1] if train_magnitude[1] else 0, epoch)
        writer.add_scalar('Mag/val', val_magnitude[0] / val_magnitude[1], epoch)

        # save network based on confidence metric of validation set
        save_status = "NO"
        if epoch <= 5:
            prev_confidence = None
        if prev_confidence is None or (val_confidence[0] > prev_confidence):
            torch.save(net.state_dict(), save_dir)
            prev_confidence = val_confidence[0]
            model_update+=1
            save_status = "YES"
            last_epoch = epoch
            update_epoch.append(epoch)
        # print some statistics
        print(f"Epoch {epoch} "
              f"train loss {epoch_running_loss:.10f} "
              f"accuracy {float(train_accuracy[0]) / float(train_accuracy[1]):.5f} "
              f"confidence {train_confidence[0] / train_confidence[1]:.5f} "
              f"magnitude {train_magnitude[0] / train_magnitude[1] if train_magnitude[1] else -1:.5f} -- "
              f"val loss {float(val_loss[0]) / float(val_loss[1]):.10f} "
              f"accuracy {float(val_accuracy[0]) / float(val_accuracy[1]):.5f} "
              f"confidence {val_confidence[0] / val_confidence[1]:.5f} "
              f"magnitude {val_magnitude[0] / val_magnitude[1] if val_magnitude[1] else -1:.5f} -- "
              f"Saving Model {save_status}")
    end_time = datetime.now()
    logdata={
        "epoch" : args.no_of_epochs,
        "name" : args.approach,
        "arch" :args.arch,
        "batch" : args.Batch_Size,
        "solver": args.solver,
        "lr"    : args.lr,
        "updates": model_update,
        "stime" : start_time,
        "etime" : end_time,
        "eps" : epsilon,
        "train": "letters",
        "last_epoch" : last_epoch,
        "update_epoch" : update_epoch

    }
    with open(logs_dir/"logdata.json", "w") as outfile:
        json.dump(logdata, outfile, default=str)


if __name__ == "__main__":

    args = command_line_options()
    if torch.cuda.is_available():
        print("cuda on")
        tools.set_device_gpu()
    else:
        print("Running in CPU mode, training might be slow")
        tools.set_device_cpu()
    
    temp_eps = 0.01
    args.no_of_epochs = 11
    args.approach = "entropic"
    args.pertubation = "FGSM"
    args.pertub_eps = temp_eps
    train(args)
        
      
    
