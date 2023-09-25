
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pathlib


from advertorch.attacks import LinfPGDAttack
from advertorch.attacks import L1PGDAttack

from advertorch.attacks import LinfPGDAttack
from advertorch.attacks import GradientSignAttack

import pandas as pd
from vast import architectures, losses, tools
import matplotlib

from matplotlib.backends.backend_pdf import PdfPages
import torchvision

import numpy as np
from matplotlib import pyplot


def accuracy(prediction, target):
    """Computes the classification accuracy of the classifier based on known samples only.
    Any target that does not belong to a certain class (target is -1) is disregarded.

    Parameters:

      prediction: the output of the network, can be logits or softmax scores

      target: the vector of true classes; can be -1 for unknown samples

    Returns a tensor with two entries:

      correct: The number of correctly classified samples

      total: The total number of considered samples
    """

    with torch.no_grad():
        known = target >= 0
        
        total = torch.sum(known, dtype=int)
        if total:
            correct = torch.sum(
                torch.max(prediction[known], axis=1).indices == target[known], dtype=int
            )
            t = torch.max(prediction[known], axis=1).indices == target[known]
            t_temp = t.nonzero(as_tuple=True)
            
            

        else:
            t = torch.max(prediction[known], axis=1).indices == target[known]
            t_temp = t.nonzero(as_tuple=True)
            correct = 0
            
    return torch.tensor((correct, total)), t_temp[0]


def confidence_max_nontarget(logits, target, negative_offset=0.1, dir_name = "BG"):
    """Measures the softmax confidence of the correct class for known samples,
    and 1 + negative_offset - max(confidence) for unknown samples.

    Parameters:

      logits: the output of the network, must be logits

      target: the vector of true classes; can be -1 for unknown samples

    Returns a tensor with two entries:

      confidence: the sum of the confidence values for the samples

      total: The total number of considered samples
    """

    with torch.no_grad():
        known = target >= 0
        pred = torch.nn.functional.softmax(logits, dim=1)
        
        ## Set the softmax value of the true label to -Inf
        pred[range(len(pred)), target] = float("-Inf")
        #### UROS
        if dir_name.startswith("BG"):
            ## Set the softmax value of the unkown label to -Inf for BGSoftmax
            unkown_target = torch.ones_like(target)*10
            pred[range(len(pred)), unkown_target] = float("-Inf")  
        confidence = 0.0
        ## standard Vast confidence
        if torch.sum(known):   
            confidence += torch.sum(torch.max(pred, dim=1).values)
            std = torch.std(torch.max(pred, dim=1).values, dim=0)    
        if torch.sum(~known):
            confidence += torch.sum(
                1.0 + negative_offset - torch.max(pred[~known], dim=1)[0]
            )
    return torch.tensor((confidence, len(logits), std))



def test(net, val_data_loader, dir_name, arch, loss, attack_name, conf_result, eps_max=0.3, eps_iter=0.02,  conf_result2=[[],[]], std_conf = []):
    with torch.no_grad():
        torch.set_printoptions(sci_mode=False)
        ## Directory to save results
        dir_name = dir_name
        arch = arch
        results_dir = pathlib.Path(f"{arch}/{dir_name}")
        results_dir.mkdir(parents=True, exist_ok=True)
        ## Empty Tensors for Evaluation
        val_accuracy = torch.zeros(2, dtype=int)
        val_confidence = torch.zeros(3, dtype=float)
        val_confidence2 = torch.zeros(2, dtype=float)
        net.eval()
        ### batch_idx was not used, as the whole dataset could be used in one batch
        for batch_idx, (x,y) in enumerate(val_data_loader):
            
            ## Put samples on cpu or gpu and run them through the network
            x = tools.device(x)
            y = tools.device(y)
            outputs = net(x)
            
            ## Calculate the accuracy of the sample prediction
            temp = accuracy(outputs[0], y)
            val_accuracy += temp[0]
            ## t are the samples that have been predicted correctly
            t = temp[1]

            ## Calculate the confidence of the samples
            
            ## samples only the correctly predicted samples
            true_x = x[t]
            true_y = y[t]

            ## confirm correctly predicted samples once more (not needed)
            confirm_val_accuracy = torch.zeros(2, dtype=int)
            confirm_outputs = net(true_x)
            confirm_temp = accuracy(confirm_outputs[0], true_y)
            confirm_val_accuracy += confirm_temp[0]
            confirm_t = confirm_temp[1]
            val_confidence += confidence_max_nontarget(confirm_outputs[0], true_y, dir_name=dir_name)
            val_confidence2 += losses.confidence(confirm_outputs[0], true_y)
            print(t.size())
            print(confirm_t.size())
            
    eps = 0.00
    print(val_confidence, "This is confidence")
    print(val_accuracy, "This is accuracy")
    print(confirm_val_accuracy, "This is confirmed accuracy")
    conf_result[0].append(float(val_confidence[0]) / float(val_confidence[1]))
    conf_result[1].append(0)
    conf_result2[0].append(float(val_confidence2[0]) / float(val_confidence2[1]))
    conf_result2[1].append(0)
    eps = 0.01
    #### Choose right Loss function
    if dir_name.startswith("Cross"):
        loss = losses.entropic_openset_loss()
    else:
        loss = nn.CrossEntropyLoss()
    ## Iterate through different epsilons
    while eps<= eps_max:
        ### Choose which attack
        if attack_name =="FGSM":
            adversary = GradientSignAttack(net, loss, eps=eps)
            adv_untargeted = adversary.perturb(true_x, true_y)    
        elif attack_name.startswith("FGSM_T"):
            adversary = GradientSignAttack(net, loss, eps=eps, targeted=True)
            T = int(attack_name[-1])
            print("fgsd target is ", T)    
            adv_untargeted = adversary.perturb(true_x, (true_y+1)%10)     
        elif attack_name == "L1GPD":
            adversary = L1PGDAttack(net, loss, eps=eps)
            adv_untargeted = adversary.perturb(true_x, true_y)
        elif attack_name.startswith("L1GPD_T"):
            adversary = L1PGDAttack(net, loss, eps=eps)
            T = int(attack_name[-1])
            print("target is ", T)
            adv_untargeted = adversary.perturb(true_x, torch.ones_like(true_y)*T)  
        elif attack_name== "LinfGPD":
            adversary = LinfPGDAttack(net, loss, eps=eps,)
            adv_untargeted = adversary.perturb(true_x, true_y) 

        elif attack_name.startswith("LinfGPD_T"):
            adversary = LinfPGDAttack(net, loss, eps=eps, targeted=True)
            T = int(attack_name[-1])
            print("Linf target is ", T)
            adv_untargeted = adversary.perturb(true_x, (true_y+1)%10)  
        elif attack_name.startswith("Noise"):
            torch.manual_seed(7)
            adv_untargeted = torch.clamp((true_x+tools.device(eps*(torch.rand(28,28)-0.5))),0,1)
            print(" ############## noise")
        else:
            torch.manual_seed(7)
            adv_untargeted = torch.clamp((true_x+tools.device(eps*(torch.rand(28,28)-0.5))),0,1)  
            print(" ############## noise misstype")

        
        with torch.no_grad():
            ## Evaluate
            adv_outputs = net(adv_untargeted)
            adv_val_confidence = torch.zeros(3, dtype=float)
            adv_val_confidence += confidence_max_nontarget(adv_outputs[0], true_y, dir_name=dir_name)
            adv_val_confidence2 = torch.zeros(2, dtype=float)
            adv_val_confidence2 += losses.confidence(adv_outputs[0], true_y)
            print(adv_val_confidence, "This is adv_confidence at eps: ", eps)
              
            conf_result[0].append(float(adv_val_confidence[0]) / float(adv_val_confidence[1])) 
            conf_result[1].append(eps)
            conf_result2[0].append(float(adv_val_confidence2[0]) / float(adv_val_confidence2[1])) 
            conf_result2[1].append(eps)
            
            eps =  round(eps + eps_iter, 2)
    plt.close("all")



def load_network(which, arch):
  net = architectures.LeNet(use_BG=which.startswith("BG"))
  net.load_state_dict(torch.load(f"{arch}/{which}/{which}.model"))
  tools.device(net)
  return net



# Set device to cpu or gpu
if torch.cuda.is_available():
    tools.set_device_gpu()
    print("Running in GPU mode, evaluation might be fast :O")
else:
    print("Running in CPU mode, evaluation might be slow")
    tools.set_device_cpu()


DS_ROOT = "/tmp"

val_set = torchvision.datasets.MNIST(
            root=DS_ROOT,
            train=False,
            download=True,
            transform=torchvision.transforms.ToTensor()
        )

class Dataset(torch.utils.data.dataset.Dataset):
    def __init__(self, which_set="test", include_unknown=True, BG=False, set="numbers"):
        self.mnist = torchvision.datasets.MNIST(
            root=DS_ROOT,
            train=False,
            download=True,
            transform=torchvision.transforms.ToTensor()
        )
        self.letters = torchvision.datasets.EMNIST(
            root= DS_ROOT,
            train= False,
            download=True,
            transform=torchvision.transforms.ToTensor(),
            split="letters"
        )
        targets = list() if not include_unknown else list(range(13)) if which_set != "test" else list(range(13,26))
        self.letter_indexes = [i for i, t in enumerate(self.letters.targets) if t in targets]
        self.BG = BG
        self.set = set
    def __getitem__(self, index):
        if self.set == "numbers":
            return self.mnist[index]
    def __len__(self):
        if self.set == "numbers":
            return len(self.mnist)
             


# Network
arch = "LeNet"
# # Model names that should be tested against


############# Experiments Part 3
graph_list = ["Softmax100None","Softmax100FGSD0.45", "BGSoftmax100Letters", "BGSoftmax100FGSD0.01",  "Cross100Letters", "Cross101FGSM0.01", ] ### best
labels = [r'$S, Baseline$', 
          r'$S, \epsilon = 0.45, FGSM$', 
          r'$B,      Baseline$' , 
          r'$B,     \epsilon = 0.01, FGSM$', 
          r'$E,     Baseline$', 
          r'$E,    \epsilon = 0.01, FGSM$'] ######## good

colors= ['#1f77b4', '#1f77b4', '#ff7f0e',  '#ff7f0e', '#2ca02c',  '#2ca02c', '#1f77b4', '#ff7f0e', '#2ca02c',]
linestyles = ["-", '--', "-", '--', "-", '--', "-", '--',]



labels = graph_list
# models and their labels 
tested_models = graph_list
label_list = labels



# Default loss function for adversary creation
loss = nn.CrossEntropyLoss()
# loss = losses.entropic_openset_loss() "Noise" "FGSD" "FGSD_T6" "L1GPD"
attack_name = "FGSM"
# attack_name = "Noise"
# attack_name = "LinfGPD"

# Results used to plot graphs
results = []
results2 = []
std_results = [] # standard deviation was not implemented in graph
# epsilon parameters for adversary
eps_max = 0.5
eps_iter = 0.01

# iteration loop to get confidence values of each model/network
test_set = Dataset(BG=True, include_unknown=False)
print(len(test_set))




for m in tested_models:
    conf_result = [[],[]]
    conf_result2 = [[],[]]
    std_conf = [[],[]]
    ## Take csv if already got eval scores
    if ( os.path.isfile("advconf/" +"Conf" + m+ attack_name+ str(eps_max) +   ".csv")):
        print("it has it")
        conf_result= pd.read_csv("advconf/" + "ConfNew" + m+  attack_name + str(eps_max) +  ".csv", header=0, ).to_numpy()
        conf_result2= pd.read_csv("advconf/" +"Conf" + m+ attack_name + str(eps_max) +  ".csv", header=0, ).to_numpy()
        results.append(conf_result)
        results2.append(conf_result2)
        continue
    net = load_network(m, arch)
    
    if m.startswith("BG"):    
        val_dl = torch.utils.data.DataLoader(Dataset(BG=True, include_unknown=False), batch_size= len(test_set))
    else:
        val_dl = torch.utils.data.DataLoader(Dataset(BG=False, include_unknown=False), batch_size= len(test_set))
    test(net, val_dl, m, arch, loss, attack_name, conf_result, eps_max, eps_iter, conf_result2, std_conf )
    results.append(conf_result)
    results2.append(conf_result2)
    std_results.append(std_conf) # not implemented




matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["font.size"] = 13




pdf = PdfPages("figures/" +"Conf" + tested_models[-1] + attack_name + str(eps_max)+ ".pdf")
pdf2 = PdfPages("figures/" +"NewConf" + tested_models[-1] + attack_name + str(eps_max)+ ".pdf")

try:
  ##### Confidence 1
  pyplot.figure()
  for i in range(len(results)):
    if (not os.path.isfile("ConfNew"+ tested_models[i] + attack_name + str(eps_max) +   ".csv")):
      pd.DataFrame(results[i]).to_csv("advconf/" + "ConfNew"+ tested_models[i] + attack_name + str(eps_max) +  ".csv", index=False) 
    # pyplot.plot(results[i][1], results[i][0], label = label_list[i], linestyle = linestyles[i], color = colors[i]  )
    pyplot.plot(results[i][1], results[i][0], label = label_list[i],  )
  pyplot.legend()
  pyplot.xlabel(" Perturbation Magnitude  " + str(r'$\varepsilon$'))
  pyplot.ylabel("Confidence of the Maximum Non-True Label")
  pyplot.tight_layout()
  plt.ylim([0, 1])
  pyplot.grid(linestyle = '--', linewidth = 0.2)
  pdf2.savefig(bbox_inches='tight', pad_inches=0)
  pyplot.figure()

    ##### Confidence 2
  for i in range(len(results2)):
    if (not os.path.isfile("advconf/"+ "Conf"+ tested_models[i] + attack_name + str(eps_max) +  ".csv")):
      pd.DataFrame(results2[i]).to_csv("advconf/" +"Conf"+ tested_models[i] + attack_name + str(eps_max) +  ".csv", index=False) 
    # pyplot.plot(results2[i][1], results2[i][0], label = label_list[i], linestyle = linestyles[i], color = colors[i] )
    pyplot.plot(results2[i][1], results2[i][0], label = label_list[i],  )
  pyplot.legend()
  pyplot.xlabel(" Perturbation Magnitude  " + str(r'$\varepsilon$'))
  pyplot.ylabel("Confidence of True Label")
  pyplot.tight_layout()
  plt.ylim([0, 1])
  pyplot.grid(linestyle = '--', linewidth = 0.2)
  pdf.savefig(bbox_inches='tight', pad_inches=0)

  

finally:
  pdf.close()
  pdf2.close()









