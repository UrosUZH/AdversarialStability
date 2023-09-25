import vast
import torch
import torchvision
import numpy
import os
import csv
import pandas as pd

import matplotlib
matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["font.size"] = 18
from matplotlib import pyplot, patches
from matplotlib.backends.backend_pdf import PdfPages

DS_ROOT = "/tmp"

if torch.cuda.is_available():
    vast.tools.set_device_gpu()
else:
    print("Running in CPU mode, training might be slow")
    vast.tools.set_device_cpu()



def load_network(which):
  net = vast.architectures.LeNet(use_BG=which.startswith("BG"))
  net.load_state_dict(torch.load(f"LeNet/{which}/{which}.model"))
  vast.tools.device(net)
  return net



graph_list = []

graph_list = ["Softmax100None","Softmax100FGSD0.45", "BGSoftmax100Letters", "BGSoftmax100FGSD0.01",  "Cross100Letters", "Cross101FGSM0.01", ] ### best
labels = [r'$S, Baseline$', 
          r'$S, \epsilon = 0.45, FGSM$', 
          r'$B,      Baseline$' , 
          r'$B,     \epsilon = 0.01, FGSM$', 
          r'$E,     Baseline$', 
          r'$E,    \epsilon = 0.01, FGSM$'] ######## good

colors= ['#1f77b4', '#1f77b4', '#ff7f0e',  '#ff7f0e', '#2ca02c',  '#2ca02c', ]

linestyles = ["-", '--', "-", '--', "-", '--', "-", '--',]



labels = graph_list


networks = {
  which: load_network(which) for which in graph_list # ("Softmax", "BGSoftmax", "Cross", 'ObjectoSphere')
}


from MNIST_SoftMax_Training import Dataset

# known unknown set
val_set = Dataset(DS_ROOT, "validation", include_unknown=True)
# unknown unknown set
test_set = Dataset(DS_ROOT, "test", include_unknown=True)

def extract(dataset):
  gt, logits = [], []
  data_loader = torch.utils.data.DataLoader(dataset, batch_size=2048, shuffle=False)

  with torch.no_grad():
    for (x, y) in data_loader:
      gt.extend(y.tolist())
      logs, feat = net(vast.tools.device(x))
      logits.extend(logs.tolist())

  return numpy.array(gt), numpy.array(logits)


results = {}
for which, net in networks.items():
  if (os.path.isfile("OSCR/" + which +  ".csv")):
    print("it has it")
    results[which]= pd.read_csv("OSCR/" + which +  ".csv", header=0, ).to_numpy()
    # pd.DataFrame(results[which]).to_csv(which +  "new.csv", index=False)
    continue
  print ("Evaluating", which)
  # extract positives
  val_gt, val_predicted = extract(val_set)
  test_gt, test_predicted = extract(test_set)

  # compute probabilities
  val_predicted = torch.nn.functional.softmax(torch.tensor(val_predicted), dim=1).detach().numpy()
  test_predicted  = torch.nn.functional.softmax(torch.tensor(test_predicted ), dim=1).detach().numpy()

  if which.startswith("BG"):
    # remove the labels for the unknown class in case of BG-softmax
    print("its bg")
    val_predicted = val_predicted[:,:-1]
    test_predicted = test_predicted[:,:-1]

  # vary thresholds
  ccr, fprv, fprt = [], [], []
  positives = val_predicted[val_gt != -1]
  val = val_predicted[val_gt == -1]
  test = test_predicted[test_gt == -1]
  gt = val_gt[val_gt != -1]
  for tau in sorted(positives[range(len(gt)),gt]):
    # correct classification rate
    ccr.append(numpy.sum(numpy.logical_and(
      numpy.argmax(positives, axis=1) == gt,
      positives[range(len(gt)),gt] >= tau
    )) / len(positives))
    # false positive rate for validation and test set
    fprv.append(numpy.sum(numpy.max(val, axis=1) >= tau) / len(val))
    fprt.append(numpy.sum(numpy.max(test, axis=1) >= tau) / len(test))

  results[which] = (ccr, fprv, fprt)


pdf = PdfPages("figures/" + "NewKU" + graph_list[-1] + ".pdf")
pdf2 = PdfPages("figures/" + "NewUU" + graph_list[-1] + ".pdf")
# colors = ['lightblue', 'orange', 'green']
linestyle= ["-", "-", "-"]
labels1 = ["$\it{S}$", "$\it{BG}$" ,"$\it{EOS}$"]
index = 0
try:
  
  # plot with known unknowns (letters 1:13)
  fig, ax = pyplot.subplots()
  pyplot.grid(linestyle = '--', linewidth = 0.2)
  for which, res in results.items():
    if (not os.path.isfile("OSCR/" +which +  ".csv")):
      pd.DataFrame(res).to_csv("OSCR/" + which +  ".csv", index=False) 
    # pyplot.semilogx(res[1], res[0], label=labels[index], linestyle = linestyles[index], color = colors[index])
    pyplot.semilogx(res[1], res[0], label=labels[index], )
    index+=1
  pyplot.legend(loc="lower right", fontsize = "xx-small",)
  pyplot.xlabel("False Positive Rate")
  pyplot.ylabel("Correct Classification Rate")
  pyplot.tight_layout()
  pdf.savefig(bbox_inches='tight', pad_inches=0)
  
  index = 0
  # plot with unknown unknowns (letters 14:26)
  pyplot.figure()
  pyplot.grid(linestyle = '--', linewidth = 0.2)
  for which, res in results.items():
    # pyplot.semilogx(res[2], res[0], label=labels[index], linestyle = linestyles[index], color = colors[index])
    pyplot.semilogx(res[2], res[0], label=labels[index], )
    index+=1
  pyplot.legend(loc="lower right", fontsize = "xx-small", )
  pyplot.xlabel("False Positive Rate")
  pyplot.ylabel("Correct Classification Rate")
  pyplot.tight_layout()
  pdf2.savefig(bbox_inches='tight', pad_inches=0)
  

finally:
  pdf.close()
  pdf2.close()

