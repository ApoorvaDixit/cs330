import torch.nn as nn
from torch.autograd import Function
import torch.optim as optim
import torch
from dataloader import UDADataLoader
from tqdm import tqdm
from models import FeatureExtractor, Classifier

correct = 0
total = 0

dataloader = UDADataLoader()

feature_extractor = FeatureExtractor()
classifier = Classifier()

feature_extractor_checkpoint = torch.load("non_uda_ckpts/feature_extractor_src_mnist_epochs_20.ckpt")
feature_extractor.load_state_dict(feature_extractor_checkpoint['model_state_dict'])

classifier_checkpoint = torch.load("non_uda_ckpts/classifier_src_mnist_epochs_20.ckpt")
classifier.load_state_dict(classifier_checkpoint['model_state_dict'])

with torch.no_grad():
    for data, labels in dataloader.mnist_test_loader:
        features = feature_extractor(data)
        outputs = classifier(features)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy_mnist = 100 * correct / total
print(f'Accuracy of the model on the MNIST test images: {accuracy_mnist:.2f}%')

correct = 0
total = 0

with torch.no_grad():
    for data, labels in dataloader.svhn_test_loader:
        features = feature_extractor(data)
        outputs = classifier(features)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy_svhn = 100 * correct / total
print(f'Accuracy of the model on the SVHN test images: {accuracy_svhn:.2f}%')

