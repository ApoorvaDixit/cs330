import torch.nn as nn
from torch.autograd import Function
import torch.optim as optim
import torch
from dataloader import UDADataLoader
from tqdm import tqdm
from models import FeatureExtractor, Classifier

    
feature_extractor = FeatureExtractor()
classifier = Classifier()

learning_rate = 0.001
num_epochs = 25
optimizer = optim.Adam(list(feature_extractor.parameters()) + list(classifier.parameters()), lr=learning_rate, weight_decay=1e-5)
classification_loss = nn.CrossEntropyLoss()

dataloader = UDADataLoader()

# Loss functions
classification_loss = nn.CrossEntropyLoss()

for epoch in tqdm(range(num_epochs)):
    print("epoch")
    for source_data, source_labels in tqdm(dataloader.svhn_train_loader):
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass for source data
        source_features = feature_extractor(source_data)
        source_preds = classifier(source_features)
        cls_loss = classification_loss(source_preds, source_labels)

        # Backward and optimize
        total_loss = cls_loss
        total_loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss.item()}')
    
    
feature_extractor_checkpoint = {
    'epoch': epoch,
    'model_state_dict': feature_extractor.state_dict(),
    # Include any other relevant information
}

torch.save(feature_extractor_checkpoint, "non_uda_ckpts/feature_extractor_src_svhn_epochs_25.ckpt")

classifier_checkpoint = {
    'epoch': epoch,
    'model_state_dict': classifier.state_dict(),
    # Include any other relevant information
}

torch.save(classifier_checkpoint, "non_uda_ckpts/classifier_src_svhn_epochs_25.ckpt")