import torch.nn as nn
from torch.autograd import Function
import torch.optim as optim
import torch
from dataloader import UDADataLoader
from tqdm import tqdm
from models import FeatureExtractor, Classifier, DomainDiscriminator

class GradientReversalFn(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

feature_extractor = FeatureExtractor()
classifier = Classifier()
domain_discriminator = DomainDiscriminator()

# Hyperparameters
learning_rate = 0.001
alpha = 0.1  # GRL hyperparameter
num_epochs = 25

# Optimizers
optimizer = optim.Adam(list(feature_extractor.parameters()) + list(classifier.parameters()), lr=learning_rate, weight_decay=1e-5)
domain_optimizer = optim.Adam(domain_discriminator.parameters(), lr=learning_rate, weight_decay=1e-5)

# Loss functions
classification_loss = nn.CrossEntropyLoss()
domain_loss = nn.CrossEntropyLoss()

dataloader = UDADataLoader()

print(len(dataloader.mnist_train_loader))
print(len(dataloader.svhn_train_loader))

for epoch in tqdm(range(num_epochs)):
    print("epoch")
    for (source_data, source_labels), (target_data, _) in tqdm(zip(dataloader.svhn_train_loader, dataloader.mnist_train_loader), total=len(dataloader.mnist_train_loader)):
        # Zero the parameter gradients
        optimizer.zero_grad()
        domain_optimizer.zero_grad()

        # Forward pass for source data
        # print(source_data.shape)
        source_features = feature_extractor(source_data)
        source_preds = classifier(source_features)
        cls_loss = classification_loss(source_preds, source_labels)

        # Forward pass for domain adaptation
        combined_data = torch.cat((source_features, feature_extractor(target_data)), 0)
        domain_labels = torch.cat((torch.zeros(source_data.size(0)), torch.ones(target_data.size(0)))).long()
        domain_preds = domain_discriminator(GradientReversalFn.apply(combined_data, alpha))
        dom_loss = domain_loss(domain_preds, domain_labels)

        # Backward and optimize
        total_loss = cls_loss + dom_loss
        total_loss.backward()
        optimizer.step()
        domain_optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss.item()}')
    
feature_extractor_checkpoint = {
    'epoch': epoch,
    'model_state_dict': feature_extractor.state_dict(),
    # Include any other relevant information
}

torch.save(feature_extractor_checkpoint, "ckpts/feature_extractor_src_svhn_epochs_25.ckpt")

classifier_checkpoint = {
    'epoch': epoch,
    'model_state_dict': classifier.state_dict(),
    # Include any other relevant information
}

torch.save(classifier_checkpoint, "ckpts/classifier_src_svhn_epochs_25.ckpt")
    
