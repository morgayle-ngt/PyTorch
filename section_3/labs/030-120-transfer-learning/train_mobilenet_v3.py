"""
Train the mobilenetv3 model for 3 epochs.

Set the scheduler as well as the save it in the checkpoint every epoch.
"""
from pre import train_loader, val_loader
from freeze_mobilenet_v3 import model
from mobilenet_v3_scheduler import criterion, optimizer, scheduler
import torch

# Set number of epochs
N_EPOCHS = 3

for epoch in range(N_EPOCHS):
    ####### TRAINING
    training_loss = 0.0
    model.train()
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()  # Clear gradients
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        training_loss += loss.item()

    ######## VALIDATION
    val_loss = 0.0
    model.eval()
    for i, data in enumerate(val_loader, 0):
        inputs, labels = data
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        val_loss += loss.item()

    # Step the scheduler at the end of the epoch
    scheduler.step()

    ######## SAVE A CHECKPOINT
    torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': training_loss,
                'val_loss': val_loss},
               f'mobilenet_v3_{epoch}_checkpoint.tar')

    # Print the training and validation losses
    print(f"Epoch: {epoch} Train Loss: {training_loss/len(train_loader)} Val Loss: {val_loss/len(val_loader)}")