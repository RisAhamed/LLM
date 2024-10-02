import torch
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
import numpy as np
import os
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

def ddp_setup(rank, world_size):
    """
    Arguments:
        rank: a unique process ID
        world_size: total number of processes in the group
    """
    # rank of machine running rank:0 process
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"

    # initialize process group using 'gloo' for CPU
    init_process_group(backend="gloo", rank=rank, world_size=world_size)

   
    # No need to set CUDA device for CPU

class ToyDataset(Dataset):
    def __init__(self, x,y):
        self.features = x
        self.labels = y

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
    

class NeuralNetwork(torch.nn.Module):
    def __init__(self,num_inputs,num_outputs):
        super().__init__()
        self.layers = torch.nn.Sequential(
            # 1st hidden layer
            torch.nn.Linear(num_inputs, 30),
            torch.nn.ReLU(),

            # 2nd hidden layer
            torch.nn.Linear(30, 20),
            torch.nn.ReLU(),

            # output layer
            torch.nn.Linear(20, num_outputs),
        )

    def forward(self, x):
        logits = self.layers(x)
        return logits

def prepare_dataset():
    x_train = torch.tensor([
        [-1.2, 3.1],
        [-0.9, 2.9],
        [-0.5, 2.6],
        [2.3, -1.1],
        [2.7, -1.5]
    ])
    y_train = torch.tensor([0,0,0,1,1])

    x_test = torch.tensor([
        [-0.8, 2.8],
        [2.6, -1.6],
    ])
    y_test = torch.tensor([0, 1])

    train_ds = ToyDataset(x_train,y_train)
    test_ds = ToyDataset(x_test,y_test)
    train_loader = Dataset(
        Dataset = train_ds,
        batch_size =32,
        shuffle =True,
    )
    test_loader = Dataset(
        dataset = test_ds,
        batch_size =32,
        shuffle =True,)
    
    return train_loader, test_loader

def main(rank, world_size, num_epochs):

    # Remove DDP and CUDA-specific setup
    ddp_setup(rank, world_size)

    train_loader, test_loader = prepare_dataset()

    # Define model and optimizer for CPU
    model = NeuralNetwork(num_inputs=2, num_outputs=2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

    for epoch in range(num_epochs):

        model.train()
        for features, labels in train_loader:

            # No need to move features and labels to GPU
            logits = model(features)
            loss = F.cross_entropy(logits, labels)  # Loss function

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # LOGGING
            print(f"Epoch: {epoch+1:03d}/{num_epochs:03d}"
                  f" | Batchsize {labels.shape[0]:03d}"
                  f" | Train/Val Loss: {loss:.2f}")

    # No need to move the model to CPU explicitly, it's already on CPU
    model.eval()
    train_acc = compute_accuracy(model, train_loader, device="cpu")
    print(f"Training accuracy", train_acc)
    test_acc = compute_accuracy(model, test_loader, device="cpu")
    print(f"Test accuracy", test_acc)

    # Clean up for CPU
    destroy_process_group()

def compute_accuracy(model, dataloader, device):
    model = model.eval()
    correct = 0.0
    total_examples = 0

    for idx, (features, labels) in enumerate(dataloader):
        features, labels = features.to(device), labels.to(device)

        with torch.no_grad():
            logits = model(features)
        predictions = torch.argmax(logits, dim=1)
        compare = labels == predictions
        correct += torch.sum(compare)
        total_examples += len(compare)
    return (correct / total_examples).item()


if __name__ == "__main__":
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("Number of GPUs available:", torch.cuda.device_count())

    torch.manual_seed(123)

    # NEW: spawn new processes
    # note that spawn will automatically pass the rank
    num_epochs = 3
    world_size = 1
    mp.spawn(main, args=(world_size, num_epochs), nprocs=world_size)
    # nprocs=world_size spawns one process per GPU
