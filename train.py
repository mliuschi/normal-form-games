import torch
from torch.utils.data import TensorDataset, DataLoader
import pickle
from model import GameModelPooling
import random
from timeit import default_timer

def acc(outputs, targets):
    return torch.sum(torch.argmax(outputs, dim=-1) == torch.argmax(targets, dim=-1))

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def load_data():
    with open("data/two_action_data_train.p", "rb") as file:
        two_action_train = pickle.load(file)
    x = torch.tensor(two_action_train['x'])
    y = torch.tensor(two_action_train['y'])
    two_action_train_loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)

    with open("data/two_action_data_test.p", "rb") as file:
        two_action_test = pickle.load(file)
    x = torch.tensor(two_action_test['x'])
    y = torch.tensor(two_action_test['y'])
    two_action_test_loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)

    with open("data/three_action_data_train.p", "rb") as file:
        three_action_train = pickle.load(file)
    x = torch.tensor(three_action_train['x'])
    y = torch.tensor(three_action_train['y'])
    three_action_train_loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)

    with open("data/three_action_data_test.p", "rb") as file:
        three_action_test = pickle.load(file)
    x = torch.tensor(three_action_test['x'])
    y = torch.tensor(three_action_test['y'])
    three_action_test_loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)

    return two_action_train_loader, three_action_test_loader, two_action_test_loader, three_action_train_loader

# https://stackoverflow.com/questions/70849287/how-to-merge-multiple-iterators-to-get-a-new-iterator-which-will-iterate-over-th
def combine_loaders_random(loaders):
    loaders = [load._get_iterator() for load in loaders]
    while loaders:
        it = random.choice(loaders)
        try:
            yield next(it)
        except StopIteration:
            loaders.remove(it)

if __name__ == '__main__':
    device = torch.device('cuda')

    # Model hyperparameters
    in_dim = 2
    out_dim = 1
    kernels = 8
    mode='max_pool'
    bias=True
    residual=False

    # Training hyperparameters
    batch_size = 32
    n_epochs = 500
    learning_rate = 1e-3
    scheduler_step = 100
    scheduler_gamma = 0.5

    # Load and process data
    two_train_loader, three_train_loader, two_test_loader, three_test_loader = load_data()
    train_loaders = {"two-action" : two_train_loader, "three-action" : three_train_loader}

    ntrain = len(two_train_loader.dataset) + len(three_train_loader.dataset)
    ntest = len(two_test_loader.dataset) + len(three_test_loader.dataset)

    # Create model
    model_name = "backbone_kernels" + str(kernels) + "_" + mode + "_bias" + str(bias) + "_residual" + str(residual) + "_ep" + str(n_epochs)
    save_path = "ckpts/" + model_name

    model = GameModelPooling(in_planes=in_dim, out_planes=out_dim, kernels=kernels,
                             mode=mode, bias=bias, residual=residual).to(device)
    
    print("Number of parameters:", count_params(model))
    print()

    # Optimizer, scheduler, and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
    loss = torch.nn.CrossEntropyLoss(reduction='sum')

    # Training loop
    train_loaders_copy = train_loaders.copy()
    combined_train_loaders = combine_loaders_random(list(train_loaders_copy.values()))

    model.train()
    for ep in range(1, n_epochs + 1):
        t1 = default_timer()
        train_loss = 0
        correct_count = 0
        for x, y in combined_train_loaders:
            x = x.to(device)
            y = y.to(device)

            out = model(x).reshape(batch_size, x.shape[-1])
            loss = loss(out, y)
            train_loss += loss.item()
            correct_count += acc(out, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        test_loss = 0
        test_correct_count = 0
        with torch.no_grad():
            for x, y in two_test_loader:
                x = x.to(device)
                y = y.to(device)

                out = model(x).reshape(batch_size, x.shape[-1])
                test_loss = loss(out, y)
                test_correct_count += acc(out, y)

        t2 = default_timer()
        scheduler.step()
        print("Epoch " + str(ep) + " completed in " + "{0:.{1}f}".format(t2-t1, 3) + \
              " seconds. Train err:", "{0:.{1}f}".format(train_loss/ntrain, 3), \
                "Train acc:", "{0:.{1}f}".format(correct_count/ntest, 3), 
                "Test err:", "{0:.{1}f}".format(test_loss/ntest, 3), \
                "Test acc:", "{0:.{1}f}".format(test_correct_count/ntest, 3))
        
        # At the end of each epoch, re-define the iterable
        train_loaders_copy = train_loaders.copy()
        combined_train_loaders = combine_loaders_random(list(train_loaders_copy.values()))

    #torch.save(model, save_path)
    print("Weights saved to", save_path)