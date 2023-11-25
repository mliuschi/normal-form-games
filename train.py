import torch
from torch.utils.data import TensorDataset, DataLoader
import pickle
from model import GameModelPooling
import random
from timeit import default_timer

def total_variation(outputs, targets):
    return torch.sum(torch.norm(outputs - targets, p=float('inf'), dim=-1))

def acc(outputs, targets):
    return torch.sum(torch.argmax(outputs, dim=-1) == torch.argmax(targets, dim=-1))

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def load_data():
    with open("data/two_action_data_train.p", "rb") as file:
        two_action_train = pickle.load(file)
    x = torch.tensor(two_action_train['x'])
    y = torch.tensor(two_action_train['y'])
    two_action_train_loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True, drop_last=True)

    with open("data/two_action_data_test.p", "rb") as file:
        two_action_test = pickle.load(file)
    x = torch.tensor(two_action_test['x'])
    y = torch.tensor(two_action_test['y'])
    two_action_test_loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True, drop_last=True)

    with open("data/three_action_data_train.p", "rb") as file:
        three_action_train = pickle.load(file)
    x = torch.tensor(three_action_train['x'])
    y = torch.tensor(three_action_train['y'])
    three_action_train_loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True, drop_last=True)

    with open("data/three_action_data_test.p", "rb") as file:
        three_action_test = pickle.load(file)
    x = torch.tensor(three_action_test['x'])
    y = torch.tensor(three_action_test['y'])
    three_action_test_loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True, drop_last=True)

    return two_action_train_loader, three_action_train_loader, two_action_test_loader, three_action_test_loader

# https://stackoverflow.com/questions/70849287/how-to-merge-multiple-iterators-to-get-a-new-iterator-which-will-iterate-over-th
def combine_loaders_random(loaders):
    loaders = [load._get_iterator() for load in loaders]
    while loaders:
        it = random.choice(loaders)
        try:
            yield next(it)
        except StopIteration:
            loaders.remove(it)

torch.manual_seed(0)
random.seed(0)

if __name__ == '__main__':
    device = torch.device('cuda')

    # ckpts/backbone_kernels32_nlayers5_max_pool_biasTrue_softmax_residualTrue_ep50
    # Model hyperparameters
    in_dim = 2
    out_dim = 1
    kernels = 32 #32
    num_layers = 5 #5
    activation = 'relu' # 'softmax'
    mode = 'max_pool'
    bias = True
    residual = True
    temperature = 1.0
    dropout = 0.5
    residual_type = 2
    attention = True

    # Training hyperparameters
    batch_size = 32
    n_epochs = 30 # attention: 30 # other: 15
    learning_rate = 1e-4
    scheduler_step = 7
    scheduler_gamma = 0.5

    # Load and process data
    two_train_loader, three_train_loader, two_test_loader, three_test_loader = load_data()
    train_loaders = {"two-action" : two_train_loader, "three-action" : three_train_loader}

    ntrain = len(two_train_loader.dataset) + len(three_train_loader.dataset)
    ntest = len(two_test_loader.dataset) + len(three_test_loader.dataset)
    print("Testing on %d two-action and %d three-action samples" %(len(two_test_loader.dataset), len(three_test_loader.dataset)))
    print()

    # Create model
    model_name = "backbone_kernels" + str(kernels) + "_nlayers" + str(num_layers) + "_" + mode + "_bias" + str(bias) + \
                 "_" + activation + "_residual" + str(residual) + "_ep" + str(n_epochs) + "_dropout" + str(dropout).replace('.', '_') + \
                 "_skiptype" + str(residual_type) + "_selfatten" + str(attention)
    save_path = "ckpts/" + model_name

    model = GameModelPooling(in_planes=in_dim, out_planes=out_dim, num_layers=num_layers,
                             kernels=kernels, mode=mode, bias=bias, residual=residual,
                             activation=activation, temperature=temperature, dropout=dropout,
                             residual_skip=residual_type, self_attention=attention).to(device)
    
    print("Number of parameters:", count_params(model))
    print()

    # Optimizer, scheduler, and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
    loss_fun = torch.nn.CrossEntropyLoss(reduction='sum')

    # Training loop
    train_loaders_copy = train_loaders.copy()
    combined_train_loaders = combine_loaders_random(list(train_loaders_copy.values()))

    model.train()
    for ep in range(1, n_epochs + 1):
        t1 = default_timer()
        train_loss = 0
        correct_count = 0
        total_count_train = 0
        for x, y in combined_train_loaders:
            x = x.float().to(device)
            y = y.float().to(device)

            out = model(x).reshape(batch_size, x.shape[-1])
            loss = loss_fun(out, y)
            train_loss += loss.item()
            correct_count += acc(out, y)
            total_count_train += batch_size

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        test_loss = 0
        test_tv = 0
        test_correct_count = 0
        total_count_test = 0
        with torch.no_grad():
            for x, y in two_test_loader:
                x = x.float().to(device)
                y = y.float().to(device)

                out = model(x).reshape(batch_size, x.shape[-1])
                loss = loss_fun(out, y)
                test_loss += loss.item()
                test_tv += total_variation(out, y)
                test_correct_count += acc(out, y)
                total_count_test += batch_size

            for x, y in three_test_loader:
                x = x.float().to(device)
                y = y.float().to(device)

                out = model(x).reshape(batch_size, x.shape[-1])
                loss = loss_fun(out, y)
                test_loss += loss.item()
                test_tv += total_variation(out, y)
                test_correct_count += acc(out, y)
                total_count_test += batch_size

        t2 = default_timer()
        scheduler.step()
        print("Epoch " + str(ep) + " completed in " + "{0:.{1}f}".format(t2-t1, 3) + \
              " seconds. Train err:", "{0:.{1}f}".format(train_loss/total_count_train, 3), \
                "Train acc:", "{0:.{1}f}".format(correct_count/total_count_train, 3), 
                "Test err:", "{0:.{1}f}".format(test_loss/total_count_test, 3), \
                "Test acc:", "{0:.{1}f}".format(test_correct_count/total_count_test, 3), \
                "Test TV:", "{0:.{1}f}".format(test_tv/total_count_test, 3))
        
        # At the end of each epoch, re-define the iterable
        train_loaders_copy = train_loaders.copy()
        combined_train_loaders = combine_loaders_random(list(train_loaders_copy.values()))

    torch.save(model, save_path)
    print("Weights saved to", save_path)