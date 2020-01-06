import argparse
import random
import math
import numpy as np
from pathlib import Path
import pickle

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from GBPR.model import GBPR
from preprocess import MovieLens 

class Data(Dataset):
    def __init__(self, user_size, item_size, pair, u2i, i2u, num_neg, group_size):
        self.user_size = user_size
        self.item_size = item_size
        self.pair = pair
        self.u2i = u2i
        self.i2u = i2u
        self.num_neg = num_neg
        self.group_size = group_size
    
    def get_neg_and_group(self):
        # resample negative items during each epoch.
        self.full_data = []
        for u, i in self.pair:
            for t in range(self.num_neg):
                j = np.random.randint(self.item_size)
        
                while j in self.u2i[u]:
                    j = np.random.randint(self.item_size)
                
                tmp = random.sample(self.i2u[i], self.group_size - 1)
                if u in tmp:
                    
                    tmp_one = random.sample(self.i2u[i], 1)
                    while u in tmp_one:
                        tmp_one = random.sample(self.i2u[i], 1)
                    
                    G = np.array(tmp + tmp_one)
                
                else:
                    G = np.array(tmp + [u])
                
                self.full_data.append([u, i, j, G]) 

    def __getitem__(self, index):
        
        u, i, j, G = self.full_data[index]
        return u, i, j, G
    
    def __len__(self): 
        return self.num_neg * len(self.pair)

def train(loader, model, optimizer, test_pos, test_data, device, args):

    ## training and running on GPU if cuda is available.
    model.train()
    model.to(device)
    total_loss = 0.0
    batch_count = 0

    for epoch_id in range(args.n_epochs):
        loader.dataset.get_neg_and_group()
        for batch_id , (batch_u, batch_i, batch_j, batch_G)in enumerate(loader):
            
            # move data to GPU
            batch_u = batch_u.to(device)
            batch_i = batch_i.to(device)
            batch_j = batch_j.to(device)
            batch_G = batch_G.to(device)
            ratio = torch.tensor(args.ratio).to(device)
             
            optimizer.zero_grad()
            
            loss = model(batch_u, batch_i, batch_j, batch_G, ratio)
            
            loss.backward()

            optimizer.step()
            
            batch_count += 1
            total_loss += loss.data

            avg_loss = total_loss / batch_count
            
            
            if batch_id % args.print_num == 0:
                print(f"Training Epoch : {epoch_id} | [{batch_id} / {len(loader)}] | Batch Loss = {loss/args.batch_size:.4f} | Total Average Loss = {avg_loss/args.batch_size:.4f}\n")
                
                if args.write:
                    with open(Path(Path().cwd() / f"results/GBPR_Loss_{args.optim}_lr_{args.lr}_ratio_{args.ratio}_gs_{args.group_size}_wd_{args.weight_decay}.txt") , 'a') as f:
                        f.write(f"Training Epoch : {epoch_id} | [{batch_id} / {len(loader)}] | Batch Loss = {loss/args.batch_size:.4f} | Total Average Loss = {avg_loss/args.batch_size:.4f}\n")
            
            if batch_id % args.eval_num == 0:
                hit_ratio, ndcg = _eval(model, test_pos, test_data)
                print(f"Training Epoch : {epoch_id} | [{batch_id} / {len(loader)}] | Hit Ratio = {hit_ratio:.4f} | NDCG = {ndcg:.4f}\n")
                 
                if args.write: 
                    with open(Path(Path().cwd() / f"results/GBPR_Hit_{args.optim}_lr_{args.lr}_ratio_{args.ratio}_gs_{args.group_size}_wd_{args.weight_decay}.txt") , 'a') as f:
                        f.write(f"Training Epoch : {epoch_id} | [{batch_id} / {len(loader)}] | Hit Ratio = {hit_ratio:.4f} | NDCG = {ndcg:.4f}\n")

def _eval(model, test_pos, test_sample):
    
    model.eval()
    result = model.predict(test_sample)
    num_users = result.shape[0]

    hit = 0
    ndcg = 0

    for i in range(num_users):
        
        retrieve_items = list(result[i])
        label = test_pos[i]

        if label in retrieve_items:
            hit += 1
            ndcg += (1 / math.log(retrieve_items.index(label)+2,2))

    return (hit / num_users), (ndcg / num_users)


def main(args):
   
    # load preprocessing data
    data_path = Path( args.data_dir / f"preprocess/{args.dataset}.pickle")
    if data_path.exists():
        print("Preprocessing file Exists!")
        with open(data_path, 'rb') as f:
            dataset = pickle.load(f)
        
        train_u2i = dataset["train_u2i"]
        train_i2u = dataset["train_i2u"]
        train_pair = dataset["train_pair"]
        test_pos = dataset["test_pos"]
        test_data = dataset["test_data"]

    else:
        train_u2i, train_i2u, train_pair, test_pos, test_data = \
                MovieLens(args.data_dir, args.dataset).load()
        
    # make directory for storing the results.
    args.result_dir.mkdir(exist_ok=True)
   
    # remove (u, i) which the number of interaction of item i is less than group size.
    train_pair = [(u, i) for u, i in train_pair if len(train_i2u[i]) >= args.group_size]

    user_size = len(train_u2i)
    item_size = len(train_i2u)

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else 'cpu')
    
    dataset = Data(user_size, item_size, train_pair, train_u2i, train_i2u, args.num_neg, args.group_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = GBPR(user_size, item_size, args.embedding_size, args.batch_size, device)
    
    if args.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    
    train(loader, 
          model, 
          optimizer, 
          test_pos,
          test_data,
          device,
          args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_dir', 
                        type=Path,
                        default= Path(Path.cwd() / "data"),
                        help="Specify the directory of preprocessed data.")

    parser.add_argument('--result_dir', 
                        type=Path,
                        default= Path(Path.cwd() / "results"),
                        help="Specify the directory of results.")

    parser.add_argument('--dataset',
                        type=str,
                        default='ml-1m',
                        help="Specify which dataset used.")

    parser.add_argument('--batch_size', 
                        type=int,
                        default=4096,
                        help="Specify the batch size")
    
    parser.add_argument('--group_size', 
                        type=int,
                        default=3,
                        help="Specify the group size")
    
    parser.add_argument('--ratio', 
                        type=float,
                        default=0.6,
                        help="Specify the ratio tradeoff between individual preference and group preference.")
    
    parser.add_argument('--num_neg', 
                        type=int,
                        default=4,
                        help="Specify the number of negative items for each positive items")
    
    parser.add_argument('--n_epochs',
                        type=int,
                        default=100,
                        help="Specify the number of epochs for training")
    
    parser.add_argument('--lr',
                        type=float,
                        default=1e-2,
                        help="Specify learning rate for training")
    
    parser.add_argument('--embedding_size',
                        type=int,
                        default=32,
                        help="Specify dimension of user and item embedding")
    
    parser.add_argument('-m', '--momentum',
                        type=float,
                        default=0.9,
                        help="Specify momentum for SGD.")
   
    parser.add_argument('--print_num',
                        type=int,
                        default=40,
                        help="Specify the number of samples to show training information")
   
    parser.add_argument('--eval_num',
                        type=int,
                        default=200,
                        help="Specify the number of samples to evaluate.")

    parser.add_argument('--gpu_id',
                        type=int,
                        default=0,
                        help="Specify the id of GPU.")
    
    parser.add_argument('--weight_decay',
                        type=float,
                        default=0.001,
                        help="Weight decay factor")

    parser.add_argument('--optim',
                        default='Adam',
                        choices=['SGD','Adam'],
                        help="Specify the optim for training.")
  
    parser.add_argument('-w', '--write',
                        action="store_true",
                        help="Whether to write results to files.")

    parser.add_argument('-b', '--bias',
                        action="store_true",
                        help="Whether to add bias to model.")

    args = parser.parse_args()
    main(args)

