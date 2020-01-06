import argparse
import math
import numpy as np
from pathlib import Path
import pickle

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from BPR.model import BPR
from preprocess import MovieLens 

class Data(Dataset):
    def __init__(self, user_size, item_size, pair, user_list):
        self.user_size = user_size
        self.item_size = item_size
        self.pair = pair
        self.user_list = user_list
    
    def get_neg(self):
        
        self.full_data = []
        for u, i in self.pair:
            for t in range(4):
                j = np.random.randint(self.item_size)
        
                while j in self.user_list[u]:
                    j = np.random.randint(self.item_size)
                self.full_data.append([u, i, j]) 

    def __getitem__(self, index):
        
        u, i, j = self.full_data[index]
        return u, i, j
    
    def __len__(self): 
        return 4 * len(self.pair)

def train(loader, model, optimizer, test_pos, test_data, device, args):

    ## training and running on GPU if cuda is available.
    model.train()
    model.to(device)
    total_loss = 0.0
    batch_count = 0

    for epoch_id in range(args.n_epochs):
        loader.dataset.get_neg()
        for batch_id , (batch_u, batch_i, batch_j)in enumerate(loader):

            # move data to GPU
            batch_u = batch_u.to(device)
            batch_i = batch_i.to(device)
            batch_j = batch_j.to(device)

            optimizer.zero_grad()
            
            loss = model(batch_u, batch_i, batch_j)
            
            loss.backward()

            optimizer.step()
            
            batch_count += 1
            total_loss += loss.data

            avg_loss = total_loss / batch_count
            
            
            if batch_id % args.print_num == 0:
                print(f"Training Epoch : {epoch_id} | [{batch_id} / {len(loader)}] | Batch Loss = {loss/args.batch_size:.4f} | Total Average Loss = {avg_loss/args.batch_size:.4f}\n")
                
                if args.write:
                    with open(Path(Path().cwd() / f"results/BPR_Loss_{args.optim}_lr_{args.lr}_dim_{args.embedding_size}_wd_{args.weight_decay}.txt") , 'a') as f:
                        f.write(f"Training Epoch : {epoch_id} | [{batch_id} / {len(loader)}] | Batch Loss = {loss/args.batch_size:.4f} | Total Average Loss = {avg_loss/args.batch_size:.4f}\n")
            
            if batch_id % args.eval_num == 0:
                hit_ratio, ndcg = _eval(model, test_pos, test_data)
                print(f"Training Epoch : {epoch_id} | [{batch_id} / {len(loader)}] | Hit Ratio = {hit_ratio:.4f} | NDCG = {ndcg:.4f}\n")
                 
                if args.write: 
                    with open(Path(Path().cwd() / f"results/BPR_Hit_{args.optim}_lr_{args.lr}_dim_{args.embedding_size}_wd_{args.weight_decay}.txt") , 'a') as f:
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
    

    user_size = len(train_u2i)
    item_size = len(train_i2u)

    print(user_size, item_size, len(test_data))

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else 'cpu')
    
    dataset = Data(user_size, item_size, train_pair, train_u2i)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = BPR(user_size, item_size, args.embedding_size, args.batch_size, train_u2i, device)
    
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
    
    parser.add_argument('--n_epochs',
                        type=int,
                        default=50,
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

