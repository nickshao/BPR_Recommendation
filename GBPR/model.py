import torch
import torch.nn as nn

class GBPR(nn.Module):
    def __init__(self, user_size, item_size, embedding_size, batch_size, device):
        super(GBPR, self).__init__()
        self.user_matrix = nn.Embedding(user_size, embedding_size)
        self.item_matrix = nn.Embedding(item_size, embedding_size)

        nn.init.normal_(self.user_matrix.weight, std=0.01)
        nn.init.normal_(self.item_matrix.weight, std=0.01)

        #self.item_bias = nn.Embedding(item_size, 1)
        #nn.init.normal_(self.item_bias.weight, std=0.01)
        
        self.batch = batch_size
        self.device = device

    def forward(self, u, i, j, G, ratio):

        "Maximize the difference of (R_Gui - R_uj)" 
        R_Gi = (self.user_matrix(G) * self.item_matrix(i)[:, None, :]).sum(dim=-1).mean(dim=1)
        R_ui = torch.mul(self.user_matrix(u), self.item_matrix(i)).sum(dim=-1)
        R_gui = ratio * (R_Gi - R_ui) + R_ui

        R_uj = torch.mul(self.user_matrix(u), self.item_matrix(j)).sum(dim=1)
        
        loss = torch.log(torch.sigmoid(R_gui - R_uj)).sum()
        
        return -loss

    def predict(self, test_data, k=10):
        
        user_emb = self.user_matrix.weight.detach()
        item_emb = self.item_matrix.weight.detach()

        result = None
        for i in range(0, user_emb.shape[0], self.batch):

            mask = user_emb.new_zeros([min([self.batch, user_emb.shape[0]-i]), item_emb.shape[0]])
            for j in range(self.batch):
                # index out of bounds.
                
                if i + j >= user_emb.shape[0]: break
                else:
                    mask[j].scatter_(dim=0, index=torch.tensor(test_data[i+j]).to(self.device), value=torch.tensor(1.0).to(self.device))
            
            cur_result = torch.mm(user_emb[i:i+min(self.batch, user_emb.shape[0] - i), :], item_emb.t()) 
            cur_result = torch.mul(mask, cur_result)
        
            _, cur_result = torch.topk(cur_result, k=k, dim=1)
            
            result = cur_result if result is None else torch.cat((result, cur_result), dim=0)

        return result.cpu().numpy()


