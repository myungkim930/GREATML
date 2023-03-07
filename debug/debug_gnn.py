#%% Setup

# Set working directory
import os

os.chdir("/storage/store3/work/mkim/gitlab/GREATML")

# load data
from data_utils import Load_data

data_name = "yago3"
main_data = Load_data(data_name)
main_data.reduce(num_rel=5)

#%%
import torch
from model.yate_gnn import yate_att_calc, yate_att_output, yate_multihead
from graphlet_construction import add_self_loops

edge_index = torch.tensor([[0, 0, 0, 0], [1, 2, 3, 4]])

q = torch.rand(5, 3)
k = torch.rand(4, 3)
v = torch.rand(4, 3)

# attention is calculated with respect to the edge_index
att = yate_att_calc(edge_index=edge_index, query=q, key=k)
output = yate_att_output(edge_index=edge_index, attention=att, value=v)

# This outputs only one row since the edge is directed towards the center node.
# To avoid this, we must include self-loops into the edge_index

edge_index = add_self_loops(edge_index=edge_index)
q = torch.rand(5, 3)
k = torch.rand(9, 3)
v = torch.rand(9, 3)

att = yate_att_calc(edge_index=edge_index, query=q, key=k)
output = yate_att_output(edge_index=edge_index, attention=att, value=v)

# Test with real-dataset - sinlge graphlet
from data_utils import Load_data
import graphlet_construction as gc

g = gc.Graphlet(main_data, num_hops=1)
idx_head = main_data.edge_index[0, :].unique()
idx = idx_head[torch.randperm(idx_head.size()[0])][0]
idx = int(idx)

# Rather than the graphlet, we need to use the make_batch method
data = g.make_batch(idx_cen=idx, aug=False)
print(data)

# For our model, we use Z, which is simply the element-wise multiplication of
# node and edge features
Z = torch.mul(data.edge_attr, data.x[data.edge_index[1, :]])
q = torch.mul(data.x, torch.rand(data.x.size()))
k = torch.mul(Z, torch.rand(Z.size()))
v = torch.mul(Z, torch.rand(Z.size()))
v = torch.ones(Z.size())
# The graphlet construction automatically adds a self-loop and the data we have preprocessed
# already contains the self-loop relation with 'hasName'. This maybe is subject to change.
att = yate_att_calc(edge_index=data.edge_index, query=q, key=k)
output = yate_att_output(edge_index=data.edge_index, attention=att, value=v)
print(att)
print(output.size())

# The yate_gnn model also includes multi-head attention..
# This takes in the yate_att_calc, yate_att_output functions
# For now, I think there are better ways to optimize the heads, so this will def change later

output, attention = yate_multihead(
    edge_index=data.edge_index,
    query=q,
    key=k,
    value=v,
    num_heads=2,
    concat=True,
)


# We can feed-forward this in the attention layer
from model.yate_gnn import YATE_Attention as YA

att_layer = YA(
    input_dim=300,
    output_dim=300,
    num_heads=1,
    concat=True,
)

output, edge_feat = att_layer(
    x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr
)

# Finally with the overall model, we have the following
# Since the last classfication layer has two outputs, the output will have two values
import torch
import graphlet_construction as gc

g = gc.Graphlet(main_data, num_hops=1)
idx_head = main_data.edge_index[0, :].unique()
idx = idx_head[torch.randperm(idx_head.size()[0])][0:10]
# idx = int(idx)

data = g.make_batch(idx_cen=idx, aug=False)
print(data)

from model.yate_gnn import YATE_Encode as YE

model = YE(
    input_dim_x=300,
    input_dim_e=300,
    hidden_dim=300,
    num_layers=3,
    ff_dim=300,
    num_heads=1,
)
output = model(data)

# Check the gradient flow to see there are any weights that are not trained
from torch import optim

# Rather than the graphlet, we need to use the make_batch method

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
idx_head = main_data.edge_index[0, :].unique()
idx = idx_head[torch.randperm(idx_head.size()[0])][0:32]
idx = int(idx)
n_batch = 10
n_pos = 3
n_neg = 1
data_batch = g.make_batch(idx_cen=idx, n_pos=n_pos, n_neg=n_neg, per_neg=0.8)


data_batch.to(device)
model = model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

optimizer.zero_grad()

output = model(data_batch)
target = data_batch.y.type(torch.int64)
loss = criterion(output, target)
loss.backward()

for name, param in model.named_parameters():
    if param.grad is None:
        print(name)

for name, param in model.named_parameters():
    print(name, param.grad.norm())

