
def create_target_node(data):

    graph_idx = data.g_idx
    pos_mask = (
        graph_idx.repeat(graph_idx.size(0), 1)
        - graph_idx.repeat(graph_idx.size(0), 1).t()
    )

    target = pos_mask.clone()
    target[pos_mask == 0] = 1
    target[pos_mask != 0] = 0
    target = target.type("torch.cuda.FloatTensor")

    return target