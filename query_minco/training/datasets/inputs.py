import torch
CONFIG={"constant_normalized_depth":0.1}

def forward_inputs(samples,condition,device):
    depth=torch.stack([s.depth_input for s in samples]).to(device)
    state=torch.stack([s.state for s in samples]).to(device)
    if condition=="constant":
        depth=depth.clone();depth[:,0].fill_(CONFIG["constant_normalized_depth"]);depth[:,1].fill_(1.)
    elif condition!="normal":raise ValueError("unknown depth condition")
    return {"depth_input":depth,"state":state}
