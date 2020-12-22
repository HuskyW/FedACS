import numpy as np
from models.Update import LocalUpdate
from copy import deepcopy


def estimateBounds(dataset,idxs,args,net,two_bounds=False):
    upper = {}
    lower = {}

    for i in range(len(idxs)):
        newsample = np.ones(1,dtype='int')
        newsample[0] = idxs[i]

        margs = deepcopy(args)
        margs.local_bs = 1
        margs.local_ep = 1

        local = LocalUpdate(args=args, dataset=dataset, idxs=newsample)
        w, loss = local.train(net=deepcopy(net).to(margs.device))

        for k in w.keys():
            wklist = w[k].cpu().numpy().reshape(-1)
            if k not in upper.keys():
                upper[k] = deepcopy(wklist)
                lower[k] = deepcopy(wklist)
            
            combined = np.vstack((upper[k],wklist,lower[k]))
            upper[k] = np.amax(combined,axis=0)
            lower[k] = np.amin(combined,axis=0)
    
    if two_bounds is True:
        return upper,lower
    
    for k in upper.keys():
        upper[k] = upper[k] - lower[k]
        for i in range(len(upper[k])):
            if upper[k][i] == 0.0:
                upper[k][i] = 100000


    return upper