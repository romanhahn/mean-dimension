import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

class RobustNet(nn.Module):
    def __init__(self, Net, y=3, g=0., grate=1e-2, use_center=False):
            super(RobustNet, self).__init__()
            self.replicas = nn.ModuleList([Net() for a in range(y)])
            self.g = g           # coupling
            self.grate = grate   # coupling increase rate
            self.y = y           # number of replicas
            self.center = Net() if use_center else None
            self.Net = Net

    def forward(self, x, split_input=True, concatenate_output=True):
        if not isinstance(x, tuple) and not isinstance(x, list):
            if split_input:
                x = torch.chunk(x, self.y)
            else: # duplicate input
                x = tuple(x for a in range(self.y))

        x = [r(x) for (r,x) in zip(self.replicas, x)]
        if concatenate_output: # recompose
            return torch.cat(x)
        else:
            return x

    def has_center(self):
        return not self.center is None

    # num params per replica
    def num_params(self):
        return sum(p.numel() for p in self.replicas[0].parameters())

    def increase_g(self):
        self.g *= 1 + self.grate

    def coupling_loss(self):
        return self.g * torch.mean(torch.stack(self.sqdistances()))

    # distances with the center of mass
    def sqdistances(self):
        dists = [0.0]*self.y
        if self.has_center():
            for a,r in enumerate(self.replicas):
                for wr, wc in zip(r.parameters(), self.center.parameters()):
                    dists[a] += F.mse_loss(wc, wr, reduction='sum')
        else:
            for wreplicas in zip(*(r.parameters() for r in self.replicas)):
                wc = torch.mean(torch.stack(wreplicas), 0)
                for a, wr in enumerate(wreplicas):
                    dists[a] += F.mse_loss(wc, wr, reduction='sum')
        return dists

    def sqnorms(self):
        sqns = [0.0]*self.y
        for wreplicas in zip(*(s.parameters() for s in self.replicas)):
            for a, wr in enumerate(wreplicas):
                sqns[a] += wr.norm()**2
        return sqns

    def build_center_of_mass(self):
        center = self.Net()
        for wc, *wreplicas in zip(center.parameters(), *(r.parameters() for r in self.replicas)):
            wc.data = torch.mean(torch.stack(wreplicas), 0).data

        for bc, *breplicas in zip(center.buffers(), *(r.buffers() for r in self.replicas)):
            if breplicas[0].dtype == torch.long:
                bc.data = torch.ceil(torch.mean(torch.stack(breplicas).double())).long().data
            else:
                bc.data = torch.mean(torch.stack(breplicas), 0).data

        return center

    def get_or_build_center(self):
        if self.has_center():
            return self.center
        else:
            return self.build_center_of_mass()


class RobustDataLoader():
    def __init__(self, dset, y, concatenate, M=-1, **kwargs):
        if M > 0:
            dset = Subset(dset, range(min(M, len(dset))))
        self.y = y
        self.dls = [DataLoader(dset, **kwargs) for a in range(y)]
        self.concatenate = concatenate
        self.dataset = dset

    def __iter__(self):
        if self.concatenate:
            for xs in zip(*self.dls):
                yield tuple(torch.cat([x[i] for x in xs]) for i in range(len(xs[0])))
        else:
            for xs in zip(*self.dls):
                yield xs

    def __len__(self):
        return len(self.dls[0])

    def single_loader(self):
        return self.dls[0]
