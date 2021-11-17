
import torch
import torch.nn as nn
import math


class SparseModule(nn.Module):
    """
    Args:

        module (nn.Module): module to be pruned
        pruning_rate (float): pruning rate (0.0: fully-connected, 1.0: fully-pruned)
        init_mode (str): the parameter distribution for weights
        ignore_params (list): parameter names to be ignored

    Example:

        >>> net = nn.Linear(7,5,bias=False)   # any network
        >>> sparse_net = SparseModule(net, 0.5)  # network with randomly initialized weights & masks
        >>> output = sparse_net(input)   # forward computation by masked net
    """

    def __init__(self,
                 module,
                 pruning_rate,
                 init_mode="kaiming_uniform",
                 ignore_params=[]):
        super().__init__()

        self._module = module
        self.pruning_rate = pruning_rate
        self.init_mode = init_mode
        self.ignore_params = ignore_params

        self.ones = dict()
        self.zeros = dict()
        self.param_twins = dict()

        self.all_module_params = []
        for m_name, m in self._module.named_modules():
            for p_name, p in m.named_parameters(recurse=False):
                if any([pat in m_name + '.' + p_name for pat in self.ignore_params]):
                    pass
                else:
                    self.all_module_params.append((m, m_name, p_name))

        for m, m_name, p_name in self.all_module_params:
            weight = getattr(m, p_name)
            self.init_param_(weight, init_mode=self.init_mode)
            del m._parameters[p_name]
            m.register_buffer(p_name, weight.data)
            m.register_buffer(p_name + '_before_pruned', weight.data)

            score = nn.Parameter(torch.ones(weight.size()))
            self.init_param_(score, init_mode='kaiming_uniform')
            m.register_parameter(p_name + '_score', score)

            self.ones[m_name + '.' + p_name] = torch.ones(weight.size())
            self.zeros[m_name + '.' + p_name] = torch.zeros(weight.size())
            self.param_twins[m_name + '.' + p_name] = torch.zeros(weight.size())

    def _get_mask(self, m, m_name, p_name):
        score = m._parameters[p_name + '_score']
        device = score.device

        zeros = self.zeros[m_name + '.' + p_name].to(device)
        ones = self.ones[m_name + '.' + p_name].to(device)
        mask = GetBinaryMask.apply(score, self.pruning_rate, zeros, ones)
        return mask

    def forward(self, *args, **kwargs):
        for m, m_name, p_name in self.all_module_params:
            weight = m._buffers[p_name + '_before_pruned']
            mask = self._get_mask(m, m_name, p_name)
            weight = weight.to(mask.device)
            pruned_weight = mask * weight
            setattr(m, p_name, pruned_weight)

        if issubclass(type(self._module), torch.nn.RNNBase):
            self._module.flatten_parameters()

        return self._module.forward(*args, **kwargs)

    def init_param_(self, param, init_mode=None):
        if init_mode == 'kaiming_normal':
            nn.init.kaiming_normal_(param, mode="fan_in", nonlinearity="relu")
        elif init_mode == 'uniform(-1,1)':
            nn.init.uniform_(param, a=-1, b=1)
        elif init_mode == 'kaiming_uniform':
            nn.init.kaiming_uniform_(param, mode='fan_in', nonlinearity='relu')
        elif init_mode == 'signed_constant':
            fan = nn.init._calculate_correct_fan(param, 'fan_in')
            gain = nn.init.calculate_gain('relu')
            std = gain / math.sqrt(fan)
            nn.init.kaiming_normal_(param)    # use only its sign
            param.data = param.data.sign() * std
        else:
            raise NotImplementedError


# This class is implemented based on github.com/allenai/hidden-networks
class GetBinaryMask(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, sparsity, zeros, ones):
        k_val = percentile(scores, sparsity*100)
        out = torch.where(scores < k_val, zeros.to(scores.device), ones.to(scores.device))
        return out

    @staticmethod
    def backward(ctx, g):
        return g, None, None, None

def percentile(t, q):
    k = 1 + round(.01 * float(q) * (t.numel() - 1))
    return t.view(-1).kthvalue(k).values.item()


if __name__ == "__main__":
    import torch.optim as optim
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

    # =======================
    #   Test for nn.Linear
    # =======================
    linear = nn.Linear(7,5,bias=False)
    model = SparseModule(linear, 0.8)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.1) 
    criterion = nn.MSELoss()

    for i in range(10):
        t_before = model._module.weight_before_pruned.clone()
        s_before = model._module.weight_score.clone()

        optimizer.zero_grad()
        input = torch.randn(3,7).to(device)
        target = torch.randn(3,5).to(device)
        loss = criterion(model(input), target)
        loss.backward()
        optimizer.step()

        t_after = model._module.weight_before_pruned.clone()
        s_after = model._module.weight_score.clone()
        assert (t_before - t_after).sum().item() == 0.0
        assert (s_before - s_after).sum().item() != 0.0

    # =======================
    #   Test for nn.LSTM
    # =======================
    bidirectional = True
    D = 2 if bidirectional else 1
    num_layers = 2
    lstm = nn.LSTM(7, 5,
                   bias=False, batch_first=True,
                   num_layers=num_layers,
                   bidirectional=bidirectional)
    all_param_names = [name for name, _ in lstm.named_parameters()]
    model = SparseModule(lstm, 0.8, init_mode='signed_constant')
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.1) 
    criterion = nn.MSELoss()

    L = 10
    before_weights = dict()
    before_scores = dict()
    for i in range(10):
        for name in all_param_names:
            before_weights[name] = getattr(model._module, name + "_before_pruned").clone().to(device)
            before_scores[name] = getattr(model._module, name + "_score").clone().to(device)

        optimizer.zero_grad()
        input = torch.randn(3,L,7).to(device)
        target = torch.randn(D*num_layers,3,5).to(device)
        output, (h, c) = model(input)
        loss = criterion(h, target)
        loss.backward()
        optimizer.step()

        for name in all_param_names:
            after_weight = getattr(model._module, name + "_before_pruned").clone().to(device)
            after_score = getattr(model._module, name + "_score").clone().to(device)
            assert (after_weight - before_weights[name]).abs().sum().item() == 0.0
            assert (after_score - before_scores[name]).abs().sum().item() != 0.0

