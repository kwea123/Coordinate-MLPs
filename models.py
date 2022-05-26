import torch
from torch import nn
import numpy as np
from einops import rearrange


class MLP(nn.Module):
    def __init__(self, n_in,
                 n_layers=4, n_hidden_units=256,
                 act='relu', act_trainable=False,
                 **kwargs):
        super().__init__()

        layers = []
        for i in range(n_layers):

            if i == 0:
                l = nn.Linear(n_in, n_hidden_units)
            elif 0 < i < n_layers-1:
                l = nn.Linear(n_hidden_units, n_hidden_units)

            if act == 'relu':
                act_ = nn.ReLU(True)
            elif act == 'gaussian':
                act_ = GaussianActivation(a=kwargs['a'], trainable=act_trainable)
            elif act == 'quadratic':
                act_ = QuadraticActivation(a=kwargs['a'], trainable=act_trainable)
            elif act == 'multi-quadratic':
                act_ = MultiQuadraticActivation(a=kwargs['a'], trainable=act_trainable)
            elif act == 'laplacian':
                act_ = LaplacianActivation(a=kwargs['a'], trainable=act_trainable)
            elif act == 'super-gaussian':
                act_ = SuperGaussianActivation(a=kwargs['a'], b=kwargs['b'],
                                               trainable=act_trainable)
            elif act == 'expsin':
                act_ = ExpSinActivation(a=kwargs['a'], trainable=act_trainable)

            if i < n_layers-1:
                layers += [l, act_]
            else:
                layers += [nn.Linear(n_hidden_units, 3), nn.Sigmoid()]

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """
        x: (B, 2) # pixel uv (normalized)
        """
        return self.net(x) # (B, 3) rgb


class PE(nn.Module):
    """
    perform positional encoding
    """
    def __init__(self, P):
        """
        P: (2, F) encoding matrix
        """
        super().__init__()
        self.register_buffer("P", P)

    @property
    def out_dim(self):
        return self.P.shape[1]*2

    def forward(self, x):
        """
        x: (B, 2)
        """
        x_ = 2*np.pi*x @ self.P # (B, F)
        return torch.cat([torch.sin(x_), torch.cos(x_)], 1) # (B, 2*F)


class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.
    
    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the 
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a 
    # hyperparameter.
    
    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of 
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6/self.in_features) / self.omega_0, 
                                             np.sqrt(6/self.in_features) / self.omega_0)
        
    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))
    
    
class Siren(nn.Module):
    def __init__(self, in_features=2, out_features=3,
                 hidden_features=256, hidden_layers=4, outermost_linear=False, 
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()
        
        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, 
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)
                
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, 
                                      is_first=False, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, x):
        return self.net(x)


# different activation functions
class GaussianActivation(nn.Module):
    def __init__(self, a=1., trainable=True):
        super().__init__()
        self.register_parameter('a', nn.Parameter(a*torch.ones(1), trainable))

    def forward(self, x):
        return torch.exp(-x**2/(2*self.a**2))


class QuadraticActivation(nn.Module):
    def __init__(self, a=1., trainable=True):
        super().__init__()
        self.register_parameter('a', nn.Parameter(a*torch.ones(1), trainable))

    def forward(self, x):
        return 1/(1+(self.a*x)**2)


class MultiQuadraticActivation(nn.Module):
    def __init__(self, a=1., trainable=True):
        super().__init__()
        self.register_parameter('a', nn.Parameter(a*torch.ones(1), trainable))

    def forward(self, x):
        return 1/(1+(self.a*x)**2)**0.5


class LaplacianActivation(nn.Module):
    def __init__(self, a=1., trainable=True):
        super().__init__()
        self.register_parameter('a', nn.Parameter(a*torch.ones(1), trainable))

    def forward(self, x):
        return torch.exp(-torch.abs(x)/self.a)


class SuperGaussianActivation(nn.Module):
    def __init__(self, a=1., b=1., trainable=True):
        super().__init__()
        self.register_parameter('a', nn.Parameter(a*torch.ones(1), trainable))
        self.register_parameter('b', nn.Parameter(b*torch.ones(1), trainable))

    def forward(self, x):
        return torch.exp(-x**2/(2*self.a**2))**self.b


class ExpSinActivation(nn.Module):
    def __init__(self, a=1., trainable=True):
        super().__init__()
        self.register_parameter('a', nn.Parameter(a*torch.ones(1), trainable))

    def forward(self, x):
        return torch.exp(-torch.sin(self.a*x))


# from https://github.com/boschresearch/multiplicative-filter-networks/blob/main/mfn/mfn.py
class GaborLayer(nn.Module):
    def __init__(self, in_features, out_features, weight_scale, alpha):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.mu = nn.Parameter(2*torch.rand(1, out_features, in_features)-1)
        self.gamma = nn.Parameter(
            torch.distributions.gamma.Gamma(alpha, 1.0).sample((out_features,)))
        self.linear.weight.data *= weight_scale*self.gamma[:, None]**0.5
        self.linear.bias.data.uniform_(-np.pi, np.pi)

    def forward(self, x):
        D = torch.norm(rearrange(x, 'b d -> b 1 d')-self.mu, dim=-1)**2
        return torch.sin(self.linear(x)) * torch.exp(-0.5*D*self.gamma[None])


class GaborNet(nn.Module):
    def __init__(
        self,
        in_size=2,
        hidden_size=256,
        out_size=3,
        n_layers=4,
        input_scale=256.0,
        alpha=6.0):
        super().__init__()

        self.linear = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(n_layers)]
        )
        self.output_linear = \
            nn.Sequential(nn.Linear(hidden_size, out_size),
                          nn.Sigmoid())

        self.filters = nn.ModuleList(
            [
                GaborLayer(
                    in_size,
                    hidden_size,
                    input_scale / np.sqrt(n_layers + 1),
                    alpha / (n_layers + 1)
                )
                for _ in range(n_layers + 1)
            ]
        )

    def forward(self, x):
        out = self.filters[0](x)
        for i in range(1, len(self.filters)):
            out = self.filters[i](x) * self.linear[i-1](out)
        out = self.output_linear(out)

        return out


# from https://github.com/computational-imaging/bacon/blob/main/modules.py
def mfn_weights_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-(12/num_input)**0.5, (12/num_input)**0.5)


class GaborLayer_Bacon(nn.Module):
    def __init__(self, in_features, out_features, weight_scale, alpha,
                 quantization_interval):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

        self.mu = nn.Parameter(2*torch.rand(1, out_features, in_features)-1)
        self.gamma = nn.Parameter(
            torch.distributions.gamma.Gamma(alpha, 1.0).sample((out_features,)))

        # sample discrete frequencies to ensure coverage
        for i in range(in_features):
            init = torch.randint_like(
                self.linear.weight.data[:, i],
                int(2*weight_scale[i]/quantization_interval)+1)
            init = init * quantization_interval - weight_scale[i]
            self.linear.weight.data[:, i] = init*self.gamma**0.5

        self.linear.weight.requires_grad = False
        self.linear.bias.data.uniform_(-np.pi, np.pi)

    def forward(self, x):
        D = torch.norm(rearrange(x, 'b d -> b 1 d')-self.mu, dim=-1)**2
        return torch.sin(self.linear(x)) * \
               torch.exp(-0.5*D*rearrange(self.gamma, 'o -> 1 o'))


class MultiscaleBACON(nn.Module):
    def __init__(self,
                 in_size=2,
                 hidden_size=256,
                 out_size=3,
                 n_layers=4,
                 alpha=6.0,
                 frequency=(128, 128),
                 quantization_interval=2*np.pi,
                 input_scales=[1/8, 1/8, 1/4, 1/4, 1/4],
                 output_layers=[4]):
        super().__init__()

        self.output_layers = output_layers

        # we need to multiply by this to be able to fit the signal
        if len(input_scales) != n_layers+1:
            raise ValueError('require n+1 scales for n hidden_layers')
        input_scales = [[round((np.pi*freq*s)/quantization_interval) * \
                         quantization_interval
                         for freq in frequency] for s in input_scales]

        self.filters = nn.ModuleList([
                        GaborLayer_Bacon(in_size, hidden_size,
                                         input_scales[i]/np.sqrt(n_layers+1),
                                         alpha/(n_layers+1),
                                         quantization_interval)
                        for i in range(n_layers+1)])
        self.linear = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(n_layers)])
        # linear layers to extract intermediate outputs
        self.output_linear = nn.ModuleList(
            [nn.Sequential(nn.Linear(hidden_size, out_size),
                           nn.Sigmoid()) 
             for _ in range(len(self.output_layers))])
        self.linear.apply(mfn_weights_init)
        self.output_linear.apply(mfn_weights_init)

    def forward(self, x):
        outs = []
        out = self.filters[0](x)
        k = 0
        for i in range(1, len(self.filters)):
            l = self.linear[i-1](out)
            out = self.filters[i](x) * l
            # outs += [l]
            if i in self.output_layers:
                outs += [self.output_linear[k](out)]
                k += 1

        return outs
