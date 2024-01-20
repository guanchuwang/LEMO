import torch
import torch.nn as nn


class mlp_Gaussian(nn.Module):

    def __init__(self, input_dim, output_dim, layer_num=1, hidden_dim=64, activation=None):
        super(mlp_Gaussian, self).__init__()

        self.mlp = nn.ModuleList()
        self.layer_num = layer_num
        self.activation = eval(activation) # nn.ReLU(inplace=True)
        # self.dropout = nn.Dropout(p=0.5)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.activation_str = activation


        if layer_num == 1:
            self.layer_mean = nn.Linear(hidden_dim, output_dim)
            self.layer_log_std = nn.Linear(hidden_dim, output_dim)

        else:
            for layer_index in range(layer_num - 1):
                if layer_index == 0:
                    layer1 = nn.Linear(input_dim, hidden_dim)

                else:
                    layer1 = nn.Linear(hidden_dim, hidden_dim)

                # print(layer1.weight.shape)
                # layer1.weight.data.mul_(1e-3)
                # nn.init.constant_(layer1.bias.data, 0.)
                self.mlp.append(layer1)

            self.layer_mean = nn.Linear(hidden_dim, output_dim)
            self.layer_log_std = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):

        for layer_index in range(self.layer_num - 1):
            layer = self.mlp[layer_index]
            # print(layer.weight.shape)
            if self.activation == None:
                x = layer(x)
                # x = self.dropout(x)
            else:
                x = layer(x)
                # x = self.dropout(x)
                x = self.activation(x)

            # print(x)

        output_mean = self.layer_mean(x)
        output_log_std = self.layer_log_std(x)
        # print(layer_lst(x))

        return output_mean, output_log_std

    # def forward_softmax(self, x):
        # return torch.softmax(self.forward(x), dim=1)

    # def forward_1(self, x):
    #     return self.forward(x)[:, 1].unsqueeze(dim=1)
    #
    # def forward_softmax_1(self, x):
    #     return self.forward_softmax(x)[:, 1].unsqueeze(dim=1)
    #
    # def forward_wo_sigmoid(self, x):
    #     return self.forward_softmax(x)[:, 1].unsqueeze(dim=1) - self.forward_softmax(x)[:, 0].unsqueeze(dim=1)
    #
    # @torch.no_grad()
    # def predict_proba(self, x):
    #
    #     return torch.softmax(self.forward(x), dim=1)[:, 1] # .unsqueeze(dim=1)

