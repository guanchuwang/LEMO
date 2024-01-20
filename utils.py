import torch
import sys

from model.model import mlp, Model_for_shap
from model.explainer_model import mlp_Gaussian
from data import XAIDataSet
from torch.utils.data import DataLoader
import time

def save_checkpoint(fname, **kwargs):

    checkpoint = {}

    for key, value in kwargs.items():
        checkpoint[key] = value
        # setattr(self, key, value)

    torch.save(checkpoint, fname)


def load_checkpoint(fname):

    return torch.load(fname)


@torch.no_grad()
def val_prepare(target_checkpoint, explaienr_checkpoint, batch_size=256):

    dense_feat_index = target_checkpoint["dense_feat_index"]
    sparse_feat_index = target_checkpoint["sparse_feat_index"]
    cate_attrib_book = target_checkpoint["cate_attrib_book"]

    model = mlp(input_dim=target_checkpoint["input_dim"],
                output_dim=target_checkpoint["output_dim"],
                layer_num=target_checkpoint["layer_num"],
                hidden_dim=target_checkpoint["hidden_dim"],
                activation=target_checkpoint["activation"])

    model.load_state_dict(target_checkpoint["state_dict"])

    model_for_shap = Model_for_shap(model, dense_feat_index, sparse_feat_index, cate_attrib_book)

    x_test = target_checkpoint["test_data_x"]
    y_test = target_checkpoint["test_data_y"]
    z_test = target_checkpoint["test_data_z"]
    reference_test = target_checkpoint["reference_test"]

    shapley_value_gt = target_checkpoint["test_shapley_value"]
    shapley_ranking_gt = target_checkpoint["test_shapley_ranking"]

    explainer = mlp_Gaussian(input_dim=explaienr_checkpoint["input_dim"],
                             output_dim=explaienr_checkpoint["output_dim"],
                             layer_num=explaienr_checkpoint["layer_num"],
                             hidden_dim=explaienr_checkpoint["hidden_dim"],
                             activation=explaienr_checkpoint["activation"])

    explainer.load_state_dict(explaienr_checkpoint["state_dict"])
    # print(explainer.state_dict())

    test_loader = DataLoader(XAIDataSet(x_test, y_test, z_test, shapley_value_gt, shapley_ranking_gt), batch_size=batch_size,
                             shuffle=False, drop_last=False, pin_memory=True)

    return model_for_shap, explainer, test_loader, reference_test


@torch.no_grad()
def add_bias(phi, target_model, x, reference):
    # ipdb.set_trace()
    y1 = target_model(x)
    class_idx = y1.argmax(dim=1, keepdim=True)
    y0 = target_model(reference.unsqueeze(dim=0))
    y0 = y0.repeat((y1.shape[0], 1))
    y1 = torch.gather(y1, dim=1, index=class_idx)
    y0 = torch.gather(y0, dim=1, index=class_idx)

    return phi - (phi.sum(dim=1, keepdim=True) - (y1 - y0)) / reference.shape[-1]


def sv_norm(sv):
    return sv / torch.norm(sv, p=2, dim=-1, keepdim=True)


@torch.no_grad()
def val(data_loader, model, target_model, reference):
    error_total = 0
    error_buf = []
    total_time = 0

    for index, (data, index_buf) in enumerate(data_loader):
        x_input, _, x, sv_gt, sv_rk = data

        t0 = time.time()
        y_mean, y_log_std = model(x_input)
        total_time += time.time() - t0

        y_hat = add_bias(y_mean, target_model, x, reference)
        # error_std=torch.norm((y_hat_std - sv_gt_std), p=2, dim=-1).sum(dim=0)
        # error_std = torch.sqrt(torch.square(y_hat_std - sv_gt_std).sum(dim=-1)).sum(dim=0)
        error = torch.sqrt(torch.square(y_hat - sv_gt).sum(dim=-1))
        # error_std=torch.square(y_hat_std - sv_gt_std).mean(dim=-1).sum(dim=0)
        error_buf.append(error)

        # if index == 0:
        #     print(x_input[0])
        #     print(sv_gt[0])
        #     print(y_hat[0])
        #     print(error[0])

    error_buf = torch.cat(error_buf, dim=0)
    # print(error_buf.shape)
    # if epoch >= 5:
    # ipdb.set_trace()
    avg_time = total_time/len(data_loader.dataset)

    return error_buf, error_buf.mean(), error_buf.std(), avg_time


@torch.no_grad()
def feature_l2error(data_loader, model, target_model, reference):
    error_total = 0
    error_buf = []
    std_pred_buf = []

    for (data, index_buf) in data_loader:
        x_input, _, x, sv_gt, sv_rk = data

        y_mean, y_log_std = model(x_input)

        y_hat = add_bias(y_mean, target_model, x, reference)
        # error_std=torch.norm((y_hat_std - sv_gt_std), p=2, dim=-1).sum(dim=0)
        # error_std = torch.sqrt(torch.square(y_hat_std - sv_gt_std).sum(dim=-1)).sum(dim=0)
        error = torch.abs(y_hat - sv_gt)
        std_pred = torch.exp(y_log_std)
        # error_std=torch.square(y_hat_std - sv_gt_std).mean(dim=-1).sum(dim=0)
        error_buf.append(error)
        std_pred_buf.append(std_pred)

    error_buf = torch.cat(error_buf, dim=0)
    std_pred_buf = torch.cat(std_pred_buf, dim=0)

    return error_buf, std_pred_buf

# @torch.no_grad()
# def val(data_loader, model, target_model, reference):
#     error_total = 0
#     for (data, index_buf) in data_loader:
#         x_input, _, x, sv_gt, sv_rk = data
#         y_mean, y_log_std = model(x_input)
#         y_hat = add_bias(y_mean, target_model, x, reference)
#         error_std = torch.sqrt(torch.square(y_hat - sv_gt).sum(dim=-1)).sum(dim=0)
#         error_total += error_std
#
#     error_avg = error_total / data_loader.dataset.dataset.tensors[2].shape[0]
#     return error_avg, error_avg, 0


