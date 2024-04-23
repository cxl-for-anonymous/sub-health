import torch.nn as nn
import torch.nn.functional as F
import torch
import os
import numpy as np
import random

from data_reader import load_labeled_patterns


class SINET(nn.Module):
    def __init__(self, n_hour, n_day):
        super(SINET, self).__init__()
        self.hour_multihead = nn.MultiheadAttention(1, 1)
        self.day_multihead = nn.MultiheadAttention(1, 1)
        self.hour_norm = nn.LayerNorm(n_hour)
        self.day_norm = nn.LayerNorm(n_day)
        # self.hour_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=1, nhead=1, batch_first=True),
        #                                           num_layers=1)
        # self.day_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=1, nhead=1, batch_first=True),
        #                                          num_layers=1)
        self.hour_decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=1, nhead=1, batch_first=True),
                                                  num_layers=1)
        self.day_decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=1, nhead=1, batch_first=True),
                                                 num_layers=1)

        self.hour_encoder = nn.Conv1d(1, 1, 3, padding=1)
        self.day_encoder = nn.Conv1d(1, 1, 3, padding=1)

        self.project_hour1 = nn.Linear(n_hour, 1)
        self.project_day1 = nn.Linear(n_day, 1)

        self.hour_thrd = nn.Linear(1, 1)
        self.day_thrd = nn.Linear(1, 1)
        self.reduce = nn.Linear(2, 1)

        self.predict_fc1 = nn.Linear(32, 32)

        self.dropout = nn.Dropout(p=0.3)

        self.relu = nn.ReLU()

    def forward(self, x_hour, x_day, y_hour, y_day, label=1, tmp=(None, None)):
        x_hour_detach = x_hour.detach()
        x_day_detach = x_day.detach()

        y_day_detach = y_day.detach()
        y_hour_detach = y_hour.detach()

        x_hour_decoder = self.hour_decoder(x_hour, y_hour_detach)
        x_day_decoder = self.day_decoder(x_day, y_day_detach)

        # euclidean
        hour_dis = torch.abs(torch.sub(x_hour, y_hour))
        day_dis = torch.abs(torch.sub(x_day, y_day))

        # hour_dis = torch.pow(hour_dis,2)
        # day_dis = torch.pow(day_dis,2)

        # dot product
        # hour_dis = torch.mul(x_hour, y_hour) / (x_hour.shape[1]-1)
        # day_dis = torch.mul(x_day, y_day) / (x_day.shape[1]-1)

        x_hour_atn, _ = self.hour_multihead(x_hour, hour_dis, hour_dis, need_weights=False)
        x_day_atn, _ = self.day_multihead(x_day, day_dis, day_dis, need_weights=False)

        # y_hour_atn, _ = self.hour_multihead(y_hour, hour_dis, hour_dis, need_weights=False)
        # y_day_atn, _ = self.day_multihead(y_day, day_dis, day_dis, need_weights=False)

        # y_hour_atn = y_hour_atn.detach()
        # y_day_atn = y_day_atn.detach()

        hour_atn = x_hour_atn
        day_atn = x_day_atn

        # hour_atn = self.relu(self.hour_mid_thrd(x_hour_atn))
        # day_atn = self.relu(self.day_mid_thrd(x_day_atn))

        # hour_atn = torch.div(x_hour_atn, torch.add(x_hour,1.0))
        # day_atn = torch.div(x_day_atn, torch.add(x_day,1.0))
        # hour_atn = torch.add(x_hour_atn, y_hour_atn)
        # day_atn = torch.add(x_day_atn, y_day_atn)

        hour_atn = hour_atn.squeeze(dim=2)
        day_atn = day_atn.squeeze(dim=2)

        # hour_atn = self.hour_norm(hour_atn)
        # day_atn = self.day_norm(day_atn)

        hour_dis_val = torch.sum(hour_atn, dim=1).unsqueeze(dim=1)
        day_dis_val = torch.sum(day_atn, dim=1).unsqueeze(dim=1)

        hour_dis_val = self.relu(self.hour_thrd(hour_dis_val))
        day_dis_val = self.relu(self.day_thrd(day_dis_val))

        total_dis = torch.cat((hour_dis_val, day_dis_val), dim=1)

        agg_dis = F.sigmoid(self.reduce(total_dis))

        # agg_dis = F.softmax(agg_dis, dim=1)

        return agg_dis


def train(period_num, day_num, period_support, day_support, train_alert_pair, model_folder_path, epoch=None, model_name=None):
    random.seed(8)
    torch.random.manual_seed(8)
    model = SINET(period_num, day_num)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.8)
    mse_criteon = nn.MSELoss()
    train_losses = []
    train_alert_pair = list(train_alert_pair)
    random.shuffle(train_alert_pair)
    if not epoch:
        epoch_range = range(1000)
    else:
        epoch_range = range(epoch)
    for e_id in epoch_range:
        e_loss = 0
        for batch_n, (label, alert_id1, alert_id2) in enumerate(train_alert_pair):
            x_hour = np.array(period_support[alert_id1])
            x_day = np.array(day_support[alert_id1])
            x_hour = torch.Tensor(x_hour.transpose()).unsqueeze(dim=0).unsqueeze(dim=2)
            x_day = torch.Tensor(x_day.transpose()).unsqueeze(dim=0).unsqueeze(dim=2)
            y_hour = np.array(period_support[alert_id2])
            y_day = np.array(day_support[alert_id2])
            y_hour = torch.Tensor(y_hour.transpose()).unsqueeze(dim=0).unsqueeze(dim=2)
            y_day = torch.Tensor(y_day.transpose()).unsqueeze(dim=0).unsqueeze(dim=2)
            z = model(x_hour, x_day, y_hour, y_day, label, tmp=(alert_id1, alert_id2))
            if label == 1:
                loss = mse_criteon(z, torch.Tensor([1]).unsqueeze(dim=0))
            else:
                loss = mse_criteon(z, torch.Tensor([0]).unsqueeze(dim=0))
            e_loss += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # scheduler.step()
        avg_loss = e_loss * 1.0 / len(train_alert_pair)
        if model_name and epoch:
            model_path = os.path.join(model_folder_path, 'network_'+model_name+'.pth')
        else:
            model_path = os.path.join(model_folder_path,
                                  'network_epoch' + str(e_id) + '_loss' + str(round(avg_loss, 5)) + '.pth')
        if not epoch or e_id == epoch-1:
            torch.save(model.state_dict(), model_path)
            print('epoch', e_id, 'train size', len(train_alert_pair), 'epoch total loss', e_loss, 'epoch avg loss',
                  avg_loss)
        train_losses.append((e_id, len(train_alert_pair), e_loss, avg_loss))
    if epoch:
        return model_path
    else:
        return train_losses


def evaluate(model, evaluate_alert_pair, hour_support, day_support):
    with torch.no_grad():
        total_loss = 0
        for batch_n, (label, alert_id1, alert_id2) in enumerate(evaluate_alert_pair):
            x_hour = np.array(hour_support[alert_id1])
            x_day = np.array(day_support[alert_id1])
            x_hour = torch.Tensor(x_hour.transpose()).unsqueeze(dim=0).unsqueeze(dim=2)
            x_day = torch.Tensor(x_day.transpose()).unsqueeze(dim=0).unsqueeze(dim=2)
            y_hour = np.array(hour_support[alert_id2])
            y_day = np.array(day_support[alert_id2])
            y_hour = torch.Tensor(y_hour.transpose()).unsqueeze(dim=0).unsqueeze(dim=2)
            y_day = torch.Tensor(y_day.transpose()).unsqueeze(dim=0).unsqueeze(dim=2)

            loss = model(x_hour, x_day, y_hour, y_day, label).squeeze().item()
            if loss < 0.65 and label == 1:
                print(label, alert_id1, alert_id2)
            if loss > 0.65 and label != 1:
                print(label, alert_id1, alert_id2)
    return total_loss


def experiment(n_hour, n_day, model_path, period_support, day_support, sim_thrd, sub_healthy_alerts, normal_alerts,
               subheal_hours, train_alert_pair, labeled_pair_patterns_file_path):
    model = SINET(n_hour, n_day)
    model.load_state_dict(torch.load(model_path))
    labeled_right_pair_patterns, labeled_wrong_pair_patterns = load_labeled_patterns(labeled_pair_patterns_file_path)
    right_pair_patterns = set()
    wrong_pair_patterns = set()
    unknown_pair_patterns = set()
    similarity = dict()
    with torch.no_grad():
        for i in range(len(sub_healthy_alerts)):
            alert_id1 = sub_healthy_alerts[i]
            if alert_id1 == 1:
                continue
            if alert_id1 in normal_alerts:
                continue
            for j in range(i + 1, len(sub_healthy_alerts)):
                alert_id2 = sub_healthy_alerts[j]
                if alert_id2 in normal_alerts:
                    continue
                if len(subheal_hours[alert_id1].intersection(subheal_hours[alert_id2])) == 0:
                    continue
                if (1, alert_id1, alert_id2) in train_alert_pair or (1, alert_id2, alert_id1) in train_alert_pair:
                    continue
                x_period = np.array(period_support[alert_id1])
                x_day = np.array(day_support[alert_id1])
                x_period = torch.Tensor(x_period.transpose()).unsqueeze(dim=0).unsqueeze(dim=2)
                x_day = torch.Tensor(x_day.transpose()).unsqueeze(dim=0).unsqueeze(dim=2)
                y_period = np.array(period_support[alert_id2])
                y_day = np.array(day_support[alert_id2])
                y_period = torch.Tensor(y_period.transpose()).unsqueeze(dim=0).unsqueeze(dim=2)
                y_day = torch.Tensor(y_day.transpose()).unsqueeze(dim=0).unsqueeze(dim=2)
                sim = model(x_period, x_day, y_period, y_day, None, tmp=(alert_id1, alert_id2)).squeeze().item()
                if sim > sim_thrd:
                    pattern = (alert_id1, alert_id2)
                    similarity[pattern] = sim
                    if pattern in labeled_right_pair_patterns:
                        right_pair_patterns.add(pattern)
                    elif pattern in labeled_wrong_pair_patterns:
                        wrong_pair_patterns.add(pattern)
                    else:
                        unknown_pair_patterns.add(pattern)
    return right_pair_patterns, wrong_pair_patterns, unknown_pair_patterns, similarity


def experiment_no_judge(n_hour, n_day, model_path, period_support, day_support, sub_healthy_alerts, normal_alerts,
               subheal_hours, train_alert_pair):
    model = SINET(n_hour, n_day)
    model.load_state_dict(torch.load(model_path))
    similarity = dict()
    with torch.no_grad():
        for i in range(len(sub_healthy_alerts)):
            alert_id1 = sub_healthy_alerts[i]
            if alert_id1 == 1:
                continue
            if alert_id1 in normal_alerts:
                continue
            for j in range(i + 1, len(sub_healthy_alerts)):
                alert_id2 = sub_healthy_alerts[j]
                if alert_id2 in normal_alerts:
                    continue
                if len(subheal_hours[alert_id1].intersection(subheal_hours[alert_id2])) == 0:
                    continue
                if (1, alert_id1, alert_id2) in train_alert_pair or (1, alert_id2, alert_id1) in train_alert_pair:
                    continue
                x_period = np.array(period_support[alert_id1])
                x_day = np.array(day_support[alert_id1])
                x_period = torch.Tensor(x_period.transpose()).unsqueeze(dim=0).unsqueeze(dim=2)
                x_day = torch.Tensor(x_day.transpose()).unsqueeze(dim=0).unsqueeze(dim=2)
                y_period = np.array(period_support[alert_id2])
                y_day = np.array(day_support[alert_id2])
                y_period = torch.Tensor(y_period.transpose()).unsqueeze(dim=0).unsqueeze(dim=2)
                y_day = torch.Tensor(y_day.transpose()).unsqueeze(dim=0).unsqueeze(dim=2)
                sim = model(x_period, x_day, y_period, y_day, None, tmp=(alert_id1, alert_id2)).squeeze().item()
                pattern = (alert_id1, alert_id2)
                similarity[pattern] = sim
    return similarity
