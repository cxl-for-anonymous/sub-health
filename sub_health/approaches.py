from matplotlib import pyplot as plt
import numpy as np
import os
import seaborn as sns
import model as si_net
from datetime import datetime
from detect import *
import csv


def plt_naive_distribution(alerts, output_folder_path, id_set=None, pre='', single_pre=None):
    os.makedirs(output_folder_path, exist_ok=True)
    occurrence = dict()
    # print(alerts[0].timestamp.year, alerts[0].timestamp.month, alerts[0].timestamp.day, alerts[0].timestamp.hour)
    first_day = pd.Timestamp(alerts[0].timestamp.year, alerts[0].timestamp.month, alerts[0].timestamp.day,
                             alerts[0].timestamp.hour)
    last_day = pd.Timestamp(alerts[-1].timestamp.year, alerts[-1].timestamp.month, alerts[-1].timestamp.day,
                            alerts[-1].timestamp.hour)
    for alert in alerts:
        id = alert.id
        day = (alert.timestamp.year, alert.timestamp.month, alert.timestamp.day)
        hour = alert.timestamp.hour
        if id not in occurrence:
            occurrence[id] = defaultdict(int)
        occurrence[id][(day, hour)] += 1
    distribution = defaultdict(list)
    day_list = []
    flag = False
    hour_list = [h for h in range(0, 24)]

    for id in occurrence:
        if id_set and id not in id_set:
            continue
        if len(occurrence[id]) <= 1:
            continue
        cur_day = first_day
        while cur_day <= last_day:
            if not flag:
                day_list.append(str(cur_day.year).replace('20', '') + '/' + str(cur_day.month) + '/' + str(cur_day.day))
            day_occur = []
            for cur_hour in range(0, 24):
                day_hour = ((cur_day.year, cur_day.month, cur_day.day), cur_hour)
                if day_hour in occurrence[id]:
                    # day_occur.append(occurrence[id][day_hour])
                    day_occur.append(1)
                else:
                    day_occur.append(0)
            cur_day += pd.Timedelta(days=1)
            distribution[id].append(day_occur)
        flag = True
    plt.rc('font', family='Times New Roman')
    hour_labels = [0, '', 2, '', 4, '', 6, '', 8, '', 10, '', 12, '', 14, '', 16, '', 18, '', 20, '', 22, '']
    day_labels = []
    for i in range(len(day_list)):
        if i % 3 == 0:
            day_labels.append(day_list[i])
        else:
            day_labels.append('')

    for id in distribution:
        plt.close("all")
        fig, ax = plt.subplots(figsize=(6, 4.5))
        data = np.array(distribution[id]).transpose()

        # exit(1)
        ax = sns.heatmap(pd.DataFrame(data=data, index=hour_labels, columns=day_labels), vmin=0, vmax=1,
                         cmap=plt.get_cmap('Blues'), linewidths=0.5, linecolor='w', cbar=False, center=0.5)
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom + 0.5, top - 0.5)
        plt.xticks(fontsize='xx-large', rotation=-45)
        plt.yticks(fontsize='xx-large', rotation=0)
        plt.xlabel('Day', fontsize='xx-large', color='k')
        plt.ylabel('Time Period', fontsize='xx-large', color='k')

        for _, spine in ax.spines.items():
            spine.set_visible(True)
        plt.tight_layout()
        if single_pre:
            plt.savefig(os.path.join(output_folder_path, str(single_pre[id]) + '_' + pre + '_' + str(id) + '_.png'),
                        bbox_inches='tight', dpi = 512)
        else:
            plt.savefig(os.path.join(output_folder_path, pre + str(id) + '.png'), bbox_inches='tight', dpi = 512)
    return distribution


def euclidean(distribution, normal_alerts, period_support, subheal_hours, euclidean_dis_thrd):
    subheal_alerts = set(period_support.keys()).difference(normal_alerts)
    subheal_alerts = list(subheal_alerts)
    alert_euclid_dis = dict()
    euclid_vals = []
    for i in range(len(subheal_alerts)):
        alert_id = subheal_alerts[i]
        if alert_id not in alert_euclid_dis:
            alert_euclid_dis[alert_id] = dict()
        for j in range(i + 1, len(subheal_alerts)):
            alert_id2 = subheal_alerts[j]
            if len(subheal_hours[alert_id].intersection(subheal_hours[alert_id2])) == 0:
                continue
            if alert_id2 not in alert_euclid_dis:
                alert_euclid_dis[alert_id2] = dict()
            alert_euclid_dis[alert_id][alert_id2] = euclidean_distance(distribution[alert_id], distribution[alert_id2])
            alert_euclid_dis[alert_id2][alert_id] = alert_euclid_dis[alert_id][alert_id2]
            euclid_vals.append(alert_euclid_dis[alert_id][alert_id2])
    euclid_vals.sort()

    relations = dict()
    pair_patterns = set()
    for i in range(len(subheal_alerts)):
        for j in range(i + 1, len(subheal_alerts)):
            if len(subheal_hours[subheal_alerts[i]].intersection(subheal_hours[subheal_alerts[j]])) == 0:
                continue
            euclid_dis = alert_euclid_dis[subheal_alerts[i]][subheal_alerts[j]]
            if euclid_dis <= euclidean_dis_thrd:
                if subheal_alerts[i] not in relations:
                    relations[subheal_alerts[i]] = set()
                if subheal_alerts[j] not in relations:
                    relations[subheal_alerts[j]] = set()
                relations[subheal_alerts[i]].add(subheal_alerts[j])
                relations[subheal_alerts[j]].add(subheal_alerts[i])
                pair_patterns.add((subheal_alerts[i], subheal_alerts[j]))
    return pair_patterns


def euclidean_distance(p, q):
    dis = 0
    for i in range(len(p)):
        for j in range(len(p[i])):
            dis += abs(p[i][j] - q[i][j])
    return dis


def cos_similarity(x, y):
    sim = np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
    sim = 0.5 + sim * 0.5
    return sim


def cos(normal_alerts, subheal_hours, period_sim_thrd, day_sim_thrd, period_support, day_support):
    # Z-score
    period_support = z_normalize(period_support)
    day_support = z_normalize(day_support)

    subheal_alerts = set(period_support.keys()).difference(normal_alerts)
    subheal_alerts = list(subheal_alerts)
    hour_cos_sim = dict()
    day_cos_sim = dict()
    hou_cos_val = []
    day_cos_val = []
    for i in range(len(subheal_alerts)):
        alert_id = subheal_alerts[i]
        if alert_id not in hour_cos_sim:
            hour_cos_sim[alert_id] = dict()
        if alert_id not in day_cos_sim:
            day_cos_sim[alert_id] = dict()
        for j in range(i + 1, len(subheal_alerts)):
            alert_id2 = subheal_alerts[j]
            if len(subheal_hours[alert_id].intersection(subheal_hours[alert_id2])) == 0:
                continue
            if alert_id2 not in hour_cos_sim:
                hour_cos_sim[alert_id2] = dict()
            hour_cos_sim[alert_id][alert_id2] = cos_similarity(period_support[alert_id], period_support[alert_id2])
            hour_cos_sim[alert_id2][alert_id] = hour_cos_sim[alert_id][alert_id2]
            hou_cos_val.append(hour_cos_sim[alert_id][alert_id2])
            if alert_id2 not in day_cos_sim:
                day_cos_sim[alert_id2] = dict()
            day_cos_sim[alert_id][alert_id2] = cos_similarity(day_support[alert_id], day_support[alert_id2])
            day_cos_sim[alert_id2][alert_id] = day_cos_sim[alert_id][alert_id2]
            day_cos_val.append(day_cos_sim[alert_id][alert_id2])
    hou_cos_val.sort()
    day_cos_val.sort()

    relations = dict()
    pair_patterns = set()
    for i in range(len(subheal_alerts)):
        for j in range(i + 1, len(subheal_alerts)):
            if len(subheal_hours[subheal_alerts[i]].intersection(subheal_hours[subheal_alerts[j]])) == 0:
                continue
            hour_sim = hour_cos_sim[subheal_alerts[i]][subheal_alerts[j]]
            day_sim = day_cos_sim[subheal_alerts[i]][subheal_alerts[j]]
            if hour_sim >= period_sim_thrd and day_sim >= day_sim_thrd:
                if subheal_alerts[i] not in relations:
                    relations[subheal_alerts[i]] = set()
                if subheal_alerts[j] not in relations:
                    relations[subheal_alerts[j]] = set()
                relations[subheal_alerts[i]].add(subheal_alerts[j])
                relations[subheal_alerts[j]].add(subheal_alerts[i])
                pair_patterns.add((subheal_alerts[i], subheal_alerts[j]))
    return pair_patterns


def z_normalize(alert_support):
    for alert_id in alert_support:
        alert_support[alert_id] = np.array(alert_support[alert_id])
        if alert_support[alert_id].any():
            std = np.std(alert_support[alert_id])
            alert_support[alert_id] = (alert_support[alert_id] - np.mean(alert_support[alert_id]))
            if std > 0:
                alert_support[alert_id] /= std
    return alert_support


def dtw(normal_alerts, subheal_hours, period_dtw_dis_thrd, day_dtw_dis_thrd, period_support, day_support):
    # Z-score
    period_support = z_normalize(period_support)
    day_support = z_normalize(day_support)

    subheal_alerts = set(period_support.keys()).difference(normal_alerts)
    subheal_alerts = list(subheal_alerts)
    hour_dtw_dis = dict()
    day_dtw_dis = dict()
    hour_dtw_vals = []
    day_dtw_vals = []
    for i in range(len(subheal_alerts)):
        alert_id = subheal_alerts[i]
        if alert_id not in hour_dtw_dis:
            hour_dtw_dis[alert_id] = dict()
        if alert_id not in day_dtw_dis:
            day_dtw_dis[alert_id] = dict()
        for j in range(i + 1, len(subheal_alerts)):
            alert_id2 = subheal_alerts[j]
            if len(subheal_hours[alert_id].intersection(subheal_hours[alert_id2])) == 0:
                continue
            if alert_id2 not in hour_dtw_dis:
                hour_dtw_dis[alert_id2] = dict()
            hour_dtw_dis[alert_id][alert_id2] = dtw_distance(period_support[alert_id], period_support[alert_id2])
            hour_dtw_dis[alert_id2][alert_id] = hour_dtw_dis[alert_id][alert_id2]
            hour_dtw_vals.append(hour_dtw_dis[alert_id][alert_id2])
            if alert_id2 not in day_dtw_dis:
                day_dtw_dis[alert_id2] = dict()
            day_dtw_dis[alert_id][alert_id2] = dtw_distance(day_support[alert_id], day_support[alert_id2])
            day_dtw_dis[alert_id2][alert_id] = day_dtw_dis[alert_id][alert_id2]
            day_dtw_vals.append(day_dtw_dis[alert_id][alert_id2])
    hour_dtw_vals.sort()
    day_dtw_vals.sort()

    relations = dict()
    pair_patterns = set()
    for i in range(len(subheal_alerts)):
        for j in range(i + 1, len(subheal_alerts)):
            if len(subheal_hours[subheal_alerts[i]].intersection(subheal_hours[subheal_alerts[j]])) == 0:
                continue
            hour_dis = hour_dtw_dis[subheal_alerts[i]][subheal_alerts[j]]
            day_dis = day_dtw_dis[subheal_alerts[i]][subheal_alerts[j]]
            if hour_dis <= period_dtw_dis_thrd and day_dis <= day_dtw_dis_thrd:
                if subheal_alerts[i] not in relations:
                    relations[subheal_alerts[i]] = set()
                if subheal_alerts[j] not in relations:
                    relations[subheal_alerts[j]] = set()
                relations[subheal_alerts[i]].add(subheal_alerts[j])
                relations[subheal_alerts[j]].add(subheal_alerts[i])
                pair_patterns.add((subheal_alerts[i], subheal_alerts[j]))
    return pair_patterns


def jaccard_similarity(x, y):
    sim = len(x.intersection(y)) * 1.0 / len(x.union(y))
    return sim


def jaccard(distribution, period_support, normal_alerts, subheal_periods, period_jaccard_thrd, day_jaccard_thrd):
    subheal_days = dict()
    for alert_id in subheal_periods:
        subheal_days[alert_id] = set()
        for hour in subheal_periods[alert_id]:
            for day in range(len(distribution[alert_id])):
                if distribution[alert_id][day][hour] > 0:
                    subheal_days[alert_id].add(day)
                    break
        if len(subheal_days) == 0:
            subheal_days.pop(alert_id)

    subheal_alerts = set(period_support.keys()).difference(normal_alerts)
    subheal_alerts = list(subheal_alerts)
    hour_jaccard_sim = dict()
    day_jaccard_sim = dict()
    hour_jaccard_vals = []
    day_jaccard_vals = []
    for i in range(len(subheal_alerts)):
        alert_id = subheal_alerts[i]
        if alert_id not in hour_jaccard_sim:
            hour_jaccard_sim[alert_id] = dict()
        if alert_id not in day_jaccard_sim:
            day_jaccard_sim[alert_id] = dict()
        for j in range(i + 1, len(subheal_alerts)):
            alert_id2 = subheal_alerts[j]
            if len(subheal_periods[alert_id].intersection(subheal_periods[alert_id2])) == 0:
                continue
            if alert_id2 not in hour_jaccard_sim:
                hour_jaccard_sim[alert_id2] = dict()
            hour_jaccard_sim[alert_id][alert_id2] = jaccard_similarity(subheal_periods[alert_id],
                                                                       subheal_periods[alert_id2])
            hour_jaccard_sim[alert_id2][alert_id] = hour_jaccard_sim[alert_id][alert_id2]
            hour_jaccard_vals.append(hour_jaccard_sim[alert_id][alert_id2])
            if alert_id2 not in day_jaccard_sim:
                day_jaccard_sim[alert_id2] = dict()
            day_jaccard_sim[alert_id][alert_id2] = jaccard_similarity(subheal_days[alert_id], subheal_days[alert_id2])
            day_jaccard_sim[alert_id2][alert_id] = day_jaccard_sim[alert_id][alert_id2]
            day_jaccard_vals.append(day_jaccard_sim[alert_id][alert_id2])
    hour_jaccard_vals.sort()
    day_jaccard_vals.sort()

    relations = dict()
    pair_patterns = set()
    for i in range(len(subheal_alerts)):
        for j in range(i + 1, len(subheal_alerts)):
            if len(subheal_periods[subheal_alerts[i]].intersection(subheal_periods[subheal_alerts[j]])) == 0:
                continue
            hour_sim = hour_jaccard_sim[subheal_alerts[i]][subheal_alerts[j]]
            day_sim = day_jaccard_sim[subheal_alerts[i]][subheal_alerts[j]]
            if hour_sim >= period_jaccard_thrd and day_sim >= day_jaccard_thrd:
                if subheal_alerts[i] not in relations:
                    relations[subheal_alerts[i]] = set()
                if subheal_alerts[j] not in relations:
                    relations[subheal_alerts[j]] = set()
                relations[subheal_alerts[i]].add(subheal_alerts[j])
                relations[subheal_alerts[j]].add(subheal_alerts[i])
                pair_patterns.add((subheal_alerts[i], subheal_alerts[j]))
    return pair_patterns


def dfs(cur, graph, pattern, visited, candidate_alerts):
    if cur not in graph:
        return
    for nxt in graph[cur]:
        if nxt not in candidate_alerts:
            continue
        if nxt in visited:
            continue
        visited.add(nxt)
        pattern.add(nxt)
        dfs(nxt, graph, pattern, visited, candidate_alerts)


def dtw_distance(p, q):
    dp = [[-1 for j in range(len(q))] for i in range(len(p))]
    dp[0][0] = abs(p[0] - q[0])
    for i, v1 in enumerate(p):
        for j, v2 in enumerate(q):
            if i == 0 and j == 0:
                dp[i][j] = abs(v1 - v2)
            elif i == 0:
                dp[i][j] = dp[i][j - 1] + abs(v1 - v2)
            elif j == 0:
                dp[i][j] = dp[i - 1][j] + abs(v1 - v2)
            else:
                dp[i][j] = min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1]) + abs(v1 - v2)
    return dp[len(p) - 1][len(q) - 1]


def load_train_patterns(pattern_file_path, subheal_hours, flag=True):
    train_right_patterns = []
    train_wrong_patterns = []
    with open(pattern_file_path) as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            is_right = int(row['is_right'])
            pattern = [int(id) for id in row['pattern'].replace('(', '').replace(')', '').split(',')]
            if is_right == 1:
                train_right_patterns.append(pattern)
            else:
                if len(pattern) != 2:
                    print('size of wrong pattern should be 2')
                    exit(1)
                train_wrong_patterns.append(pattern)

    train_alert_pair = set()
    for pattern in train_right_patterns:
        for i in range(len(pattern)):
            for j in range(len(pattern)):
                if i == j:
                    continue
                if (1, pattern[i], pattern[j]) in train_alert_pair:
                    continue
                if (1, pattern[j], pattern[i]) in train_alert_pair:
                    continue
                if pattern[i] not in subheal_hours or pattern[j] not in subheal_hours:
                    print('invalid train correlation (', pattern[i], pattern[j], ')')
                    continue
                if len(subheal_hours[pattern[i]]) == 0:
                    continue
                if len(subheal_hours[pattern[j]]) == 0:
                    continue
                if len(subheal_hours[pattern[i]].intersection(subheal_hours[pattern[j]])) == 0:
                    continue
                train_alert_pair.add((1, pattern[i], pattern[j]))
                if flag:
                    train_alert_pair.add((1, pattern[j], pattern[i]))
    for pattern in train_wrong_patterns:
        for i in range(len(pattern)):
            for j in range(len(pattern)):
                if i == j:
                    continue
                if (-1, pattern[i], pattern[j]) in train_alert_pair:
                    continue
                if (-1, pattern[j], pattern[i]) in train_alert_pair:
                    continue
                if pattern[i] not in subheal_hours:
                    continue
                if pattern[j] not in subheal_hours:
                    continue
                if len(subheal_hours[pattern[i]]) == 0:
                    continue
                if len(subheal_hours[pattern[j]]) == 0:
                    continue
                if len(subheal_hours[pattern[i]].intersection(subheal_hours[pattern[j]])) == 0:
                    continue
                train_alert_pair.add((-1, pattern[i], pattern[j]))
                if flag:
                    train_alert_pair.add((-1, pattern[j], pattern[i]))
    return train_alert_pair


def train_sinet_attention(alerts, output_folder_path, pattern_file_path, hour_support_thrd,
                          first_day=None, last_day=None, epoch=None, model_name=None):
    if not model_name:
        ts_val = datetime.now().strftime('%Y%m%d%H%M%S')
        output_folder_path = os.path.join(output_folder_path, 'neural_network')
        model_folder_path = os.path.join(output_folder_path, 'models' + str(ts_val))
    else:
        output_folder_path = os.path.join(output_folder_path, 'neural_network')
        model_folder_path = os.path.join(output_folder_path, 'models')
    os.makedirs(model_folder_path, exist_ok=True)

    distribution, period_num, day_num = calculate_distribution(alerts, first_day, last_day)
    period_support = calculate_period_support(distribution)
    day_support = calculate_day_support(distribution, period_support)
    normal_alerts, subheal_hours = get_normal_and_subheal_alert(period_support, hour_support_thrd, distribution)

    period_support = z_normalize(period_support)
    day_support = z_normalize(day_support)
    train_alert_pair = load_train_patterns(pattern_file_path, subheal_hours)
    ret = si_net.train(period_num, day_num, period_support, day_support, train_alert_pair, model_folder_path, epoch,
                       model_name)
    return ret


# 注意这里改了，period support thrd参数位置在前面！！！20220412
def predict_sinet_attention(alerts, output_folder_path, pattern_file_path,
                            period_num, period_support,
                            day_num, day_support,
                            normal_alerts, subheal_periods,
                            similairity_thrd, model_path,
                            labeled_pair_patterns_file_path=None,
                            first_day=None, last_day=None, draw=True):
    ts_val = datetime.now().strftime('%Y%m%d%H%M%S')
    output_folder_path = os.path.join(output_folder_path, 'neural_network')
    fig_folder_path = os.path.join(output_folder_path, 'figures' + str(ts_val))
    if draw:
        os.makedirs(fig_folder_path, exist_ok=True)
    train_alert_pair = load_train_patterns(pattern_file_path, subheal_periods)
    sub_heal_alerts = list(set(subheal_periods.keys()).difference(normal_alerts))
    rpair_patterns, wpair_patterns, upair_patterns, similarity = si_net.experiment(period_num, day_num, model_path,
                                                                                   period_support,
                                                                                   day_support, similairity_thrd,
                                                                                   sub_heal_alerts,
                                                                                   normal_alerts,
                                                                                   subheal_periods,
                                                                                   train_alert_pair,
                                                                                   labeled_pair_patterns_file_path)
    pattern_folder_path = os.path.join(fig_folder_path, 'pair_patterns')
    total_patterns = set()
    total_patterns.update(rpair_patterns)
    total_patterns.update(wpair_patterns)
    total_patterns.update(upair_patterns)
    print('num of pair patterns', len(total_patterns))
    print(len(rpair_patterns), len(wpair_patterns), len(upair_patterns))
    if draw:
        for pattern_id, pattern in enumerate(total_patterns):
            sim = similarity[pattern]
            pre = 'pattern_' + str(pattern_id) + '_' + str(round(sim, 3)) + '_'
            if pattern in rpair_patterns:
                sub_pattern_folder_path = os.path.join(pattern_folder_path, 'right_patterns')
                plt_naive_distribution(alerts, sub_pattern_folder_path, pattern, pre)
            elif pattern in wpair_patterns:
                sub_pattern_folder_path = os.path.join(pattern_folder_path, 'wrong_patterns')
                plt_naive_distribution(alerts, sub_pattern_folder_path, pattern, pre)
            else:
                sub_pattern_folder_path = os.path.join(pattern_folder_path, 'unkno_patterns')
                plt_naive_distribution(alerts, sub_pattern_folder_path, pattern, pre)
    return rpair_patterns, wpair_patterns, upair_patterns


# 注意这里改了，period support thrd参数位置在前面！！！20220412
def predict_sinet_attention_no_judge(pattern_file_path,
                                     period_num, period_support,
                                     day_num, day_support,
                                     normal_alerts, subheal_periods,
                                     model_path):
    train_alert_pair = load_train_patterns(pattern_file_path, subheal_periods)
    sub_heal_alerts = list(set(subheal_periods.keys()).difference(normal_alerts))
    pattern_similarity = si_net.experiment_no_judge(period_num, day_num,
                                                    model_path,
                                                    period_support,
                                                    day_support,
                                                    sub_heal_alerts,
                                                    normal_alerts,
                                                    subheal_periods,
                                                    train_alert_pair)
    return pattern_similarity

