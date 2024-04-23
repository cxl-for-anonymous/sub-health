from approaches import *
from datetime import datetime
from data_reader import *

def load_alert(first_day = None,last_day = None):
    alerts = read_alert(alert_file_path, timestamp_col, template_id_col, template_content_col,first_day,last_day)
    return alerts


def load_ground_truth_sub_heal_alerts():
    total_subheal_alerts = set()
    with open(ground_truth_alert_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            total_subheal_alerts.add(int(row[0]))
    return total_subheal_alerts


def experiment_nerual_network_simialrity_thrd(output_folder_path, period_mins, period_support_thrd, epoch,
                                              first_day=None, last_day=None):
    pattern_file_path = train_data_path
    ground_truth_correlations_file_path = ground_truth_pattern_path

    output_file_path = os.path.join(output_folder_path, 'neural_network_experiment.csv')

    data_read_start_time = datetime.now()
    alerts = load_alert()
    data_read_end_time = datetime.now()
    data_read_time = (data_read_end_time - data_read_start_time).total_seconds()

    # train model
    train_start_time = datetime.now()
    model_name = str(period_mins) + '_' + str(period_support_thrd) + '_' + str(epoch)
    model_path = train_sinet_attention(alerts, output_folder_path, pattern_file_path, period_support_thrd,
                                       first_day, last_day, epoch=epoch, model_name=model_name)
    train_end_time = datetime.now()
    train_time = (train_end_time - train_start_time).total_seconds()

    # mine sub-health alerts
    mine_subheal_alert_start_time = datetime.now()
    distribution, period_num, day_num = calculate_distribution(alerts, first_day, last_day)
    period_support = calculate_period_support(distribution)
    normal_alerts, subheal_periods = get_normal_and_subheal_alert(period_support, period_support_thrd,
                                                                  distribution)
    mine_subheal_alert_end_time = datetime.now()
    mine_subheal_alert_time = (mine_subheal_alert_end_time - mine_subheal_alert_start_time).total_seconds()

    # experiment model
    feature_extract_start_time = datetime.now()
    day_support = calculate_day_support(distribution, period_support)
    period_support = z_normalize(period_support)
    day_support = z_normalize(day_support)
    feature_extract_end_time = datetime.now()
    feature_extract_time = (feature_extract_end_time - feature_extract_start_time).total_seconds()

    correlation_mine_start_time = datetime.now()
    pattern_smilarity = predict_sinet_attention_no_judge(pattern_file_path,
                                                         period_num, period_support,
                                                         day_num, day_support,
                                                         normal_alerts, subheal_periods, model_path)
    correlation_mine_end_time = datetime.now()
    correlation_mine_time = (correlation_mine_end_time - correlation_mine_start_time).total_seconds()

    # measure different similarity thresholds
    result = []
    total_upairs = set()
    labeled_right_pair_patterns, labeled_wrong_pair_patterns = load_labeled_patterns(
        ground_truth_correlations_file_path)
    with open(output_file_path, 'a+') as f:
        writer = csv.writer(f)
        writer.writerow(
            ['data_read_time', 'train_time', 'mine_subheal_alert_time', 'feature_extract_time',
             'correlation_mine_time', 'period_mins', 'period_support_thrd', 'similarity thrd',
             'total correlations', 'right correlations', 'wrong correlations', 'unkonwn correlations'])
        for similairity_thrd in range(20, 50):
            similairity_thrd /= 100.0
            rpair_patterns = set()
            wpair_patterns = set()
            upair_patterns = set()

            pattern_folder_path = os.path.join(output_folder_path, 'model_pattern')
            os.makedirs(pattern_folder_path, exist_ok=True)
            patterns_result_file_path = os.path.join(pattern_folder_path, 'patterns_' + str(period_mins) + '_' +
                                                     str(period_support_thrd) + '_' + str(
                similairity_thrd) + '.csv')
            with open(patterns_result_file_path, 'w') as pattern_f:
                pattern_writer = csv.writer(pattern_f)
                pattern_writer.writerow(['is_right', 'pattern id1', 'pattern id2'])
                for pattern in pattern_smilarity:
                    sim = pattern_smilarity[pattern]
                    if sim > similairity_thrd:
                        if pattern in labeled_right_pair_patterns:
                            rpair_patterns.add(pattern)
                            pattern_writer.writerow([1, pattern[0], pattern[1]])
                        elif pattern in labeled_wrong_pair_patterns:
                            wpair_patterns.add(pattern)
                            pattern_writer.writerow([-1, pattern[0], pattern[1]])
                        else:
                            pattern_writer.writerow([0, pattern[0], pattern[1]])
                            upair_patterns.add(pattern)
            total_upairs.update(upair_patterns)
            result.append((data_read_time, train_time, mine_subheal_alert_time, feature_extract_time,
                           correlation_mine_time, period_mins,
                           period_support_thrd, similairity_thrd,
                           len(rpair_patterns) + len(wpair_patterns) + len(upair_patterns),
                           len(rpair_patterns), len(wpair_patterns), len(upair_patterns)))
            print(result[-1])
            writer.writerow(result[-1])
            f.flush()
    for pattern_id, pattern in enumerate(total_upairs):
        pre = 'pattern_' + str(pattern_id) + '_' + 'nill' + '_'
        sub_pattern_folder_path = os.path.join(output_folder_path, 'unkown_patterns')
        plt_naive_distribution(alerts, sub_pattern_folder_path, pattern, pre)


def experiment_period_width(output_folder_path, period_mins, hour_support_thrd, first_day=None, last_day=None):
    alerts = load_alert()
    total_subheal_alerts = load_ground_truth_sub_heal_alerts()
    result = dict()
    distribution, period_num, day_num = calculate_distribution(alerts, first_day, last_day,
                                                               period_mins=period_mins)
    period_support = calculate_period_support(distribution)
    normal_alerts, subheal_periods = get_normal_and_subheal_alert(period_support, hour_support_thrd)
    subheal_alerts = set(subheal_periods.keys())
    precision = len(subheal_alerts.intersection(total_subheal_alerts)) / len(subheal_alerts)
    recall = len(subheal_alerts.intersection(total_subheal_alerts)) / len(total_subheal_alerts)
    f1_score = precision * recall / (precision + recall) * 2
    result[(period_mins, hour_support_thrd)] = (period_mins, hour_support_thrd, len(total_subheal_alerts),
                                                len(subheal_alerts),
                                                len(subheal_alerts.intersection(total_subheal_alerts)),
                                                precision, recall, f1_score)
    print(period_mins, hour_support_thrd, len(total_subheal_alerts),
          len(subheal_alerts), len(subheal_alerts.intersection(total_subheal_alerts)),
          precision, recall, f1_score)
    output_file_path = os.path.join(output_folder_path, 'period_mins_experiment.csv')
    return output_file_path


def check_patterns(pair_correlations):
    labeled_right_pair_patterns, labeled_wrong_pair_patterns = load_labeled_patterns(
        ground_truth_pattern_path)
    rp = set()
    wp = set()
    up = set()
    for pattern in pair_correlations:
        if pattern in labeled_right_pair_patterns:
            rp.add(pattern)
        elif pattern in labeled_wrong_pair_patterns:
            wp.add(pattern)
        else:
            up.add(pattern)
    return rp, wp, up


def experiment_other_approaches(output_folder_path, period_mins, period_support_thrd, first_day=None, last_day=None,
                                draw=False):
    stime = datetime.now()
    alerts = load_alert()
    etime = datetime.now()
    read_alert_time = (etime - stime).total_seconds()

    stime = datetime.now()
    distribution, _, _ = calculate_distribution(alerts, first_day, last_day, period_mins=period_mins)
    period_support = calculate_period_support(distribution)
    day_support = calculate_day_support(distribution, period_support)
    normal_alerts, subheal_hours = get_normal_and_subheal_alert(period_support, period_support_thrd=period_support_thrd)
    etime = datetime.now()
    mine_alert_time = (etime - stime).total_seconds()
    print('mine sub_heal alert time', mine_alert_time)
    print('num of sub health alert types',len(subheal_hours))

    total_up = set()
    stime = datetime.now()
    dtw_patterns = dtw(normal_alerts, subheal_hours, 10, 12, period_support, day_support)
    etime = datetime.now()
    dtw_time = (etime - stime).total_seconds()
    rp, wp, up = check_patterns(dtw_patterns)
    total_up.update(up)
    total_patterns = set()
    total_patterns.update(rp)
    total_patterns.update(wp)
    total_patterns.update(up)
    construct_incident(output_folder_path, total_patterns, subheal_hours, period_mins, 'DTW')
    print('approach', 'DTW', 'read data time', read_alert_time, 'approach time', dtw_time, 'right num ', len(rp),
          'wrong num', len(wp),
          'uknown num', len(up), 'total num', len(rp) + len(wp) + len(up))

    stime = datetime.now()
    cos_patterns = cos(normal_alerts, subheal_hours, 0.89, 0.75, period_support, day_support)
    etime = datetime.now()
    cos_time = (etime - stime).total_seconds()
    rp, wp, up = check_patterns(cos_patterns)
    total_up.update(up)
    total_patterns = set()
    total_patterns.update(rp)
    total_patterns.update(wp)
    total_patterns.update(up)
    construct_incident(output_folder_path, total_patterns, subheal_hours, period_mins, 'COS')
    print('approach', 'COS', 'read data time', read_alert_time, 'approach time', cos_time, 'right num ', len(rp),
          'wrong num', len(wp),
          'uknown num', len(up), 'total num', len(rp) + len(wp) + len(up))

    stime = datetime.now()
    eud_patterns = euclidean(distribution, normal_alerts, period_support, subheal_hours, 33)
    etime = datetime.now()
    eud_time = (etime - stime).total_seconds()
    rp, wp, up = check_patterns(eud_patterns)
    total_up.update(up)
    total_patterns = set()
    total_patterns.update(rp)
    total_patterns.update(wp)
    total_patterns.update(up)
    construct_incident(output_folder_path, total_patterns, subheal_hours, period_mins, 'Euclidean')
    print('approach', 'Euclidean', 'read data time', read_alert_time, 'approach time', eud_time, 'right num ', len(rp),
          'wrong num', len(wp),
          'uknown num', len(up), 'total num', len(rp) + len(wp) + len(up))

    stime = datetime.now()
    jcd_patterns = jaccard(distribution, period_support, normal_alerts, subheal_hours, 0.5, 0.5)
    etime = datetime.now()
    jcd_time = (etime - stime).total_seconds()
    rp, wp, up = check_patterns(jcd_patterns)
    total_up.update(up)
    total_patterns = set()
    total_patterns.update(rp)
    total_patterns.update(wp)
    total_patterns.update(up)
    construct_incident(output_folder_path, total_patterns, subheal_hours, period_mins, 'Jaccard')
    print('approach', 'Jaccard', 'read data time', read_alert_time, 'approach time', jcd_time, 'right num ', len(rp),
          'wrong num', len(wp),
          'uknown num', len(up), 'total num', len(rp) + len(wp) + len(up))
    if draw:
        for pattern_id, pattern in enumerate(total_up):
            sim = 'nill'
            pre = 'pattern_' + str(pattern_id) + '_' + str(sim) + '_'
            sub_pattern_folder_path = os.path.join(output_folder_path, 'unkown_patterns_others')
            plt_naive_distribution(load_alert(first_day,last_day), sub_pattern_folder_path, pattern, pre)


def construct_incident(output_folder_path, pair_correlations, sub_health_periods, period_mins, pre=''):
    stime = datetime.now()
    alert_ids = set()
    relations = dict()
    for alert_id1, alert_id2 in pair_correlations:
        alert_ids.add(alert_id1)
        alert_ids.add(alert_id2)
        if alert_id1 not in relations:
            relations[alert_id1] = set()
        if alert_id2 not in relations:
            relations[alert_id2] = set()
        relations[alert_id1].add(alert_id2)
        relations[alert_id2].add(alert_id1)
    for alert_id in sub_health_periods:
        alert_ids.add(alert_id)
        if alert_id not in relations:
            relations[alert_id] = set()

    period_alerts = dict()
    for alert_id in sub_health_periods:
        for period_id in sub_health_periods[alert_id]:
            if period_id not in period_alerts:
                period_alerts[period_id] = set()
            period_alerts[period_id].add(alert_id)
    result_incident = set()
    incident_periods = dict()
    for period_id in period_alerts:
        visited = set()
        for alert_id in period_alerts[period_id]:
            if alert_id not in alert_ids:
                continue
            if alert_id not in visited:
                pattern = {alert_id, }
                visited.add(alert_id)
                dfs(alert_id, relations, pattern, visited, period_alerts[period_id])
                pattern = tuple(sorted(pattern))
                result_incident.add(pattern)
                if pattern not in incident_periods:
                    incident_periods[pattern] = set()
                incident_periods[pattern].add(period_id)
    remove = set()
    for incident1 in result_incident:
        for incident2 in result_incident:
            if incident1 == incident2:
                continue
            if len(incident1) >= len(incident2):
                continue
            incident1_st = set(incident1)
            incident2_st = set(incident2)
            if len(incident1_st.intersection(incident2_st)) == len(incident1_st):
                remove.add(incident1)
    for incident in remove:
        result_incident.remove(incident)
    labeled_right_pair_patterns, labeled_wrong_pair_patterns = load_labeled_patterns(ground_truth_pattern_path)
    right_incidents = []
    wrong_incidents = []
    for incident in result_incident:
        is_wrong = True
        if len(incident) == 1:
            is_wrong = False
        else:
            for ald1 in incident:
                for ald2 in incident:
                    if ald1 == ald2:
                        continue
                    if (ald1, ald2) in labeled_right_pair_patterns:
                        is_wrong = False
                        break
                if not is_wrong:
                    break
        if is_wrong:
            wrong_incidents.append(incident)
        else:
            right_incidents.append(incident)
    right_alert_ids = set()
    for rincident in right_incidents:
        for alert_id in rincident:
            right_alert_ids.add(alert_id)
    print('involved alert types', len(alert_ids))
    print('involved right alert types', len(right_alert_ids))
    print('total incidents', len(result_incident), 'right incidents', len(right_incidents),
          'wrong incidents', len(wrong_incidents))
    output_file_path = os.path.join(output_folder_path, pre+'_incident.csv')
    with open(output_file_path,'w') as f:
        writer = csv.writer(f)
        writer.writerow(['sub-health periods('+str(period_mins)+')','alert types'])
        for incident in result_incident:
            writer.writerow([str(incident_periods[incident]),str(incident)])
    etime = datetime.now()
    print('construct incident time(s)', (etime - stime).total_seconds())


if __name__ == '__main__':
    alert_file_path = ''
    train_data_path = ''
    ground_truth_alert_path = ''
    ground_truth_pattern_path = ''
    timestamp_col = ''
    template_id_col = ''
    template_content_col = ''

    output_folder_path = r'../output/BankB'
    first_day = '2019/10/01'
    last_day = '2019/10/31'
    period_mins = 60
    hour_support_thrd = 2.9

    # Sub-health Alert
    period_experiment_result = experiment_period_width(output_folder_path,first_day,last_day)

    # Sub-health Correlation
    experiment_nerual_network_simialrity_thrd(output_folder_path, period_mins, hour_support_thrd, 600, first_day, last_day)

    # Other Approaches
    experiment_other_approaches(output_folder_path, period_mins, hour_support_thrd,first_day,last_day, draw=True, incident = False)
