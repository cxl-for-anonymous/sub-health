import pandas as pd
import math
from collections import defaultdict


def get_period_id(ts, period_mins):
    day_start = pd.Timestamp(ts.year, ts.month, ts.day)
    day_start = day_start.tz_localize(ts.tz)
    delta_s = (ts - day_start).total_seconds()
    width_s = period_mins * 60
    # left close right open time period
    period_id = math.floor(delta_s * 1.0 / width_s)
    return period_id


def calculate_distribution(alerts, first_day=None, last_day=None, period_mins=60):
    if first_day is not None or last_day is not None:
        first_day = alerts[0].timestamp if first_day is None else first_day
        last_day = alerts[-1].timestamp if last_day is None else last_day
        # transform time range
        first_day = pd.Timestamp(first_day, tz=alerts[0].timestamp.tz)
        last_day = pd.Timestamp(last_day, tz=alerts[0].timestamp.tz)
        # get valid alert
        valid_alerts = []
        for alert in alerts:
            if first_day <= alert.timestamp <= last_day:
                valid_alerts.append(alert)
        alerts = valid_alerts
    else:
        first_day = alerts[0].timestamp
        last_day = alerts[-1].timestamp

    # calculate alert occurrence
    occurrence = dict()
    for alert in alerts:
        alert_id = alert.id
        day = (alert.timestamp.year, alert.timestamp.month, alert.timestamp.day)
        period_id = get_period_id(alert.timestamp, period_mins)
        if alert_id not in occurrence:
            occurrence[alert_id] = defaultdict(int)
        occurrence[alert_id][(day, period_id)] += 1

    # calculate alert distribution
    period_num = math.ceil(24 * 60 / period_mins)
    distribution = defaultdict(list)
    for alert_id in occurrence:
        if len(occurrence[alert_id]) <= 1:
            continue
        cur_day = first_day
        while cur_day <= last_day:
            day_occur = []
            for period_id in range(period_num):
                day_hour = ((cur_day.year, cur_day.month, cur_day.day), period_id)
                if day_hour in occurrence[alert_id]:
                    day_occur.append(1)
                else:
                    day_occur.append(0)
            cur_day += pd.Timedelta(days=1)
            distribution[alert_id].append(day_occur)
        day_num = len(distribution[alert_id])
    print('alert first day', first_day)
    print('alert last day', last_day)
    print('num of periods',period_num)
    print('num of days', day_num)
    print('num of alerts', len(alerts))
    print('num of alert types', len(distribution))
    return distribution, period_num, day_num

def calculate_period_support(distribution):
    cnt_days_thrd = 4
    day_span_thrd = 14
    period_support = dict()
    alert_days = dict()
    # for each alert
    for alert_id in distribution:
        period_support[alert_id] = []
        alert_days[alert_id] = set()
        # for each time period
        # print(alert_id)
        # print(len(distribution[alert_id][0]))
        for period in range(len(distribution[alert_id][0])):
            cur_day = 0
            cnt_days = 0
            start_day = None
            end_day = None
            # find the first day with alert occurrence
            while cur_day < len(distribution[alert_id]):
                flag = False
                # search the period window
                for nbr_period in range(period - 1, period + 2):
                    if nbr_period < 0 or nbr_period >= len(distribution[alert_id][0]):
                        continue
                    # find it!
                    if distribution[alert_id][cur_day][nbr_period] != 0:
                        flag = True
                        break
                if flag:
                    break
                cur_day += 1
            # if the period has occurrence during the days
            if cur_day < len(distribution[alert_id]):
                # start from next days, skipping the first occurrence
                cnt_days += 1
                start_day = cur_day
                end_day = cur_day
            pre_day = cur_day
            cur_day += 1
            is_second_event = True
            pre_delta_period = None
            pre_delta_day = None
            support = 0
            # for each day
            while cur_day < len(distribution[alert_id]):
                best_delta_cycle = None
                best_delta_sum = None
                best_delta_period = None
                best_delta_day = None
                # for a period window
                for nbr_period in range(max(0, period - 1), min(len(distribution[alert_id][0]), period + 2)):
                    if distribution[alert_id][cur_day][nbr_period] == 0:
                        continue
                    # period delta
                    cur_delta_period = min(abs(period - nbr_period), 24 - (period - nbr_period))
                    # day delta
                    cur_delta_day = cur_day - pre_day
                    # total delta
                    cur_delta_sum = cur_delta_period + cur_delta_day - 1
                    # if the occurrence and the previous belong to the same anomaly event
                    if cur_delta_sum == 0:
                        # the occurrence has no cycle delta with the previous one as they belong to the same event
                        best_delta_cycle = 0
                        # do not change the cycle delta
                        best_delta_sum = None
                        best_delta_period = None
                        best_delta_day = None
                    # if the occurrence and the previous belong to difference event
                    else:
                        # if the occurrence is the first occurrence of the second event
                        if is_second_event:
                            # record the occurrence with the minimum delta in the period window
                            if best_delta_sum is None or cur_delta_sum < best_delta_sum:
                                best_delta_cycle = None
                                best_delta_sum = cur_delta_sum
                                best_delta_period = cur_delta_period
                                best_delta_day = cur_delta_day
                        # otherwise calculate the cycle change between current event and the previous event
                        else:
                            # cycle change = period_delta_change + day_delta_change
                            cycle_delta = abs(pre_delta_period - cur_delta_period) + abs(pre_delta_day - cur_delta_day)
                            # only record the occurrence with minumum cycle change in the period window
                            if best_delta_cycle is None or cycle_delta < best_delta_cycle:
                                best_delta_cycle = cycle_delta
                                best_delta_sum = cur_delta_sum
                                best_delta_period = cur_delta_period
                                best_delta_day = cur_delta_day
                # if the occurrence and the previous belong to different event, record the delta
                if best_delta_sum is not None:
                    is_second_event = False
                    pre_delta_day = best_delta_day
                    pre_delta_period = best_delta_period
                # if the occurrence is not the first occurrence of the first and second event,
                # then calculate its contribution to the period support
                if best_delta_cycle is not None:
                    support += 1.0 / (best_delta_cycle + 1)
                # record the day
                if best_delta_cycle is not None or best_delta_sum is not None:
                    pre_day = cur_day
                    cnt_days += 1
                    end_day = cur_day
                cur_day += 1
            # filter out invalid alerts
            if cnt_days < cnt_days_thrd or (end_day - start_day + 1) < day_span_thrd:
                period_support[alert_id].append(0)
            else:
                period_support[alert_id].append(round(support, 3))
    return period_support


def calculate_day_support(distribution, period_support):
    day_support = dict()
    for alert_id in distribution:
        day_support[alert_id] = [0 for _ in range(len(distribution[alert_id]))]
        for day in range(len(distribution[alert_id])):
            for period_id in range(len(distribution[alert_id][day])):
                if distribution[alert_id][day][period_id] > 0:
                    day_support[alert_id][day] += period_support[alert_id][period_id]
    return day_support


def get_normal_and_subheal_alert(period_support, period_support_thrd, distribution=None):
    normal_alerts = set()
    subheal_periods = dict()
    for alert_id in period_support:
        subheal_periods[alert_id] = set()
        if max(period_support[alert_id]) < period_support_thrd:
            normal_alerts.add(alert_id)
        for period_id in range(len(period_support[alert_id])):
            if period_support[alert_id][period_id] >= period_support_thrd:
                subheal_periods[alert_id].add(period_id)
            elif distribution:
                for day in range(len(distribution[alert_id])):
                    distribution[alert_id][day][period_id] = 0
        if len(subheal_periods[alert_id]) == 0:
            subheal_periods.pop(alert_id)
    if distribution:
        for alert_id in distribution:
            flag = False
            for day in range(len(distribution[alert_id])):
                for period_id in range(len(period_support[alert_id])):
                    if distribution[alert_id][day][period_id] > 0:
                        flag = True
                        break
                if flag:
                    break
            if not flag and alert_id in subheal_periods:
                subheal_periods.pop(alert_id)
    return normal_alerts, subheal_periods

