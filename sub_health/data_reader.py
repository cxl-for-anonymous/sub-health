import pandas as pd
import csv


class Alert:
    def __init__(self,data,timestamp,content,id):
        self.data = data
        self.timestamp = timestamp
        self.content = content
        self.id = id


def read_alert(file_path,timestamp_col='time',tempate_id_col='templateId',template_col='alm_note',first_day=None,last_day=None):
    alert_df = pd.read_csv(file_path, low_memory=False)
    # print(alert_df.columns)
    alert_df.loc[:, timestamp_col] = pd.to_datetime(alert_df.loc[:, timestamp_col])
    alert_df.sort_values(by=timestamp_col,inplace=True)
    alert_df.rename(columns={timestamp_col:timestamp_col.replace('@',''),
                             tempate_id_col:tempate_id_col.replace('@',''),
                             template_col: template_col.replace('@', '')}, inplace = True)
    timestamp_col = timestamp_col.replace('@','')
    tempate_id_col = tempate_id_col.replace('@', '')
    template_col = template_col.replace('@', '')
    alerts = []
    for row in alert_df.itertuples():
        # print(row)
        alert = Alert(row,getattr(row,timestamp_col),getattr(row,template_col),getattr(row,tempate_id_col))
        alerts.append(alert)
    if first_day:
        first_day = pd.Timestamp(first_day, tz=alerts[0].timestamp.tz)
    if last_day:
        last_day = pd.Timestamp(last_day, tz=alerts[0].timestamp.tz)
    if first_day or last_day:
        valid_alerts = []
        for alert in alerts:
            if first_day and first_day > alert.timestamp:
                    continue
            if last_day and alert.timestamp > last_day:
                    continue
            valid_alerts.append(alert)
        alerts = valid_alerts
    return alerts


def load_labeled_patterns(labeled_pair_patterns_file_path):
    right_pair_patterns = set()
    wrong_pair_patterns = set()
    if labeled_pair_patterns_file_path is not None:
        with open(labeled_pair_patterns_file_path) as csvfile:
            csv_reader = csv.reader(csvfile)
            for row in csv_reader:
                is_right = int(row[0])
                alert_id1 = int(row[1])
                alert_id2 = int(row[2])
                if is_right == 1:
                    right_pair_patterns.add((alert_id1, alert_id2))
                    right_pair_patterns.add((alert_id2, alert_id1))
                else:
                    wrong_pair_patterns.add((alert_id1, alert_id2))
                    wrong_pair_patterns.add((alert_id2, alert_id1))
    return right_pair_patterns, wrong_pair_patterns
