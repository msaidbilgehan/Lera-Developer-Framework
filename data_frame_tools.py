import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from stdo import stdo
from tools import path_control


def convert_Date(date_str):
    date_part, time_part = date_str.split(' ')
    year, month, day = map(int, date_part.split('-'))
    hour, minute, second = map(int, time_part.split(':'))
    return pd.Timestamp(year=year, month=month, day=day, hour=hour, minute=minute, second=second)


def convert_Date_Using_Read_Component(date_str):
    date_part, time_part = date_str.split('-')
    year, month, day = map(int, date_part.split('_'))
    hour, minute, second = map(int, time_part.split('.')[0].split('_'))
    return pd.Timestamp(year=year, month=month, day=day, hour=hour, minute=minute, second=second)


def save_To_CSV(df, csv_filename):
    if not path_control(csv_filename, is_file=True, is_directory=False)[0]:
        os.makedirs(os.path.dirname(csv_filename), exist_ok=True)
    df.to_csv(csv_filename, mode='a', header=not os.path.exists(csv_filename), index=False)


def edit_Values_In_CSV(df=None):
    if df is None:
        return None
    df_first = df.copy()
    df_first['FullDateTime'] = df_first['Date'].apply(convert_Date)
    df_first.drop(columns=['Date'], inplace=True)
    df_first['PCB Name'] = df_first['PCB Name'].apply(
        lambda x: 'Artouch-12_13_14' if x.startswith('artouch')
                else ('Cay_Makinesi' if x.startswith('Çay')
                    else ('Rob14' if x.startswith('rob')
                        else ('Beast_Gas_Timer' if x.startswith('Beast Gas')
                            else ('mind-364' if x.startswith('mind')
                                else x
                                )
                            )
                        )
                    )
    )
    df_first['Sub Folder'] = df_first['Sub Folder'].apply(lambda x: 'False' if x in ['False_up', 'False_left', 'False_down', 'False_right', 'NG'] else 'True')
    df_first = df_first.sort_values(by='FullDateTime').reset_index(drop=True)
    return pd.DataFrame(df_first)


def create_Organized_CSV(df=None):
    df = edit_Values_In_CSV(df)
    df['FullDateTime'] = pd.to_datetime(df['FullDateTime'], errors='coerce')
    df = df.sort_values(by='FullDateTime')
    return pd.DataFrame(df)


def data_Preprocess_For_The_All_Model(data=None):
    data['label'] = data['Sub Folder'].apply(lambda x: 0 if x == "False" else 1)
    data['dayofmonth'] = data["FullDateTime"].dt.day
    data['hour'] = data["FullDateTime"].dt.hour
    data['minute'] = data["FullDateTime"].dt.minute
    data['month'] = data["FullDateTime"].dt.month
    data['year'] = data["FullDateTime"].dt.year

    data['PCB_enc'] = data['PCB Name'].astype('category').cat.codes
    data["PCB_Component_Combo"] = data["PCB Name"] + "+" + data["Component"]

    le = LabelEncoder()
    data["PCB_Component_enc"] = le.fit_transform(data["PCB_Component_Combo"])
    data["PCB_Component_dec"] = le.inverse_transform(data["PCB_Component_enc"])
    stdo(1,"Combo:{} | Encode:{} | Decode:{}".format(len(data["PCB_Component_Combo"].unique()),len(data["PCB_Component_enc"].unique()), len(data["PCB_Component_dec"].unique())))
    data = data.sort_values("FullDateTime").reset_index(drop=True)

    pcb_state = {}
    pcb_counters = []

    for i, row in data.iterrows():
        pcb = row['PCB Name']
        date = row["FullDateTime"]
        year = row['year']
        month = row['month']
        day = row['dayofmonth']
        key = (pcb, year, month, day)

        if key not in pcb_state:
            # İlk kez bu gün ve PCB için bir kayıt
            pcb_state[key] = [date, 1]
        else:
            last_time, current_count = pcb_state[key]
            diff = (date - last_time).total_seconds()
            same_day = (date.date() == last_time.date())
            if diff > 12 and same_day:
                current_count += 1
            pcb_state[key] = [date, current_count]

        pcb_counters.append(pcb_state[key][1])

    data['pcb_counter'] = pcb_counters
    return data


def process_LSTM_Datatype(pcb_data=None):
    pcb_data['FullDateTime'] = pd.to_datetime(pcb_data['FullDateTime']).dt.date
    daily_counts = pcb_data.groupby(['FullDateTime', 'PCB_enc', "dayofmonth", "month", "year"]).agg({
        'pcb_counter': 'max'
    }).reset_index()

    pivot_df = daily_counts.pivot_table(
        index='FullDateTime',
        columns='PCB_enc',
        values='pcb_counter',
        aggfunc='sum'
    ).fillna(0)

    pivot_df = pivot_df.reset_index()

    for col in pivot_df.columns:
        if col != 'FullDateTime':
            pivot_df[col] = pivot_df[col].astype(int)

    pcb_counts = pd.DataFrame({
        'FullDateTime': pivot_df['FullDateTime'],
        **{col: pivot_df[col].values for col in pivot_df.columns if col != 'FullDateTime'}
    })
    return pcb_counts


def split_Data_and_Normalize_For_LSTM_Model(lstm_data):
    """
    Train: Son iki günü hariç tüm veriler
    Test: Son iki gün
    """
    columns = list(lstm_data.columns)
    columns.remove("FullDateTime")

    # FullDateTime sıralı olmalı!
    lstm_data = lstm_data.sort_values("FullDateTime").reset_index(drop=True)

    # Son iki günü buls
    unique_dates = lstm_data["FullDateTime"].unique()
    if len(unique_dates) < 3:
        raise ValueError("Veri setinde en az iki farklı gün olmalı!")

    test_days = unique_dates[-3:]
    train = lstm_data[~lstm_data["FullDateTime"].isin(test_days)]
    test = lstm_data[lstm_data["FullDateTime"].isin(test_days)]

    scaler = MinMaxScaler()
    normalized_data_train = scaler.fit_transform(train[columns]) if len(train) > 0 else np.empty((0, len(columns)))
    normalized_data_test = scaler.transform(test[columns]) if len(test) > 0 else np.empty((0, len(columns)))

    train_df = pd.DataFrame(normalized_data_train, columns=columns)
    test_df = pd.DataFrame(normalized_data_test, columns=columns)

    train_df["FullDateTime"] = train["FullDateTime"].values
    test_df["FullDateTime"] = test["FullDateTime"].values

    return train_df, test_df, scaler, columns


def generate_Sequence_Data_For_LSTM_Model(df, seq_length=30):
    X = df.reset_index(drop=True)
    y = df.reset_index(drop=True)

    sequences = []
    labels = []
    target_dates = []

    for index in range(len(X) - seq_length + 1):
        sequences.append(X.iloc[index : index + seq_length].drop(columns="FullDateTime").values)
        labels.append(y.iloc[index + seq_length - 1].drop("FullDateTime").values)
        target_dates.append(X.iloc[index + seq_length - 1]["FullDateTime"])

    sequences = np.array(sequences)
    labels = np.array(labels)

    return sequences, labels, target_dates


def make_Auto_Pct(values):
    def inner(pct):
        total = sum(values)
        val = int(round(pct * total / 100.0))
        return f'{val} adet' if val > 0 else ''
    return inner


def plot_Pred_vs_Real_Pie_Chart(result_df, lstm_pred_date, pcb_names_decode):
    # Tarihi datetime yap
    lstm_pred_date = pd.to_datetime(lstm_pred_date).date()
    # lstm_pred_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    result_df["FullDateTime"] = pd.to_datetime(result_df["FullDateTime"]).dt.date

    pred_row = result_df[result_df["FullDateTime"] == lstm_pred_date]
    # real_row = lstm_test_data[lstm_test_data["FullDateTime"] == lstm_pred_date]
    stdo(1, f"Pred Row:\n {pred_row}")
    if pred_row.empty:  # or real_row.empty:
        print(f"{lstm_pred_date} tarihi için veri bulunamadı.")
        return

    # Sadece tahmin değerlerini al (0-6 arası PCB'ler)
    pred_vals = pred_row.drop(columns=["FullDateTime"]).iloc[0]
    # real_vals = real_row.drop(columns=["FullDateTime"]).iloc[0]

    # 0 olmayanları al ve label eşleştir
    pred_vals = pred_vals[pred_vals > 0]
    # real_vals = real_vals[real_vals > 0]

    # Ortak PCB'leri (tahmin ya da gerçek olanlar) birleştir
    # all_keys = sorted(set(pred_vals.index).union(real_vals.index), key=int)

    # pred_data = [pred_vals.get(k, 0) for k in all_keys]
    # real_data = [real_vals.get(k, 0) for k in all_keys]
    # labels = [pcb_names_decode.get(k, f"PCB Adı: {k}") for k in all_keys]
    labels = [pcb_names_decode.get(str(k), f"PCB Adı: {k}") for k in pred_vals.index]

    # Pie chart çizimi (yan yana)
    # fig, axes = plt.subplots(figsize=(12, 6))
    # explode = [0.05] * len(labels)

    # Tahmin grafiği
    fig, ax = plt.subplots(figsize=(6, 6))
    explode = [0.05] * len(labels)

    # Tahmin grafiği
    ax.pie(pred_vals, labels=labels, autopct=make_Auto_Pct(pred_vals),
        explode=explode, shadow=False, startangle=90)
    ax.set_title(f"Tahmin - {lstm_pred_date}")
    ax.text(1.1, 1.1, f"Toplam Tahmin: {sum(pred_vals)}", ha="right", va="top", fontsize=10)

    # Gerçek grafiği
    # axes[1].pie(real_data, labels=labels, autopct=make_Auto_Pct(real_data),
    #             explode=explode, shadow=True, startangle=90)
    # axes[1].set_title(f"Gerçek - {lstm_pred_date}")
    # axes[1].text(1.1, 1.1, f"Toplam Gerçek: {sum(real_data)}", ha="right", va="top", fontsize=10)

    # plt.tight_layout()
    # plt.show()
    if not path_control('../predictive-manufacturing-intelligence/results/lstm/', is_file=False, is_directory=True)[0]:
        os.makedirs(os.path.dirname('../predictive-manufacturing-intelligence/results/lstm/'), exist_ok=True)
    plt.savefig(f"../predictive-manufacturing-intelligence/results/lstm/{lstm_pred_date}.png", dpi=300, bbox_inches="tight")


def get_Components_by_PCB_From_Predictions(data, result_df, day_of_month=1, month=2, year=2025):
    total_seconds = 60 * 24 * 60  # bir gün = 86400 saniye
    current_time = datetime(int(year), int(month), int(day_of_month), 0, 0, 0)

    all_rows = []

    row = result_df.iloc[0]  # sadece bir günlük tahmin
    for pcb_enc_str, pcb_number in row.items():
        if pcb_enc_str == "FullDateTime":
            continue  # tarih sütunu varsa atla

        pcb_enc = int(pcb_enc_str)
        pcb_number = int(pcb_number)

        if pcb_number == 0:
            continue

        components = data[data["PCB_enc"] == pcb_enc]["PCB_Component_enc"].unique()

        if len(components) == 0:
            continue

        seconds_per_component = total_seconds / pcb_number

        for _ in range(pcb_number):
            for component in components:
                all_rows.append({
                    "PCB Name": pcb_enc,
                    "Component": component,
                    "dayofmonth": day_of_month,
                    "month": month,
                    "year": year,
                    "hour": current_time.hour,
                    "minute": current_time.minute
                })
            current_time += timedelta(seconds=int(seconds_per_component))

    return pd.DataFrame(all_rows)


def plot_LGBM_Pred_Bar_Chart(df_combined, lgbm_y_pred, next_day):

    data = {
        'PCB_Component_enc': lgbm_y_pred[:,0],
        'label': lgbm_y_pred[:,1]
    }

    df_pred = pd.DataFrame(data)
    df_pred.iloc[:, 0].unique()

    df_pred = df_pred.merge(
        df_combined[['PCB_Component_enc', 'PCB_Component_dec']].drop_duplicates(),
        on='PCB_Component_enc',
        how='left'
    )

    df_ng = df_pred[df_pred["label"] == 0]["PCB_Component_dec"].value_counts().sort_index()
    df_ng = df_ng.to_frame().reset_index()

    lgbm_top5_pred_name = []
    lgbm_top5_pred_count = []
    for i in range(5 - df_ng.shape[0]):
        lgbm_top5_pred_name.append(f"N/A {i+1}")
        lgbm_top5_pred_count.append(0)
    for data in df_ng.itertuples():
        lgbm_top5_pred_name.append(data[1])
        lgbm_top5_pred_count.append(data[2])

    fig, ax = plt.subplots(figsize=(15, 6))
    bars = ax.barh(lgbm_top5_pred_name, lgbm_top5_pred_count, color='red')
    ax.set_xlabel('Adet')
    ax.set_title(f"{next_day} | Top 5 Hatalı Komponent ")
    ax.set_xlim(0, max(lgbm_top5_pred_count)*1.05)  # grafiği biraz boşluk bırakacak şekilde ayarladık
    # ax.show()
    for id, bar in enumerate(bars):
        if lgbm_top5_pred_count[id] == 0:
            continue
        width = bar.get_width()
        ax.text(width-50,                      # X konumu (barın ortası)
                bar.get_y() + bar.get_height()/2,  # Y konumu (barın ortası)
                str(int(width)),              # Yazılacak değer
                ha='center', va='center', color='white', fontsize=12, fontweight='bold')

    ax.set_axisbelow(True)
    ax.xaxis.grid(color='gray', linestyle='dashed')
    ax.yaxis.grid(color='gray', linestyle='dashed')
    plt.tight_layout()

    if not path_control('../predictive-manufacturing-intelligence/results/lgbm/', is_file=False, is_directory=True)[0]:
        os.makedirs(os.path.dirname('../predictive-manufacturing-intelligence/results/lgbm/'), exist_ok=True)
    plt.savefig(f"../predictive-manufacturing-intelligence/results/lgbm/{next_day}.png", dpi=300)
