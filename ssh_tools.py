import os
from datetime import datetime
from collections import defaultdict
from glob import glob
# import logging
from inspect import currentframe, getframeinfo

from stdo import stdo, get_time
import pandas as pd
import paramiko

from tools import path_control
from data_frame_tools import convert_Date_Using_Read_Component, save_To_CSV


def ssh_List_And_Extract_Metadata(hostname, port, username, password, remote_root_dir, file_ext=".png", csv_columns=[]):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    stdo(1, "ssh Policy is Set!")

    try:

        ssh.connect(hostname, port=port, username=username, password=password)
        stdo(1, f"ssh Connection is Successful!: {hostname}:{port} - {username}")

        # Bugünün tarihini klasör formatına uygun al
        if os.name == 'nt':  # Windows
            # today_str = datetime.now().strftime("%Y_%#m_%#d")
            current_date = "2025_8_7"
        else:  # Linux / Mac
            current_date = datetime.now().strftime("%Y_%-m_%-d")  # Linux/Mac

        # command = f"find " + remote_root_dir + " -type f -path " + "'*/" + current_date + "/*'" +  " -name " + "'*" + file_ext + "'"
        # command  = f'find "{remote_root_dir}" -type f -path "*/{current_date}/*" -name "*{file_ext}"'
        # command = f"find /home/mic-710ailx/Workspace/pcb-component-inspection-v3-CAIC-v1/dataset/ -type f -path */2025_8_7/* -name *.png"

        # current_date = datetime.now().strftime("%d-%m-%Y")

        csv_filename = os.path.join("dataset", f"remote_pcb_dataset_{current_date}.csv")
        # Mevcut CSV'deki date değerlerini yükle
        if path_control(csv_filename, is_file=True, is_directory=False)[0]:
            existing_dates = set(pd.read_csv(csv_filename, usecols=["Date"])["Date"].unique())
            stdo(1, f"CSV’de mevcut {len(existing_dates)} tarih yüklendi.")
        else:
            existing_dates = set()

        command = "/home/mic-710ailx/Workspace/counter_path.sh"
        stdo(1, f"ssh Execution Command is running!: bash {command} {current_date}")
        stdin, stdout, stderr = ssh.exec_command(f"bash {command} {current_date}")

        data = []
        count_map = defaultdict(int)
        total_added = 0
        for line in iter(stdout.readline, ""):
            line = line.strip()
            if line:  # boş satır yok
                # stdo(1, f"New read line: {line}")  # isteğe bağlı debug/log

                path = line.strip()
                parts = path.split('/')

                # component = date klasörünün bir üstü
                card_type = parts[0]   # PCB adı
                component = parts[1]   # bir üst klasör
                label = parts[2]
                full_date_time = convert_Date_Using_Read_Component(parts[3])
                date = parts[3].split("-")[0]  # sadece tarih kısmı

                # Eğer date CSV’de zaten varsa atla
                if date in existing_dates:
                    continue

                key = (card_type, date)
                count_map[key] += 1
                stdo(1, f"[+] PCB: {card_type:<10} | Date: {date} | Component:{component} | Current Count: {count_map[key]}")

                data.append([card_type, component, label, full_date_time])

                # Batch işleme
                # if len(data) >= batch_size:
                df = pd.DataFrame(data, columns=csv_columns)
                save_To_CSV(df, csv_filename)
                total_added += len(df)
                data = []

        # Kalanları ekle
        # if data:
        #     df = pd.DataFrame(data, columns=csv_columns + ["Date"])
        #     safe_to_csv(df, csv_filename)
        #     total_added += len(df)

        stdo(1, f"ssh_List_And_Extract_Metadata is Completed. Total Added Number: {total_added}")

        # err = stderr.read().decode()
        # stdo(1, f"stderr çıktı: {err if err else 'YOK'}")
        # if err:
        #     stdo(1, f"ssh_List_And_Extract_Metadata-Error: {err}")

        # current_date = datetime.now().strftime("%d-%m-%Y")
        # csv_filename = os.path.join("dataset", f"remote_pcb_dataset_{current_date}.csv")

        # data = []
        # count_map = defaultdict(int)
        # total_added = 0

        # for line in stdout:
        #     path = line.strip()
        #     parts = path.split('/')
        #     try:
        #         # component = date klasörünün bir üstü
        #         card_type = parts[-5]   # PCB adı
        #         component = parts[-4]   # bir üst klasör
        #         date = parts[-3]
        #         label = parts[-2]
        #         full_date_time = parts[-1].replace(file_ext, "")
        #         full_date_time = convert_Date(full_date_time)

        #         # Eğer date CSV’de zaten varsa atla
        #         if date in existing_dates:
        #             continue

        #         key = (card_type, date)
        #         count_map[key] += 1
        #         stdo(1, f"[+] PCB: {card_type:<10} | Date: {date} | Current Count: {count_map[key]}")

        #         data.append([card_type, component, label, full_date_time, date])

        #         # Batch işleme
        #         if len(data) >= batch_size:
        #             df = pd.DataFrame(data, columns=csv_columns + ["date"])
        #             safe_to_csv(df, csv_filename)
        #             total_added += len(df)
        #             data = []

        #     except IndexError:
        #         stdo(1, f"ssh_List_And_Extract_Metadata-Unexpected Index: {path}")

        # # Kalanları ekle
        # if data:
        #     df = pd.DataFrame(data, columns=csv_columns + ["date"])
        #     safe_to_csv(df, csv_filename)
        #     total_added += len(df)

        # stdo(1, f"ssh_List_And_Extract_Metadata tamamlandı. Toplam eklenen satır: {total_added}")

    finally:
        ssh.close()
