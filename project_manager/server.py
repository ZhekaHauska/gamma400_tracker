# start tracker with specific .mac and output dir n times
import subprocess as sub
import os
import sys

# читаем переменную TRACKER_DIR, чтобы узнать, где хранится программа


def start_tracker(mac_file_name, n_processes):
    tracker_dir = os.getenv("TRACKER_DIR")
    scripts_dir = os.getenv("TRACKER_SCRIPTS_DIR")
    if tracker_dir is None or scripts_dir is None:
        raise Exception("Should be specified all environment variables.")

    # запускаем с параметрами
    processes = list()
    with open(scripts_dir + os.sep + 'log.txt', 'w') as log:
        for i in range(n_processes):
            process = sub.Popen([tracker_dir + os.sep + "tracker", mac_file_name], stdout=log, stderr=log)
            processes.append(process)

        for p in processes:
            p.wait()


if __name__ == "__main__":
    start_tracker(sys.argv[1], int(sys.argv[2]))
