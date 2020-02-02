# читаем исходную таблицу
import subprocess as sub
import os
import yaml
from math import cos, sin, pi
from itertools import product
import pandas as pd
import numpy as np
import tables as tb
import tables.exceptions as exceptions
from time import sleep, time
import smtplib

# TODO выводить объём папки database


class Client:
    def __init__(self):
        self.last_email_time = 0
        self.total_time = time()
        self.wait = False
        self.tracker_dir = os.getenv("TRACKER_DIR")
        self.tracker_mac_dir = os.getenv("TRACKER_MAC_DIR")
        self.tracker_output_dir = os.getenv("TRACKER_OUTPUT_DIR")
        self.scripts_dir = os.getenv("TRACKER_SCRIPTS_DIR")
        self.database_dir = os.getenv("TRACKER_DATABASE_DIR")
        try:
            if (self.tracker_output_dir is None or
                    self.tracker_mac_dir is None or
                    self.tracker_dir is None or
                    self.database_dir is None or
                    self.scripts_dir is None):
                raise Exception("""Should be specified all environment variables: 
                                    TRACKER_DIR={}
                                    TRACKER_MAC_DIR={}
                                    TRACKER_OUTPUT_DIR={}
                                    TRACKER_SCRIPTS_DIR={}
                                    TRACKER_DATABASE_DIR={}""".format(self.tracker_dir,
                                                                      self.tracker_mac_dir,
                                                                      self.tracker_output_dir,
                                                                      self.scripts_dir,
                                                                      self.database_dir))
        except Exception as answer:
            print(answer)

        # настройки
        with open('input.yaml') as file:
            self.data = yaml.load(file, Loader=yaml.Loader)
        try:
            file = open('tmp/cache.yaml')
            data = yaml.load(file, Loader=yaml.Loader)
            self.queue_job_id = data['job_id']
            self.queue_set_id = data['set_id']
            self.total_time = data['time']
            file.close()
            if len(self.queue_job_id) != 0:
                self.wait = True
        except FileNotFoundError:
            self.queue_job_id = list()
            self.queue_set_id = list()
        # создать таблицу, в которой будет храниться информация о
        # сгенерированных событиях
        # если таблица существует, то будут использованы старые наборы параметров для частиц
        try:
            self.info = pd.read_hdf("info.h5")
        except FileNotFoundError:
            data = self.data
            combinations = list(product(data['particle'], data['direction']['azimuth'], data['direction']['zenith'],
                                        data['energy'], data['polarization']['x'],
                                        data['polarization']['y'], data['polarization']['z'],
                                        data['position']['x'], data['position']['y'], data['position']['z']))
            info = pd.DataFrame(combinations, columns=['particle', 'azimuth', 'zenith', 'energy',
                                                       'pol_x', 'pol_y', 'pol_z', 'pos_x', 'pos_y', 'pos_z'])
            self.info = info.assign(registered=np.zeros(info.shape[0]))
            self.info = self.info.assign(bad_events=np.zeros(info.shape[0]))
            self.info = self.info.assign(good_events=np.zeros(info.shape[0]))
            self.info = self.info.assign(generated=np.zeros(info.shape[0]))
            self.info = self.info.assign(number_of_jobs=np.zeros(info.shape[0]))
            self.info.to_hdf("info.h5", key='info')
        self.index_set = set(self.info.index)
        if self.data['verbose'] > 1:
            # test
            self.send_info()

    def send_info(self):
        # for sending emails
        message_server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        message_server.ehlo()
        message_server.login("gamma400messenger@gmail.com", "BIGblueelephant400")
        # test
        self.last_email_time = time()
        message_server.sendmail("gamma400messenger", self.data['email'], self.__repr__())
        message_server.close()

    # делаем запрос в очередь
    def start_query(self):
        # ставим в очередь строку с самым маленьким good_events и
        # которую ещё не поставили в очередь
        diff = self.index_set - set(self.queue_set_id)
        row = self.info.index[0]
        for idx in self.info.index:
            if (idx in diff) and (self.info.loc[idx, 'good_events'] < self.data['number_of_events']):
                row = idx
                break

        self.info.loc[row, "number_of_jobs"] += 1
        mac_name = self.tracker_mac_dir + os.sep + "{}.mac".format(row)
        try:
            mac_file = open(mac_name, 'r')
        except FileNotFoundError:
            # конвертируем сферические координаты направления импульса в декартовы
            pars = self.info.loc[row, :]
            zen = pars['zenith'] * pi / 180
            az = pars['azimuth'] * pi / 180
            z = sin(zen) * cos(az)
            y = sin(zen) * sin(az)
            x = cos(zen)
            direction = (x, y, z)

            mac_file = open(mac_name, 'w')
            mac_file.write("""
/control/verbose 0
/run/verbose 0
/event/verbose 0
/tracking/verbose 0
/gun/particle {}
/gun/energy {} MeV
/gun/position {} {} {} cm
/gun/direction {} {} {}
/gun/polarization {} {} {} 
/run/beamOn {}
                        """.format(pars['particle'],
                                   pars['energy'],
                                   pars['pos_x'],
                                   pars['pos_y'],
                                   pars['pos_z'],
                                   *direction,
                                   pars['pol_x'],
                                   pars['pol_y'],
                                   pars['pol_z'],
                                   self.data['batch_size']
                                   ))
        mac_file.close()

        file = open("tmp/query.sh", 'w')
        file.write("""#! /bin/bash
#PBS -q {queue}
#PBS -l ncpus={n}
#PBS -l walltime={walltime}:00:00
#PBS -N database_hauska
#PBS -e {scripts_dir}/log_err
#PBS -o {scripts_dir}/log_out

python {scripts_dir}/server.py {mac} {n} > {scripts_dir}/log_server.txt
""".format(scripts_dir=self.scripts_dir, mac="{}.mac".format(row), n=self.data['number_of_cores'],
                   queue=self.data['queue'], walltime=self.data['walltime']))
        file.close()

        process = sub.Popen(["qsub", "tmp/query.sh"], stdout=sub.PIPE, text=True)
        job_id = process.communicate()[0]
        job_id = job_id.split(sep='.')
        job_id = job_id[0] + '.' + job_id[1]
        self.queue_job_id.append(job_id)
        self.queue_set_id.append(row)

    def check_queue(self):
        for i, job in enumerate(self.queue_job_id):
            process = sub.Popen(["qstat", job], stdout=sub.PIPE, text=True)
            answer = process.communicate()[0]
            if len(answer) == 0:
                self.queue_job_id.pop(i)
                self.queue_set_id.pop(i)
        return len(self.queue_job_id)

    def update_info(self):
        files = os.scandir(self.tracker_output_dir)
        for file_name in files:
            # странно, но это работало и без .path
            try:
                file = tb.open_file(file_name.path)
                set_number = file.root.data._v_attrs.P_Set_number[0]
                bad_events = file.root.data._v_attrs.N_Bad_events[0]
                good_events = file.root.data._v_attrs.N_Good_events[0]
                generated = file.root.data._v_attrs.N_Generated_events[0]
                file.close()
                self.info.loc[set_number, 'generated'] += generated
                self.info.loc[set_number, 'bad_events'] += bad_events
                self.info.loc[set_number, 'good_events'] += good_events
                self.info.loc[set_number, 'registered'] += (good_events + bad_events)
                sub.call(['mv', file_name.path, self.database_dir + os.sep])
            except FileNotFoundError:
                pass
            except exceptions.NoSuchNodeError:
                sub.call(['rm', file_name.path])

        self.info.to_hdf("info.h5", key='info')

    def check_iteration(self):
        self.info = self.info.sort_values(by='good_events')
        # если самая мальенькая good_events достигла необходимого значения
        # то прекращаем цикл
        if self.info.iloc[0]['good_events'] >= self.data['number_of_events']:
            raise StopIteration

    def __repr__(self):
        answer = "Running {} hours\n".format(round(-(self.total_time - time()) / 3600, 1))
        answer += self.info.head().__repr__()
        answer += os.linesep
        answer += "Now running: {} jobs".format(len(self.queue_job_id))
        answer += os.linesep
        total_events = self.info.loc[:, 'generated'].sum()
        total_registered = self.info.loc[:, 'registered'].sum()
        total_jobs = self.info.loc[:, 'number_of_jobs'].sum()
        total_bad_events = self.info.loc[:, 'bad_events'].sum()
        total_good_events = self.info.loc[:, 'good_events'].sum()
        total_events_by_energy = self.info.groupby(by='energy').aggregate({'generated': np.sum,
                                                                           'good_events': np.sum,
                                                                           'bad_events': np.sum,
                                                                           'registered': np.sum})
        total_events_by_zenith = self.info.groupby(by='zenith').aggregate({'generated': np.sum,
                                                                           'good_events': np.sum,
                                                                           'bad_events': np.sum,
                                                                           'registered': np.sum})
        answer += """
Number of jobs              {}
Number of generated events: {}
Number of registered events {}
Number of good events:      {}
Number of bad events:       {}\n
        """.format(total_jobs, total_events, total_registered, total_good_events, total_bad_events)
        answer += total_events_by_energy.__repr__()
        answer += os.linesep
        answer += total_events_by_zenith.__repr__()
        answer += os.linesep
        return answer

    def __iter__(self):
        return self

    def __next__(self):
        n = self.check_queue()
        if n == 0:
            self.update_info()
            self.check_iteration()
            self.start_query()
            self.wait = False
        elif n < self.data['query_size'] and not self.wait:
            self.start_query()
        elif n == self.data['query_size'] and not self.wait:
            self.wait = True
            # update number of jobs
            self.info.to_hdf("info.h5", key='info')
        else:
            sleep(60 * self.data['update_duration'])
            # update cache
            with open("tmp/cache.yaml", 'w') as file:
                yaml.dump({'job_id': self.queue_job_id,
                           'set_id': self.queue_set_id,
                           'time': self.total_time}, file)

        return self


if __name__ == "__main__":
    db = Client()
    for x in db:
        if db.data['verbose'] == 2 or db.data['verbose'] == 3:
            if (time() - db.last_email_time)/3600 >= db.data["email_duration"]:
                db.send_info()
        if db.data['verbose'] == 1 or db.data['verbose'] == 3:
            sub.call("clear")
            print(x)
    # by the end, send email also
    if db.data['verbose'] == 2 or db.data['verbose'] == 3:
        db.send_info()
