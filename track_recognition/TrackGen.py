from os import scandir
import yaml
from itertools import product
from pandas import read_hdf
import tables as tb
from keras.preprocessing import sequence
import numpy as np
from math import ceil
import pickle


class FileManager:
    def __init__(self, directory, mode):
        """
        :param directory: where search for data
        :param mode: which type of ann will be used
        -- tracker
        -- filter
        -- divider
        """

        self.dir = directory
        self.mode = mode
        files = scandir(self.dir)

        with open('input.yaml') as file:
            self.data = yaml.load(file, Loader=yaml.Loader)

        try:
            settings = open('temp/settings.yaml')
            data = yaml.load(settings, Loader=yaml.Loader)
            settings.close()
            self.load_cache = (data == self.data)
        except FileNotFoundError:
            self.load_cache = False

        self.batch_size = self.data['batch_size']

        self.info = None
        try:
            self.info = read_hdf("info.h5")
        except FileNotFoundError:
            self.update_info()

        if self.load_cache:
            with open("temp/cache.pkl", 'rb') as file:
                cache = pickle.load(file)
            self.sets = cache['sets']
            self.micro_batch = cache['micro_batch']
            self.iterations_valid = cache['iterations_valid']
            self.iterations_train = cache['iterations_train']
            self.expected_iterations_valid = cache['expected_iterations_valid']
            self.expected_iterations_train = cache['expected_iterations_train']
            self.total_events_valid = cache['total_events_valid']
            self.total_events_train = cache['total_events_train']
            self.valid_files_pool = cache['valid_files_pool']
            self.valid_files_pool_bad = cache['valid_files_pool_bad']
            self.train_files_pool = cache['train_files_pool']
            self.train_files_pool_bad = cache['train_files_pool_bad']
            print("Cache loaded.")
        else:
            # forming sets
            combinations = list(
                product(self.data['particle'], self.data['direction']['azimuth'], self.data['direction']['zenith'],
                        self.data['energy'], self.data['polarization']['x'],
                        self.data['polarization']['y'], self.data['polarization']['z'],
                        self.data['position']['x'], self.data['position']['y'], self.data['position']['z']))
            self.sets = list()
            for c in combinations:
                idx = self.info.query(
                    "particle=='{}' & azimuth=={} & zenith=={} & energy=={} & pol_x=={} & pol_y=={} & pol_z=={} & "
                    "pos_x=={} "
                    "& pos_y=={} & pos_z=={}".format(
                        *c
                    )).index
                if idx.empty:
                    raise Exception("""There is no such combination: 
                                            particle=={} & azimuth=={} & zenith=={} & energy=={} & 
                                            pol_x=={} & pol_y=={} & pol_z=={} & 
                                            pos_x=={} & pos_y=={} & pos_z=={}""".format(*c))
                else:
                    self.sets.append(idx[0])

            self.train_files_pool = dict(zip(self.sets, [list() for _ in range(len(self.sets))]))
            self.valid_files_pool = dict(zip(self.sets, [list() for _ in range(len(self.sets))]))
            self.train_files_pool_bad = dict(zip(self.sets, [list() for _ in range(len(self.sets))]))
            self.valid_files_pool_bad = dict(zip(self.sets, [list() for _ in range(len(self.sets))]))
            events_number_train = dict(zip(self.sets, [0 for _ in self.sets]))
            events_number_valid = dict(zip(self.sets, [0 for _ in self.sets]))
            events_number_train_bad = dict(zip(self.sets, [0 for _ in self.sets]))
            events_number_valid_bad = dict(zip(self.sets, [0 for _ in self.sets]))
            total_events_number_train = self.data['number_of_events'] * (1 - self.data['split_data_ratio'])
            total_events_number_valid = self.data['number_of_events'] * self.data['split_data_ratio']

            if self.data['type_of_events'] == 'good' or self.data['type_of_events'] == 'bad':
                self.micro_batch = ceil(self.batch_size / len(self.sets))
                expected_events_number_train = ceil(total_events_number_train / len(self.sets))
                expected_events_number_valid = ceil(total_events_number_valid / len(self.sets))
            else:
                self.micro_batch = ceil(self.batch_size / (2 * len(self.sets)))
                expected_events_number_train = ceil(total_events_number_train / (2 * len(self.sets)))
                expected_events_number_valid = ceil(total_events_number_valid / (2 * len(self.sets)))

            # создание пула из файлов для каждого сета
            print("Manager is preparing files for generators.")
            for file_posix in files:
                file_name = file_posix.path
                file = tb.open_file(file_name)
                set_number = file.root.data._v_attrs.P_Set_number[0]

                events_number_good = file.root.data._v_attrs.N_Good_events[0]
                events_number_bad = file.root.data._v_attrs.N_Bad_events[0]
                valid_ready = False
                valid_bad_ready = False
                if set_number in self.sets:
                    if self.data['type_of_events'] == 'good' or self.data['type_of_events'] == 'both':
                        if events_number_train[set_number] < expected_events_number_train:
                            self.train_files_pool[set_number].append(file_name)
                            events_number_train[set_number] += events_number_good
                        elif events_number_valid[set_number] < expected_events_number_valid:
                            self.valid_files_pool[set_number].append(file_name)
                            events_number_valid[set_number] += events_number_good
                        else:
                            valid_ready = True
                    if self.data['type_of_events'] == 'bad' or self.data['type_of_events'] == 'both':
                        if events_number_train_bad[set_number] < expected_events_number_train:
                            self.train_files_pool_bad[set_number].append(file_name)
                            events_number_train_bad[set_number] += events_number_bad
                        elif events_number_valid_bad[set_number] < expected_events_number_valid:
                            self.valid_files_pool_bad[set_number].append(file_name)
                            events_number_valid_bad[set_number] += events_number_bad
                        else:
                            valid_bad_ready = True
                file.close()
                if valid_ready and self.data['type_of_events'] == 'good':
                    break
                elif valid_bad_ready and self.data['type_of_events'] == 'bad':
                    break
                elif valid_ready and valid_bad_ready:
                    break
            # проверить, что хватило событий
            if (min(events_number_train.values()) < expected_events_number_train) and\
                    (self.data['type_of_events'] != 'bad'):
                raise Exception("There is no enough good train events for this setup, change setup"
                                "or watch info.h5 for additional information.")
            if (min(events_number_train_bad.values()) < expected_events_number_train) and\
                    (self.data['type_of_events'] != 'good'):
                raise Exception("There is no enough bad train events for this setup, change setup"
                                "or watch info.h5 for additional information.")
            if (min(events_number_valid.values()) < expected_events_number_valid) and\
                    (self.data['type_of_events'] != 'bad'):
                raise Exception("There is no enough good valid events for this setup, change setup"
                                "or watch info.h5 for additional information.")
            if (min(events_number_valid_bad.values()) < expected_events_number_valid) and\
                    (self.data['type_of_events'] != 'good'):
                raise Exception("There is no enough bad valid events for this setup, change setup"
                                "or watch info.h5 for additional information.")
            # сколько можно делать итераций по каждому из генераторов
            self.total_events_train = min(events_number_train.values()) *\
                                         (len(events_number_train.values()) if (self.data['type_of_events'] == 'good' or
                                                                                self.data['type_of_events'] == 'both') else 0  
                                          + len(events_number_train_bad.values()) if (self.data['type_of_events'] == 'bad' or
                                                                                      self.data['type_of_events'] == 'both') else 0)
            self.total_events_valid = min(events_number_valid.values()) *\
                                         (len(events_number_valid.values()) if (self.data['type_of_events'] == 'good' or
                                                                                self.data['type_of_events'] == 'both') else 0  
                                          + len(events_number_valid_bad.values()) if (self.data['type_of_events'] == 'bad' or
                                                                                      self.data['type_of_events'] == 'both') else 0)
            self.iterations_train = int(self.total_events_train / self.data['batch_size'])
            self.iterations_valid = int(self.total_events_valid / self.data['batch_size'])
            self.expected_iterations_train = int(total_events_number_train / self.data['batch_size'])
            self.expected_iterations_valid = int(total_events_number_valid / self.data['batch_size'])

            with open('temp/cache.pkl', 'wb') as file, open('temp/settings.yaml', 'w') as settings:
                to_dump = {'micro_batch': self.micro_batch,
                           'total_events_valid': self.total_events_valid, 'total_events_train': self.total_events_train,
                           'iterations_train': self.iterations_train, 'iterations_valid': self.iterations_valid,
                           'expected_iterations_train': self.expected_iterations_train,
                           'expected_iterations_valid': self.expected_iterations_valid,
                           'valid_files_pool': self.valid_files_pool, 'valid_files_pool_bad': self.valid_files_pool_bad,
                           'train_files_pool': self.train_files_pool, 'train_files_pool_bad': self.train_files_pool_bad,
                           'sets': self.sets}

                pickle.dump(to_dump, file)
                yaml.dump(self.data, settings)
            print("Done.")

    def get_train_gen(self):
        return TrackGen(self.train_files_pool, self.train_files_pool_bad, self.micro_batch,
                        self.iterations_train, self.expected_iterations_train, self.data['type_of_events'], self.mode)
    
    def get_valid_gen(self):
        return TrackGen(self.valid_files_pool, self.valid_files_pool_bad, self.micro_batch,
                        self.iterations_valid, self.expected_iterations_valid, self.data['type_of_events'], self.mode)

    # TODO просномтреть все файлы с данными и собрать самую полную информацию
    def update_info(self):
        pass


class TrackGen:
    def __init__(self, files_pool_good, files_pool_bad, micro_batch,
                 max_iterations, expected_iterations, type_of_events, mode):
        self.stop = False
        self.iteration = 0
        self.type = type_of_events
        self.max_iterations = max_iterations
        self.expected_iterations = expected_iterations
        self.micro_batch = micro_batch
        self.sets = list()
        self.files_pool_good = files_pool_good
        self.files_pool_bad = files_pool_bad
        self.mode = mode
        if self.type == 'good' or self.type == 'both':
            for s in self.files_pool_good.keys():
                self.sets.append(ParSet(s, self.files_pool_good[s], self.micro_batch, ev_type='good', mode=self.mode))
        if self.type == 'bad' or self.type == 'both':
            for s in self.files_pool_bad.keys():
                self.sets.append(ParSet(s, self.files_pool_bad[s], self.micro_batch, ev_type='bad', mode=self.mode))

    def __len__(self):
        return self.expected_iterations

    def __next__(self):
        x = list()
        y = list()
        if self.stop:
            self.load()
        for s in self.sets:
            data = next(s)
            x.extend(data[0])
            y.extend(data[1])
        self.iteration += 1
        if self.iteration == self.expected_iterations:
            self.stop = True
            
        x = sequence.pad_sequences(x, dtype=np.float32, value=-1.0)
        y = sequence.pad_sequences(y, dtype=np.float32, value=-1.0)
        idx = np.arange(x.shape[0])
        np.random.shuffle(idx)
        return x[idx], y[idx]

    def __iter__(self):
        return self

    def load(self):
        self.iteration = 0
        self.sets = list()
        if self.type == 'good' or self.type == 'both':
            for s in self.files_pool_good.keys():
                self.sets.append(ParSet(s, self.files_pool_good[s], self.micro_batch, ev_type='good', mode=self.mode))
        if self.type == 'bad' or self.type == 'both':
            for s in self.files_pool_bad.keys():
                self.sets.append(ParSet(s, self.files_pool_bad[s], self.micro_batch, ev_type='bad', mode=self.mode))


class ParSet:
    def __init__(self, set_id, file_pool, batch_size, ev_type='good', mode='tracker'):
        self.mode = mode
        self.set_id = set_id
        self.file_pool = iter(file_pool)
        self.batch_size = batch_size
        self.ev_type = ev_type
        self.pos = {'file': 0, 'batch': 0}
        self.data = None
        self.cur_file = None
        self.table = None
        self.buffer = dict()

        self.reload()
        self.next_chunk()

    def next_chunk(self):
        self.data = self.get_data(self.table, self.pos['file'])
        self.pos['file'] += 1
        self.pos['batch'] = 0

    def reload(self):
        self.cur_file = tb.open_file(next(self.file_pool), 'r')
        if self.ev_type == 'good':
            self.table = self.cur_file.root.data.good_events.strips
        else:
            self.table = self.cur_file.root.data.bad_events.strips
        self.pos = {'file': 0, 'batch': 0}

    def __iter__(self):
        return self

    def __next__(self):
        pos = self.pos['batch']
        x = self.data['x'][(pos * self.batch_size): ((pos+1) * self.batch_size)]
        y = self.data['y'][(pos * self.batch_size): ((pos+1) * self.batch_size)]
        self.pos['batch'] += 1
        if len(x) < self.batch_size:
            if self.data['is_end']:
                self.cur_file.close()
                self.reload()
            self.next_chunk()
            change = self.batch_size - len(x)
            x.extend(self.data['x'][:change])
            y.extend(self.data['y'][:change])
            self.data['x'] = self.data['x'][change:]
            self.data['y'] = self.data['y'][change:]
        return x, y

    def get_data(self, table, pos):
        is_end = False
        chunk = table.read(pos * table.chunkshape[0], (pos + 1) * table.chunkshape[0])
        if chunk.shape[0] < table.chunkshape[0]:
            is_end = True
        chunk.sort(order=['event_id', 'layer_id', 'axis', 'strip_id'])
        # data features
        x = np.array([chunk['strip_id'].astype('float64'),
                      chunk['axis'].astype('float64'),
                      chunk['layer_id'].astype('float64'),
                      chunk['dep_energy'].astype('float64')]).transpose()

        s = np.unique(chunk['event_id'], return_index=True)
        x = np.split(x, s[1][1:], axis=0)
        # data labels: the first position e- track and second is for e+
        if self.mode == 'tracker':
            labels1 = np.asarray(((chunk['strip_type'] == 1) | (chunk['strip_type'] == 2)), dtype=np.float32)
            labels2 = np.asarray(((chunk['strip_type'] == -1) | (chunk['strip_type'] == 2)), dtype=np.float32)
            labels = np.concatenate([labels1[:, None], labels2[:, None]], axis=-1)
            y = np.split(labels, s[1][1:], axis=0)
        elif self.mode == 'filter':
            labels = np.asarray(((chunk['strip_type'] == -1) | (chunk['strip_type'] == 1) | (chunk['strip_type'] == 2)),
                                dtype=np.float32)
            labels = labels[:, None]
            y = np.split(labels, s[1][1:], axis=0)
        elif self.mode == 'divider':
            # TODO Implement divider mode
            raise NotImplementedError('divider mode is not implemented yet')

        if len(self.buffer) != 0:
            if self.buffer['event_id'] == s[0][0]:
                x[0] = np.concatenate([x[0], self.buffer['x']], axis=0)
                y[0] = np.concatenate([y[0], self.buffer['y']], axis=0)
            else:
                x.insert(0, self.buffer['x'])
                y.insert(0, self.buffer['y'])
        if not is_end:
            self.buffer['x'] = x.pop()
            self.buffer['y'] = y.pop()
            self.buffer['event_id'] = s[0][-1]
        else:
            self.buffer.clear()

        return dict(is_end=is_end, x=x, y=y)


if __name__ == '__main__':
    manager = FileManager("/home/whosuka/Desktop/gamma_400_tracker_exp/data", mode='tracker')
    train_gen = manager.get_train_gen()
    valid_gen = manager.get_valid_gen()
    counter_train = 0
    counter_valid = 0
    print("Train generator")
    print("Total: {}".format(manager.total_events_train))
    for x, y in train_gen:
        pass
        counter_train += 1
    print("Valid generator")
    print("Total: {}".format(manager.total_events_valid))
    for x, y in valid_gen:
        pass
        counter_valid += 1
    print("counter train")
    print(manager.iterations_train, counter_train)
    print("counter valid")
    print(manager.iterations_valid, counter_valid)




