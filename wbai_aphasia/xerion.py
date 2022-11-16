# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division
from six.moves.urllib import request

import glob
import os
import platform  # Mac or Linux special for uncompress command
import errno
import sys
import numpy as np
import codecs
import re
import subprocess
import sys
import tarfile
import matplotlib.pyplot as plt

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

class xerion(object):
    """Managing xerion datafiles.

    Read datafiles ending with '-nsyl.ex' and '-syl.ex' from `xerion_prefix/datadir`, and
    SAve them to `pkl_dir` as pickle files.

    Usage:
    ```python
    print(xerion().Orthography)  # for input data format
    print(xerion().Phonology)    # for output data format
    X = xerion().input
    y = xerion().output
    ```

    The original datafiles can be obtained from http://www.cnbc.cmu.edu/~plaut/xerion/
    """

    def __init__(self,
                 data='SM-nsyl',
                 datadir='./data/',
                 pkl_dir='./data/',
                 remake=False, readall=False, saveall=False,
                 forceDownload=False):
        self.module_path = os.path.dirname(__file__)
        self.xerion_prefix = 'nets/share/'
        self.datadir = datadir # + self.xerion_prefix
        self.pkl_dir = pkl_dir
        self.url_base = 'http://www.cnbc.cmu.edu/~plaut/xerion/'
        self.url_file = 'xerion-3.1-nets-share.tar.gz'
        self.origfile_size = 1026691
        self.syl_files = ['SM-syl.ex', 'besnerNW-syl.ex', 'bodies-syl.ex', 'bodiesNW-syl.ex',
                          'friedmanNW-syl.ex', 'glushkoNW-syl.ex', 'graphemes-syl.ex',
                          'jared1-syl.ex', 'jared2-syl.ex', 'megaNW-syl.ex',
                          'pureNW-syl.ex', 'surface-syl.ex', 'taraban-syl.ex',
                          'tarabanALL-syl.ex', 'tarabanEvN-syl.ex', 'tarabanNRE-syl.ex',
                          'vcoltheartNW-syl.ex']
        self.nsyl_files = ['SM-nsyl.ex', 'besnerNW-nsyl.ex', 'glushkoNW-nsyl.ex',
                          'graphemes-nsyl.ex', 'jared1-nsyl.ex', 'jared2-nsyl.ex',
                          'markPH-nsyl.ex', 'megaNW-nsyl.ex', 'surface-nsyl.ex',
                          'taraban-nsyl.ex', 'tarabanALL-nsyl.ex', 'tarabanEvN-nsyl.ex',
                          'tarabanNRE-nsyl.ex']
        self.datafilenames = [ *self.nsyl_files, *self.syl_files]
        self._tags = ('#', 'seq', 'grapheme', 'phoneme', 'freq', 'tag', 'input', 'output')
        self.Orthography={'onset':['Y', 'S', 'P', 'T', 'K', 'Q', 'C', 'B', 'D', 'G',
                                   'F', 'V', 'J', 'Z', 'L', 'M', 'N', 'R', 'W', 'H',
                                   'CH', 'GH', 'GN', 'PH', 'PS', 'RH', 'SH', 'TH', 'TS', 'WH'],
                          'vowel':['E', 'I', 'O', 'U', 'A', 'Y', 'AI', 'AU', 'AW', 'AY', 
                                   'EA', 'EE', 'EI', 'EU', 'EW', 'EY', 'IE', 'OA', 'OE', 'OI', 
                                   'OO', 'OU', 'OW', 'OY', 'UE', 'UI', 'UY'],
                          'coda':['H', 'R', 'L', 'M', 'N', 'B', 'D', 'G', 'C', 'X', 
                                  'F', 'V', '∫', 'S', 'Z', 'P', 'T', 'K', 'Q', 'BB', 
                                  'CH', 'CK', 'DD', 'DG', 'FF', 'GG', 'GH', 'GN', 'KS', 'LL', 
                                  'NG', 'NN', 'PH', 'PP', 'PS', 'RR', 'SH', 'SL', 'SS', 'TCH', 
                                  'TH', 'TS', 'TT', 'ZZ', 'U', 'E', 'ES', 'ED']}
        self.Phonology={'onset':['s', 'S', 'C', 'z', 'Z', 'j', 'f', 'v', 'T', 'D', 
                                 'p', 'b', 't', 'd', 'k', 'g', 'm', 'n', 'h', 'I', 
                                 'r', 'w', 'y'],
                        'vowel': ['a', 'e', 'i', 'o', 'u', '@', '^', 'A', 'E', 'I', 
                                  'O', 'U', 'W', 'Y'],
                        'coda':['r', 'I', 'm', 'n', 'N', 'b', 'g', 'd', 'ps', 'ks', 
                                'ts', 's', 'z', 'f', 'v', 'p', 'k', 't', 'S', 'Z', 
                                'T', 'D', 'C', 'j']}
        self.bibtex=[
            '@article{SM89,',
            'title={A Distributed, Developmental Model of Word Recognition and Naming},',
            'auhor={Mark S. Seidenberg and James L. McClelland},',
            'year={1989},',
            'journal={psychological review},',
            'volume={96},',
            'number={4},',
            'pages={523-568}}'
            '}',
            '@article{PMSP96,',
            'title={Understanding Normal and Impaired Word Reading:',
            ' Computational Principles in Quasi-Regular Domains},',
            'author={David C. Plaut and James L. McClelland and Mark S. Seidenberg and Karalyn Patterson},',
            'year={1996},',
            'volume={103},',
            'number={1},',
            'pages={56-115},',
            'journal={psychological review}',
            '}']
        self.dbs = {}
        if remake:
            self.dbs = self.make_all()
            saveall = True
        if saveall == True:
            self.save_all()
            readall = True
        if readall:
            self.dbs = self.read_all()
        self.dataname = data
        pkl_file = self.pkl_dir + self.dataname + '.pkl'
        self.db = self.load_pickle(filename=pkl_file)
        self.input = self.db[self._tags.index('input')]
        self.output = self.db[self._tags.index('output')]
        self.freq = self.db[self._tags.index('freq')]
        self.graph = self.db[self._tags.index('grapheme')]
        self.phone = self.db[self._tags.index('phoneme')]
        self.tag = self.db[self._tags.index('tag')]
        self.dbs[self.dataname] = self.db

    #def read_a_xerion_file(filename='SM-nsyl.pkl'):
    #    pass

    def read_all(self):
        """reading data files named ening with '-nsyl.ex'."""
        dbs = {}
        for dname in self.datafilenames:
            dname_ = re.sub('.ex', '', dname)
            filename = self.pkl_dir + dname_ + '.pkl'
            if not os.path.isfile(filename):
                raise ValueError('{0} could not found'.format(filename))
            dbs[dname_] = self.load_pickle(filename=filename)
        return dbs

    def save_all(self):
        """saving data files to be pickled."""

        dirname = self.pkl_dir
        if not os.path.exists(self.pkl_dir):
            os.makedirs(self.pkl_dir)
            if not os.path.exists(self.pkl_dir):
                raise OSError('{} was not found'.format(self.pkl_dir))
        for db in self.dbs:
            dest_filename = self.pkl_dir + re.sub('.ex', '.pkl', db)
            try:
                with codecs.open(dest_filename, 'wb') as f:
                    pickle.dump(self.dbs[db], f)
            except:
                print('Error in processing {0}'.format(dest_filename))

    def load_pickle(self, filename='SM-nsyl.pk'):
        if not os.path.isfile(filename):
            raise ValueError('Could not find {}'.format(filename))
        with open(filename, 'rb') as f:
            db = pickle.load(f)
        return db

    def make_all(self):
        dbs = {}
        for dname in self.datafilenames:
            filename = self.datadir + self.xerion_prefix + dname
            if not os.path.isfile(filename):
                print('{0} could not found'.format(filename))
                downfilename, h = self.download()
                #print('downloaded file: {0}, {1}'.format(downfilename, h))
                self.extractall()
            inp, out, graph, phone, freq, tags = self.read_xerion(filename=filename)
            dbs[dname] = [dname, '#', graph, phone, freq, tags, inp, out]
        return dbs

    def read_xerion(self, filename='../data/nets/share/SM-nsyl.ex'):
        with codecs.open(filename,'r') as f:
            lines = f.readlines()

        inp_flag = False
        inpbuff, outbuff, tags = {}, {}, {}
        graph, phone, freq = {}, {}, {}
        for i, line in enumerate(lines[1:]):
            if len(line) == 0:
                continue
            a = line.strip().split(' ')
            if line[0] == '#':
                if a[0] == '#WARNING:':
                    continue
                try:
                    seq = int(a[self._tags.index('seq')])
                except:
                    continue
                _graph = a[self._tags.index('grapheme')]
                _phone = a[self._tags.index('phoneme')]
                _freq = a[self._tags.index('freq')]
                _tag = a[self._tags.index('tag')]
                inp_flag = True
                if not seq in inpbuff:
                    inpbuff[seq] = list()
                    outbuff[seq] = list()
                    graph[seq] = _graph
                    phone[seq] = _phone
                    freq[seq] = _freq
                    tags[seq] = _tag
                continue
            elif line[0] == ',':
                inp_flag = False
                continue
            elif line[0] == ';':
                inp_flag = True
                continue
            if inp_flag:
                #print('hoge seq=', seq)
                for x in a:
                    try:
                        inpbuff[seq].append(int(x))
                    except:
                        pass  #print(x, end=', ')
            else:
                for x in a:
                    try:
                        outbuff[seq].append(int(x))
                    except:
                        pass
            continue

        ret_in = np.array([inpbuff[seq] for seq in inpbuff], dtype=np.int16)
        ret_out = np.array([outbuff[seq] for seq in outbuff], dtype=np.int16)
        ret_graph = np.array([graph[seq] for seq in graph], dtype=np.unicode_)
        ret_phone = np.array([phone[seq] for seq in phone], dtype=np.unicode_)
        ret_freq = np.array([freq[seq] for seq in freq], dtype=np.float32)
        ret_tag = np.array([tags[seq] for seq in tags], dtype=np.unicode_)
        return ret_in, ret_out, ret_graph, ret_phone, ret_freq, ret_tag


    @staticmethod
    def download(forcedownload=False, destdir=None):
        if destdir is None:
            destdir = self.datadir 
        if not os.path.exists(destdir):
            os.mkdir(destdir)
        dest_filename = destdir + self.url_file
        if os.path.exists(dest_filename):
            statinfo = os.stat(dest_filename)
            if statinfo.st_size != self.origfile_size:
                forceDownload = True
                print("File {} not expected size, forcing download".format(dest_filename))
            else:
                print("File '{}' allready downloaded.".format(dest_filename))
        if forcedownload == True or not os.path.exists(dest_filename):
            print('Attempting to download: {}'.format(dest_filename)) 
            print('From {}'.format(self.url_base + self.url_file))
            fname, h = request.urlretrieve(self.url_base+self.url_file, dest_filename)
            print("Downloaded '{}' successfully".format(dest_filename))
            return fname, h
        else:
            return dest_filename, None

    @staticmethod
    def extractall(gzfile=None):
        if gzfile is None:
            gzfile, _ = self.download()
        with tarfile.open(name=gzfile, mode='r:gz') as tar:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tar, path=self.datadir)
        if platform.system() == 'Darwin':
            cmd = '/usr/bin/uncompress'
            args = self.datadir + self.xerion_prefix + '*.ex.Z'
            files = glob.glob(args)
            for file in sorted(files):
                print(cmd, file)
                try:
                    subprocess.Popen([cmd, file])
                except:
                    print('cmd {0} {1} failed'.format(cmd, file))
                    sys.exit()
            print('#extractall() completed. command:{}'.format(cmd))
        else:
            print('You must on Linux or Windows, Please uncompress manually')
            sys.exit()
        self.pkl_dir = self.datadir + self.xerion_prefix


    def note(self):
        print('\n\n# xerion() is the data management tool for PMSP96')
        print('# The original data will be found at:',
              self.url_base + self.url_file)
        print('# The data format is as following:')
        for l in [self.Orthography, self.Phonology]:
            for x in l:
                print(x, l[x])
        print('\n# The bibtex format of the original papers:')
        for l in self.bibtex:
            print(l)

    @staticmethod
    def usage():
        print('```python')
        print('import numpy')
        print('import wbai_aphasia as handson')
        print()
        print('from sklearn.neural_network import MLPRegressor')
        print()
        print('data = handson.xerion()')
        print('X = np.asarray(data.input, dtype=np.float32)')
        print('y = np.asarray(data.output, dtype=np.float32)')
        print()
        print('model = MLPRegressor()')
        print('model.fit(X,y)')
        print('model.score(X,y)')
        print('```')

    def descr(self):
        fdescr_name = os.path.join(self.module_path, 'descr', 'xerion.md')
        print('self.module_path={}'.format(self.module_path))
        print('fdescr_name={}'.format(fdescr_name))

        with codecs.open(fdescr_name, 'r') as markdownfile:
            fdescr = markdownfile.read()
        print(fdescr)
