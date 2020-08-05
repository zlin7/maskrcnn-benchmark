import torch
import time, sys
import numpy as np
import pandas as pd
import os


try:
    sys.path.append(os.path.abspath("../.."))
    from global_settings import BASE_CNN_DIR, DATA_PATH, VARYING_DIST_DATA_PATH, CACHE_CNNFAILURE, FULL_DATA_PATH, FULL_DATA_LABEL_PATH
except Exception as err:
    raise err
    print(err)
    BASE_CNN_DIR = '/media/zhen/Data'
    DATA_PATH = "/media/zhen/Research/deepsz/"
    VARYING_DIST_DATA_PATH = '/media/zhen/Research/gitRes/deepsz_clean/deepsz/data/maps/split2_10x'
    CACHE_CNNFAILURE = '/media/zhen/Research/gitRes/deepsz_clean/deepsz/data/cache/CNNFailures.pkl'

class ProgressBar:
    def __init__(self, iterable, taskname=None, barLength=40, stride = 50):
        self.l = iterable
        try:
            self.n = len(self.l)
        except TypeError:
            self.l = list(self.l)
            self.n = len(self.l)
        self.cur = 0
        self.starttime = time.time()
        self.barLength = barLength
        self.taskname = taskname
        self.last_print_time = time.time()
        self.stride = stride

    def __iter__(self):
        return self
    def _update_progress(self):
        status = "Done...\r\n" if self.cur == self.n else "\r"
        progress = float(self.cur) / self.n
        curr_time = time.time()

        block = int(round(self.barLength * progress))
        text = "{}Percent: [{}] {:.2%} Used Time:{:.2f} seconds {}".format("" if self.taskname is None else "Working on {}. ".format(self.taskname),
                                                                      "#" * block + "-"*(self.barLength - block),
                                                                      progress, curr_time - self.starttime, status)
        sys.stdout.write(text)
        sys.stdout.flush()

    def __next__(self):
        if self.cur % self.stride == 0:
            self._update_progress()
        if self.cur >= self.n:
            raise StopIteration
        else:
            self.cur += 1
            return self.l[self.cur - 1]


#======================================================================================================================

from maskrcnn_benchmark.structures.bounding_box import BoxList

class DEEPSZ(object):

    def __init__(self, which='train', batch_size=128, component='skymap',
                 ratio=1,
                 data_path = DATA_PATH,
                 seed=0, min_redshift=0.25, min_mvir=2e14,
                 resolution=0.25, normalize=True, oversample_pos=False):

        self.batch_size = batch_size
        self.data_path = data_path
        self.resolution = resolution
        self.min_redshift = min_redshift
        self.min_mvir = min_mvir
        self.which = which
        self.ratio = ratio
        self.normalize = normalize
        np.random.seed(seed)

        self._thres_str = "z{:.2f}_mvir{:.0e}".format(self.min_redshift, self.min_mvir)
        self.map_dir = os.path.join(self.data_path, "maps", "reso{:.2f}{}".format(self.resolution, "_small"))

        label_path_fulll = os.path.join(self.map_dir, "%s_label.pkl" % self._thres_str)
        assert os.path.normpath(self.map_dir) == os.path.normpath(FULL_DATA_PATH)
        assert os.path.normpath(label_path_fulll) == os.path.normpath(FULL_DATA_LABEL_PATH)

        self.map_component_dir = os.path.join(self.map_dir, "%s(with noise)" % (component))

        labels_full = pd.read_pickle(label_path_fulll)
        labels_full = labels_full[labels_full['which'] == which]
        if ratio is not None:
            pos_idx = labels_full[labels_full['y']].index
            neg_idx = labels_full.index.difference(pos_idx)
            if isinstance(oversample_pos, float):
                pos_idx = np.random.choice(pos_idx, int(oversample_pos * len(pos_idx)))
                self.labels = pd.concat([labels_full.loc[pos_idx],
                                         labels_full.loc[neg_idx[:int(self.ratio * len(pos_idx))]]])
            elif oversample_pos:
                pos_idx = np.random.choice(pos_idx, int(np.round(len(neg_idx) / float(ratio))))
                self.labels = pd.concat([labels_full.loc[pos_idx],
                                         labels_full.loc[neg_idx]])
            else:
                self.labels = pd.concat([labels_full.loc[pos_idx],
                                         labels_full.loc[neg_idx[:int(self.ratio * len(pos_idx))]]])
        else:
            self.labels = labels_full
            #self.labels = self.labels.reindex([745127, 711543, 652187, 725538, 779098, 610791, 756482, 694541, 639157, 684694, 779313, 783420, 771023, 688510, 792005, 737365, 630390, 610236, 623942, 768256, 797426, 796630, 722213, 794239, 681922, 718554, 659097, 690119, 725680, 627057, 689949, 794097, 748150, 625318, 639356, 673807, 680020, 695141, 780673, 613418, 797538, 627094, 634836, 751077, 655968, 633174, 778846, 652306, 802191, 702306, 663615, 792176, 716652, 763825, 613819, 687360, 763728, 642119, 791610, 725895, 800529, 780530, 774479, 683360, 802244, 744058, 681829, 698689, 804774, 645002, 606962, 709616, 691468, 787999, 628054, 681888, 664730, 679831, 722327, 654789, 749303, 789017, 640129, 765770, 767240, 792062, 616841, 741020, 674346, 727751, 698071, 689274, 634184, 782418, 740211, 671763, 619959, 631580, 715968, 806389, 706132, 683002, 626317, 616260, 726944, 707889, 741128, 806990, 795625, 684002, 759549, 682469, 617201, 657666, 746421, 727432, 614079, 635235, 785486, 638631, 669865, 639705, 638973, 806597, 654960, 608955, 622547, 620863, 685569, 771536, 743755, 737026, 609103, 747493, 626030, 637638, 634907, 607760, 732302, 759353, 716692, 673928, 632420, 618784, 698316, 710464, 703669, 739287, 688018, 807612, 792815, 744140, 779950, 779568, 793203, 671731, 801958, 742210, 778087, 791726, 796896, 618861, 614552, 653878, 777340, 632397, 717174, 771409, 624579, 637945, 762977, 746162, 785157, 610376, 784185, 664762, 740234, 725259, 742118, 681009, 640520, 644957, 651406, 803347, 778692, 739104, 729330, 743620, 668274, 644172, 796500, 692860, 770828, 701792, 615368, 675921, 760209, 724672, 749581, 627022, 643005, 654958, 667535, 709715, 710951, 666734, 623190, 731932, 766921, 750788, 612253, 709910, 800189, 643231, 794810, 780601, 744483, 738224, 764448, 633908, 804641, 795240, 643947, 608885, 619798, 734328, 612499, 618399, 673175, 732023, 725992, 611535, 655998, 647690, 780594, 734636, 725556, 726715, 741532, 744928, 667513, 715970, 681967, 682124, 774100, 684556, 669892, 762396, 663706, 650590, 618249, 616644, 769151, 753687, 610552, 769020, 617857, 750780, 619294, 732652, 791879, 658158, 747542, 704034, 717641, 778906, 658463, 708475, 634176, 799191, 775427, 659210, 673077, 621109, 656668, 726140, 625354, 727763, 690959, 645640, 805772, 758100, 606959, 771674, 632103, 659980, 687533, 644554, 724981, 763362, 608111, 696579, 669231, 706513, 680936, 763502, 618481, 607199, 714736, 644268, 764023, 634844, 704844, 734036, 610628, 632414, 766839, 740538, 668073, 664071, 639664, 667843, 616385, 740453, 665024, 758134, 638367, 707641, 662256, 637738, 639426, 688432, 757469, 773188, 791673, 687454, 739394, 752184, 757076, 707346, 625183, 793193, 688496, 669448, 760828, 653903, 713827, 753825, 634219, 622166, 644233, 620884, 724789, 764385, 661141, 664570, 744927, 756071, 661211, 763553, 665177, 641059, 807191, 742877, 723954, 642242, 674141, 659510, 741903, 744894, 633560, 650259, 617026, 756220, 791017, 690652, 657656, 758122, 622194, 614376, 699040, 707995, 666660, 750168, 767738, 669544, 634073, 727307, 743250, 667763, 615380, 650293, 794164, 675451, 732166, 617989, 769919, 667512, 781781, 725154, 768917, 739935, 646702, 766996, 782938, 690637, 704248, 739537, 797602, 611055, 665180, 744487, 610445, 733736, 717133, 616828, 636173, 725605, 670269, 673623, 610059, 767599, 752101, 673544, 629355, 621663, 726533, 622355, 611135, 730066, 679293, 744567, 760756, 668446, 657181, 807922, 804286, 616450, 712918, 663184, 637073, 631307, 611479, 768781, 646580, 739439, 770841, 628031, 665625, 619526, 651898, 625786, 763385, 647051, 793310, 619001, 640277, 759303, 796039, 624542, 696813, 790551, 778731, 794422, 715994, 617500, 807414, 723277, 650541, 747489, 620538, 675564, 682533, 709849, 667772, 657411, 758197, 668418, 785731, 638769, 719508, 799462, 758949, 751913, 689617, 794390, 753753, 701448, 698643, 692689, 786001, 689568, 611821, 623436, 689169, 771387, 627199, 655888, 693480, 693588, 753867, 628951, 728647, 790358, 723268, 778732, 713493, 640812, 654821, 653946, 718387, 669159, 720380, 660489, 798553, 736108, 656670, 754324, 694361, 655771, 644967, 702484, 607995, 772308, 783850, 625242, 675198, 728502, 646147, 729216, 684684, 624146, 777807, 757267, 642395, 756699, 633956, 704265, 707758, 623972, 744721, 797849, 627743, 634465, 748924, 758487, 646352, 690848, 721578, 791467, 675504, 690111, 649356, 801352, 676182, 807566, 628745, 767878, 682044, 805015, 751889, 774467, 610258, 673563, 802392, 661357, 747492, 646847, 760915, 662692, 645217, 745969, 691827, 769764, 707958, 628528, 629612, 686555, 790516, 742185, 690566, 670281, 690006, 694444, 638991, 729299, 641251, 674344, 673476, 670375, 730219, 754724, 615746, 768093, 680827, 747577, 636460, 720283, 782545, 616903, 769937, 629399, 630073, 673733, 608262, 777209, 687705, 726632, 623836, 689036, 738970, 777891, 673691, 650663, 617873, 797097, 671815, 659996, 801077, 665330, 797399, 635194, 714910, 609757, 607316, 690607, 778424, 624194, 746655, 732451, 637518, 701791, 780403, 611275, 794969, 768038, 803105, 613710, 690891, 778921, 627947, 757471, 723226, 654028, 617336, 689011, 678828, 646686, 672000, 791100, 635083, 715839, 683834, 663574, 610581, 641272, 727499, 743661, 804823, 634035, 778868, 618899, 687679, 610527, 792573, 726682, 716713, 616883, 796197, 633258, 720450, 734074, 659155, 656937, 621762, 752778, 776283])
        self.labels['y'] = self.labels['y'].map(lambda x: 1 if x else 0).astype(np.int32)
        self.perm = np.random.permutation(len(self.labels))
        #ipdb.set_trace()
        self.labels = self.labels.iloc[self.perm]
        self.n = len(self.labels)
        #self.i = 0

        self.n_batch = int(np.ceil(self.n / float(self.batch_size)))

    def __len__(self):
        return self.n_batch

    def __getitem__(self, i):
        #if i >= self.n_batch: raise IndexError("End")
        if i >= self.n_batch: i = i % self.n_batch
        curr_batch = []
        labels = np.zeros([self.batch_size, 2])
        for j in range(self.batch_size):
            iloc_j = (j + self.batch_size * i)%self.n
            idx = self.labels.index[iloc_j]
            curr_img = np.load(os.path.join(self.map_component_dir, "%d.npy"%idx))
            if self.normalize:
                _min, _max = curr_img.min(), curr_img.max()
                curr_img = (curr_img - _min) / (_max - _min)
            curr_batch.append(curr_img)
            labels[j, 1 if self.labels.iloc[iloc_j]['y'] else 0] = 1
        imgs = np.stack(curr_batch, axis=0).astype(np.float32).swapaxes(3,2).swapaxes(2,1)
        return torch.tensor(imgs), torch.tensor(labels), i



def _get_Fbeta(y, yhat, beta=1., debug=False, ratio=None):
    if ratio is None: ratio = 1.
    TP = ((y == 1) & (yhat == 1)).sum()
    FP = ((y == 0) & (yhat == 1)).sum()
    TN = ((y == 0) & (yhat == 0)).sum()
    FN = ((y == 1) & (yhat == 0)).sum()
    if FP+TP == 0 or TP + FN==0 or TP == 0: return -1.
    precision = (TP) / (FP * ratio + TP).astype(float)
    recall = (TP) / (TP + FN).astype(float)
    if debug:
        print("TP={}; FP={}; TN={}; FN={}; precision={};recall={}".format(((y == 1) & (yhat == 1)).sum(),
                                                                          ((y == 0) & (yhat == 1)).sum(),
                                                                          ((y == 0) & (yhat == 0)).sum(),
                                                                          ((y == 1) & (yhat == 0)).sum(), precision,
                                                                          recall))
        print(precision, recall, (1 + beta ** 2))
    return (1 + beta ** 2) * (precision * recall) / (beta * precision + recall)


def get_F1(y_pred, y, xlim=None, ratio=None):
    if xlim is None:
        xlim = (0, 0.997)
    Fscore = lambda x: _get_Fbeta(y, (y_pred > x).astype(int), ratio=ratio)

    x = np.linspace(xlim[0], xlim[1])
    y = np.asarray([Fscore(xx) for xx in x])
    return x[np.argmax(y)], np.max(y)

import glob
import ipdb
def find_best_results(mode='test',
                      dir_path = '/media/zhen/Research/deepsz_pytorch/%s/results/',
                      dir_name = 'nooversample_ratio1-20_convbody=R-50-C4_lr=0.005_wd=0.002_steps=1000-2500'):
    dir_path = dir_path%dir_name
    fs = sorted(glob.glob(os.path.join(dir_path,'epoch*.pkl')), key=lambda x: int(x.replace(".pkl","").split("epoch")[1]))
    df = pd.DataFrame(columns=['acc','loss', 'F1', "F1_thres"])
    for f in fs:
        res = pd.read_pickle(f)
        if len(res[mode]) == 0: continue
        epoch = int(os.path.basename(f).replace(".pkl","").split("epoch")[1])
        df.loc[epoch, 'acc'] =res[mode]['acc'] if mode == 'test' else res[mode]['acc'][max(res[mode]['acc'].keys())]
        df.loc[epoch, 'loss'] = res[mode]['loss'] if mode == 'test' else res[mode]['loss'][max(res[mode]['loss'].keys())]
        if mode =='test':
            df.loc[epoch, 'F1_thres'],df.loc[epoch, 'F1'] = get_F1(res[mode]['y_pred'], res[mode]['y'])
        else:
            ipdb.set_trace()
            df.loc[epoch, 'F1'] = res[mode]['F1'][max(res[mode]['F1'].keys())]
    return df

