import torch.utils.data as D
import random
import string
import cv2
import numpy as np
from pairs_idx_wid_iam import wid2label_tr, wid2label_te


CREATE_PAIRS = False

IMG_HEIGHT = 64
IMG_WIDTH = 216
MAX_CHARS = 7

NUM_CHANNEL = 1

NUM_WRITERS = 500 # iam

NORMAL = True
OUTPUT_MAX_LEN = MAX_CHARS+2 # (<GO>+groundtruth+<END>)
img_base = '/home/lkang/datasets/iam_final_forms/words_from_forms/'

src = '/home/lkang/datasets/iam_final_forms/gan.iam.tr_va.gt.filter27'
tar = '/home/lkang/datasets/iam_final_forms/gan.iam.test.gt.filter27'

def labelDictionary():
    labels = [' ', '!', '"', '#', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';'    , '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '_',     'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    letter2index = {label: n for n, label in enumerate(labels)}
    index2letter = {v: k for k, v in letter2index.items()}
    return len(labels), letter2index, index2letter

num_classes, letter2index, index2letter = labelDictionary()
tokens = {'GO_TOKEN': 0, 'END_TOKEN': 1, 'PAD_TOKEN': 2}
num_tokens = len(tokens.keys())
vocab_size = num_classes + num_tokens


def edits1(word, min_len=2, max_len=MAX_CHARS):
    "All edits that are one edit away from `word`."
    letters = list(string.ascii_lowercase)
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    if len(word) <= min_len:
        return random.choice(list(set(transposes + replaces + inserts)))
    elif len(word) >= max_len:
        return random.choice(list(set(deletes + transposes + replaces)))
    else:
        return random.choice(list(set(deletes + transposes + replaces + inserts)))


class IAM_words(D.Dataset):
    def __init__(self, data_dict, oov):
        self.data_dict = data_dict
        self.oov = oov
        self.output_max_len = OUTPUT_MAX_LEN

    # word [0, 15, 27, 13, 32, 31, 1, 2, 2, 2]
    def new_ed1(self, word_ori):
        word = word_ori.copy()
        start = word.index(tokens['GO_TOKEN'])
        fin = word.index(tokens['END_TOKEN'])
        word = ''.join([index2letter[i-num_tokens] for i in word[start+1: fin]])
        new_word = edits1(word)
        label = np.array(self.label_padding(new_word, num_tokens))
        return label

    def __getitem__(self, wid_idx_num):
        '''###########################'''
        '''style input'''
        words = self.data_dict[wid_idx_num]
        '''shuffle images'''
        np.random.shuffle(words)

        wids = list()
        idxs = list()
        imgs = list()
        img_widths = list()
        labels = list()

        for word in words:
            wid, idx = word[0].split(',')
            img, img_width = self.read_image_single(idx)
            label = self.label_padding(' '.join(word[1:]), num_tokens)
            wids.append(wid)
            idxs.append(idx)
            imgs.append(img)
            img_widths.append(img_width)
            labels.append(label)

        if len(list(set(wids))) != 1:
            print('Error! writer id differs')
            exit()
        final_wid = wid_idx_num

        num_imgs = len(imgs)
        if num_imgs >= NUM_CHANNEL:
            final_img = np.stack(imgs[:NUM_CHANNEL], axis=0) # 64, h, w
            final_idx = idxs[:NUM_CHANNEL]
            final_img_width = img_widths[:NUM_CHANNEL]
            final_label = labels[:NUM_CHANNEL]
        else:
            final_idx = idxs
            final_img = imgs
            final_img_width = img_widths
            final_label = labels

            while len(final_img) < NUM_CHANNEL:
                num_cp = NUM_CHANNEL - len(final_img)
                final_idx = final_idx + idxs[:num_cp]
                final_img = final_img + imgs[:num_cp]
                final_img_width = final_img_width + img_widths[:num_cp]
                final_label = final_label + labels[:num_cp]
            final_img = np.stack(final_img, axis=0)

        '''set new name'''
        final_wid_sty = final_wid
        final_idx_sty = final_idx
        final_img_sty = final_img
        final_img_width_sty = final_img_width
        final_label_sty = final_label

        '''###########################'''
        '''content input'''
        wid_choice = random.choice(list(self.data_dict.keys()))
        words = self.data_dict[wid_choice]
        '''shuffle images'''
        np.random.shuffle(words)

        wids = list()
        idxs = list()
        imgs = list()
        img_widths = list()
        labels = list()

        for word in words:
            wid, idx = word[0].split(',')
            img, img_width = self.read_image_single(idx)
            label = self.label_padding(' '.join(word[1:]), num_tokens)
            wids.append(wid)
            idxs.append(idx)
            imgs.append(img)
            img_widths.append(img_width)
            labels.append(label)

        if len(list(set(wids))) != 1:
            print('Error! writer id differs')
            exit()
        final_wid = wid_choice

        num_imgs = len(imgs)
        if num_imgs >= NUM_CHANNEL:
            final_img = np.stack(imgs[:NUM_CHANNEL], axis=0) # 64, h, w
            final_idx = idxs[:NUM_CHANNEL]
            final_img_width = img_widths[:NUM_CHANNEL]
            final_label = labels[:NUM_CHANNEL]
        else:
            final_idx = idxs
            final_img = imgs
            final_img_width = img_widths
            final_label = labels

            while len(final_img) < NUM_CHANNEL:
                num_cp = NUM_CHANNEL - len(final_img)
                final_idx = final_idx + idxs[:num_cp]
                final_img = final_img + imgs[:num_cp]
                final_img_width = final_img_width + img_widths[:num_cp]
                final_label = final_label + labels[:num_cp]
            final_img = np.stack(final_img, axis=0)

        '''set new name'''
        final_wid_con = final_wid
        final_idx_con = final_idx[0:1]
        final_img_con = final_img[0:1]
        final_img_width_con = final_img_width[0:1]
        final_label_con = final_label[0:1]

        return 'src', final_wid_sty, final_idx_sty, final_img_sty, final_img_width_sty, final_label_sty, final_wid_con, final_idx_con, final_img_con, final_img_width_con, final_label_con

    def __len__(self):
        return len(self.data_dict)

    def read_image_single(self, file_name):
        url = img_base + file_name + '.png'
        img = cv2.imread(url, 0)

        rate = float(IMG_HEIGHT) / img.shape[0]
        img = cv2.resize(img, (int(img.shape[1]*rate)+1, IMG_HEIGHT), interpolation=cv2.INTER_CUBIC) # INTER_AREA con error
        img = img/255. # 0-255 -> 0-1

        img = 1. - img
        img_width = img.shape[-1]

        if img_width > IMG_WIDTH:
            outImg = img[:, :IMG_WIDTH]
            img_width = IMG_WIDTH
        else:
            outImg = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype='float32')
            outImg[:, :img_width] = img
        outImg = outImg.astype('float32')

        mean = 0.5
        std = 0.5
        outImgFinal = (outImg - mean) / std
        return outImgFinal, img_width

    def label_padding(self, labels, num_tokens):
        new_label_len = []
        ll = [letter2index[i] for i in labels]
        new_label_len.append(len(ll)+2)
        ll = np.array(ll) + num_tokens
        ll = list(ll)
        ll = [tokens['GO_TOKEN']] + ll + [tokens['END_TOKEN']]
        num = self.output_max_len - len(ll)
        if not num == 0:
            ll.extend([tokens['PAD_TOKEN']] * num) # replace PAD_TOKEN
        return ll


def loadData(oov):
    gt_tr = src
    gt_te = tar

    with open(gt_tr, 'r') as f_tr:
        data_tr = f_tr.readlines()
        data_tr = [i.strip().split(' ') for i in data_tr]
        tr_dict = dict()
        for i in data_tr:
            wid = i[0].split(',')[0]
            if wid not in tr_dict.keys():
                tr_dict[wid] = [i]
            else:
                tr_dict[wid].append(i)
        new_tr_dict = dict()
        if CREATE_PAIRS:
            create_pairs(tr_dict)
        for k in tr_dict.keys():
            new_tr_dict[wid2label_tr[k]] = tr_dict[k]

    with open(gt_te, 'r') as f_te:
        data_te = f_te.readlines()
        data_te = [i.strip().split(' ') for i in data_te]
        te_dict = dict()
        for i in data_te:
            wid = i[0].split(',')[0]
            if wid not in te_dict.keys():
                te_dict[wid] = [i]
            else:
                te_dict[wid].append(i)
        new_te_dict = dict()
        if CREATE_PAIRS:
            create_pairs(te_dict)
        for k in te_dict.keys():
            new_te_dict[wid2label_te[k]] = te_dict[k]

    data_train = IAM_words(new_tr_dict, oov)
    data_test = IAM_words(new_te_dict, oov)
    return data_train, data_test


def create_pairs(ddict):
    num = len(ddict.keys())
    label2wid = list(zip(range(num), ddict.keys()))
    print(label2wid)


if __name__ == '__main__':
    pass
