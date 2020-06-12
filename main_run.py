import os
import torch
import glob
from torch import optim
import numpy as np
import time
import argparse
from load_data import NUM_WRITERS
from network_tro import ConTranModel
from load_data import loadData as load_data_func
from loss_tro import CER


parser = argparse.ArgumentParser(description='seq2seq net', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('start_epoch', type=int, help='load saved weights from which epoch')
args = parser.parse_args()

gpu = torch.device('cuda')

OOV = True

NUM_THREAD = 2

EARLY_STOP_EPOCH = None
EVAL_EPOCH = 20
MODEL_SAVE_EPOCH = 200
show_iter_num = 500
LABEL_SMOOTH = True
Bi_GRU = True
VISUALIZE_TRAIN = True

BATCH_SIZE = 8
lr_dis = 1 * 1e-4
lr_gen = 1 * 1e-4
lr_rec = 1 * 1e-5
lr_rec_solo = 1 * 1e-5
lr_cla = 1 * 1e-5

CurriculumModelID = args.start_epoch


def all_data_loader():
    data_train, data_test = load_data_func(OOV)
    train_loader = torch.utils.data.DataLoader(data_train, collate_fn=sort_batch, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_THREAD, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(data_test, collate_fn=sort_batch, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_THREAD, pin_memory=True)
    return train_loader, test_loader


def sort_batch(batch):
    train_domain = list()
    train_wid_sty = list()
    train_idx_sty = list()
    train_img_sty = list()
    train_img_width_sty = list()
    train_label_sty = list()

    train_wid_con = list()
    train_idx_con = list()
    train_img_con = list()
    train_img_width_con = list()
    train_label_con = list()

    for domain, wid, idx, img, img_width, label, widc, idxc, imgc, img_widthc, labelc  in batch:
        if wid >= NUM_WRITERS or widc >= NUM_WRITERS:
            print('error!')
        train_domain.append(domain)
        train_wid_sty.append(wid)
        train_idx_sty.append(idx)
        train_img_sty.append(img)
        train_img_width_sty.append(img_width)
        train_label_sty.append(label)
        train_wid_con.append(widc)
        train_idx_con.append(idxc)
        train_img_con.append(imgc)
        train_img_width_con.append(img_widthc)
        train_label_con.append(labelc)

    train_domain = np.array(train_domain)
    train_idx_sty = np.array(train_idx_sty)
    train_wid_sty = np.array(train_wid_sty, dtype='int64')
    train_img_sty = np.array(train_img_sty, dtype='float32')
    train_img_width_sty = np.array(train_img_width_sty, dtype='int64')
    train_label_sty = np.array(train_label_sty, dtype='int64')
    train_idx_con = np.array(train_idx_con)
    train_wid_con = np.array(train_wid_con, dtype='int64')
    train_img_con = np.array(train_img_con, dtype='float32')
    train_img_width_con = np.array(train_img_width_con, dtype='int64')
    train_label_con = np.array(train_label_con, dtype='int64')

    train_wid_sty = torch.from_numpy(train_wid_sty)
    train_img_sty = torch.from_numpy(train_img_sty)
    train_img_width_sty = torch.from_numpy(train_img_width_sty)
    train_label_sty = torch.from_numpy(train_label_sty)
    train_wid_con = torch.from_numpy(train_wid_con)
    train_img_con = torch.from_numpy(train_img_con)
    train_img_width_con = torch.from_numpy(train_img_width_con)
    train_label_con = torch.from_numpy(train_label_con)

    return train_domain, train_wid_sty, train_idx_sty, train_img_sty, train_img_width_sty, train_label_sty, train_wid_con, train_idx_con, train_img_con, train_img_width_con, train_label_con


def train(train_loader, model, dis_opt, gen_opt, rec_opt, rec_solo_opt, cla_opt, epoch):
    model.train()
    loss_dis = list()
    loss_dis_tr = list()
    loss_cla = list()
    loss_cla_tr = list()
    loss_rec = list()
    loss_rec_tr = list()
    loss_rec_solo = list()
    time_s = time.time()
    cer_te0 = CER()
    cer_te1 = CER()
    cer_te2 = CER()
    cer_te3 = CER()
    cer_solo0 = CER()
    cer_solo1 = CER()
    cerd0 = CER()
    cerd1 = CER()
    for train_data_list in train_loader:
        '''rec update'''
        rec_solo_opt.zero_grad()
        l_rec_solo = model(train_data_list, epoch, 'rec_solo_update', [cer_solo0, cer_solo1])
        rec_solo_opt.step()

        '''classifier update'''
        cla_opt.zero_grad()
        l_cla_tr = model(train_data_list, epoch, 'cla_update')
        cla_opt.step()

        '''dis update'''
        dis_opt.zero_grad()
        l_dis_tr = model(train_data_list, epoch, 'dis_update')
        dis_opt.step()

        '''gen update'''
        gen_opt.zero_grad()
        l_total, l_dis, l_cla, l_rec = model(train_data_list, epoch, 'gen_update', [cer_te0, cer_te1, cer_te2, cer_te3])
        gen_opt.step()

        loss_dis.append(l_dis.cpu().item())
        loss_dis_tr.append(l_dis_tr.cpu().item())
        loss_cla.append(l_cla.cpu().item())
        loss_cla_tr.append(l_cla_tr.cpu().item())
        loss_rec.append(l_rec.cpu().item())
        loss_rec_solo.append(l_rec_solo.cpu().item())

        '''disentangled recognizer update'''
        rec_opt.zero_grad()
        l_rec_tr = model(train_data_list, epoch, 'rec_disentangle_update', [cerd0, cerd1])
        loss_rec_tr.append(l_rec_tr.cpu().item())
        rec_opt.step()

    fl_dis = np.mean(loss_dis)
    fl_dis_tr = np.mean(loss_dis_tr)
    fl_cla = np.mean(loss_cla)
    fl_cla_tr = np.mean(loss_cla_tr)
    fl_rec = np.mean(loss_rec)
    fl_rec_tr = np.mean(loss_rec_tr)
    fl_rec_solo = np.mean(loss_rec_solo)

    res_cer_te0 = cer_te0.fin()
    res_cer_te1 = cer_te1.fin()
    res_cer_te2 = cer_te2.fin()
    res_cer_te3 = cer_te3.fin()
    res_cer_d0 = cerd0.fin()
    res_cer_d1 = cerd1.fin()
    res_cer_solo0 = cer_solo0.fin()
    res_cer_solo1 = cer_solo1.fin()
    print('epo%d <tr>-<gen>: l_dis=%.2f-%.2f, l_cla=%.2f-%.2f, l_rec=%.2f-%.2f-%.2f, cer=%.2f-%.2f-%.2f-%.2f, cer_dist=%.2f--%.2f, cer_solo=%.2f--%.2f, time=%.1f' % (epoch, fl_dis_tr, fl_dis, fl_cla_tr, fl_cla, fl_rec_tr, fl_rec_solo, fl_rec, res_cer_te0, res_cer_te1, res_cer_te2, res_cer_te3, res_cer_d0, res_cer_d1, res_cer_solo0, res_cer_solo1, time.time()-time_s))
    return res_cer_d0 + res_cer_d1

def test(test_loader, epoch, modelFile_o_model):
    if type(modelFile_o_model) == str:
        model = ConTranModel(NUM_WRITERS, show_iter_num, OOV).to(gpu)
        print('Loading ' + modelFile_o_model)
        model.load_state_dict(torch.load(modelFile_o_model)) #load
    else:
        model = modelFile_o_model
    model.eval()
    loss_dis = list()
    loss_cla = list()
    loss_rec = list()
    time_s = time.time()
    cer_te0 = CER()
    cer_te1 = CER()
    cer_te2 = CER()
    cer_te3 = CER()
    cerd0 = CER()
    cerd1 = CER()
    for test_data_list in test_loader:
        l_dis, l_cla, l_rec = model(test_data_list, epoch, 'eval', [cer_te0, cer_te1, cer_te2, cer_te3])

        loss_dis.append(l_dis.cpu().item())
        loss_cla.append(l_cla.cpu().item())
        loss_rec.append(l_rec.cpu().item())

    fl_dis = np.mean(loss_dis)
    fl_cla = np.mean(loss_cla)
    fl_rec = np.mean(loss_rec)

    with torch.no_grad():
        l_rec_dist = model(test_data_list, epoch, 'rec_disentangle_eval', [cerd0, cerd1])
    res_cer_te0 = cer_te0.fin()
    res_cer_te1 = cer_te1.fin()
    res_cer_te2 = cer_te2.fin()
    res_cer_te3 = cer_te3.fin()
    res_cerd0 = cerd0.fin()
    res_cerd1 = cerd1.fin()
    print('EVAL: l_dis=%.3f, l_cla=%.3f, l_rec=%.3f, cer=%.2f-%.2f-%.2f-%.2f, cer_dist=%.2f-%.2f, time=%.1f' % (fl_dis, fl_cla, fl_rec, res_cer_te0, res_cer_te1, res_cer_te2, res_cer_te3, res_cerd0, res_cerd1, time.time()-time_s))
    return res_cerd0 + res_cerd1

def main(train_loader, test_loader, num_writers):
    model = ConTranModel(num_writers, show_iter_num, OOV).to(gpu)

    if CurriculumModelID > 0:
        model_file = 'save_weights/contran-' + str(CurriculumModelID) +'.model'
        print('Loading ' + model_file)
        model.load_state_dict(torch.load(model_file)) #load

    dis_params = list(model.dis.parameters())
    gen_params = list(model.gen.parameters())
    rec_params = list(model.rec.parameters())
    rec_solo_params = list(model.rec_solo.parameters())
    cla_params = list(model.cla.parameters())
    dis_opt = optim.Adam([p for p in dis_params if p.requires_grad], lr=lr_dis)
    gen_opt = optim.Adam([p for p in gen_params if p.requires_grad], lr=lr_gen)
    rec_opt = optim.Adam([p for p in rec_params if p.requires_grad], lr=lr_rec)
    rec_solo_opt = optim.Adam([p for p in rec_solo_params if p.requires_grad], lr=lr_rec_solo)
    cla_opt = optim.Adam([p for p in cla_params if p.requires_grad], lr=lr_cla)
    epochs = 50001
    min_cer = 1e5
    min_idx = 0
    min_count = 0

    for epoch in range(CurriculumModelID, epochs):
        cer = train(train_loader, model, dis_opt, gen_opt, rec_opt, rec_solo_opt, cla_opt, epoch)

        if epoch % MODEL_SAVE_EPOCH == 0:
            folder_weights = 'save_weights'
            if not os.path.exists(folder_weights):
                os.makedirs(folder_weights)
            torch.save(model.state_dict(), folder_weights+'/contran-%d.model'%epoch)

        if epoch % EVAL_EPOCH == 0:
            cer_t = test(test_loader, epoch, model)

        if EARLY_STOP_EPOCH is not None:
            if min_cer > cer_t:
                min_cer = cer_t
                min_idx = epoch
                min_count = 0
                rm_old_model(min_idx)
            else:
                min_count += 1
            if min_count >= EARLY_STOP_EPOCH:
                print('Early stop at %d and the best epoch is %d' % (epoch, min_idx))
                model_url = 'save_weights/contran-'+str(min_idx)+'.model'
                os.system('mv '+model_url+' '+model_url+'.bak')
                os.system('rm save_weights/contran-*.model')
                break

def rm_old_model(index):
    models = glob.glob('save_weights/*.model')
    for m in models:
        epoch = int(m.split('.')[0].split('-')[1])
        if epoch < index:
            os.system('rm save_weights/contran-'+str(epoch)+'.model')

if __name__ == '__main__':
    print(time.ctime())
    train_loader, test_loader = all_data_loader()
    main(train_loader, test_loader, NUM_WRITERS)
    print(time.ctime())
