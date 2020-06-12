import torch
import torch.nn as nn
from load_data import vocab_size, IMG_WIDTH, OUTPUT_MAX_LEN
from modules_tro import GenModel_FC, DisModel, WriterClaModel, RecModel, write_image
from loss_tro import crit, log_softmax
import numpy as np


w_dis = 1.
w_cla = 1.
w_rec = 1.

gpu = torch.device('cuda')


# an example to do the trade-off between gamma1 and gamma2
def obtain_gamma(epoch):
    gamma_1 = 1
    if epoch < 60:
        gamma_2 = 1
    else:
        gamma_2 = 0
    return gamma_1, gamma_2


class ConTranModel(nn.Module):
    def __init__(self, num_writers, show_iter_num, oov):
        super(ConTranModel, self).__init__()
        self.gen = GenModel_FC(OUTPUT_MAX_LEN).to(gpu)
        self.cla = WriterClaModel(num_writers).to(gpu)
        self.dis = DisModel().to(gpu)
        self.rec = self.gen.rec
        self.rec_solo = RecModel(pretrain=False).to(gpu)
        self.iter_num = 0
        self.show_iter_num = show_iter_num
        self.oov = oov

    def forward(self, train_data_list, epoch, mode, cer_func=None):
        tr_domain, tr_wid, tr_idx, tr_img, tr_img_width, tr_label, tr_widc, tr_idxc, tr_imgc, tr_img_widthc, tr_labelc = train_data_list
        tr_wid = tr_wid.to(gpu)
        tr_img = tr_img.to(gpu)
        tr_img_width = tr_img_width.to(gpu)
        tr_label = tr_label[:,:,1:].squeeze(1).to(gpu)
        tr_widc = tr_widc.to(gpu)
        tr_imgc = tr_imgc.to(gpu)
        tr_img_widthc = tr_img_widthc.to(gpu)
        tr_labelc = tr_labelc[:,:,1:].squeeze(1).to(gpu)
        batch_size = tr_domain.shape[0]
        gamma_1, gamma_2 = obtain_gamma(epoch)

        if mode == 'rec_disentangle_update':
            f_rec, f_xt = self.gen.enc_text(tr_img, None, mode='con_enc') # b,4096  b,512,8,27
            f_recc, f_xtc = self.gen.enc_text(tr_imgc, None, mode='con_enc') # b,4096  b,512,8,27
            pred_xt_tr = self.rec(f_rec, tr_label, img_width=torch.from_numpy(np.array([IMG_WIDTH]*batch_size)), mode='feature')
            pred_xt_trc = self.rec(f_recc, tr_labelc, img_width=torch.from_numpy(np.array([IMG_WIDTH]*batch_size)), mode='feature')
            l_rec_tr = crit(log_softmax(pred_xt_tr.reshape(-1,vocab_size)), tr_label.reshape(-1))
            l_rec_trc = crit(log_softmax(pred_xt_trc.reshape(-1,vocab_size)), tr_labelc.reshape(-1))
            l_rec = gamma_1 * (l_rec_tr + l_rec_trc)/2.

            cer0, cer1 = cer_func
            cer0.add(pred_xt_tr, tr_label)
            cer1.add(pred_xt_trc, tr_labelc)
            l_rec.backward()
            return l_rec

        elif mode == 'rec_disentangle_eval':
            with torch.no_grad():
                f_rec, f_xt = self.gen.enc_text(tr_img, None, mode='con_enc') # b,4096  b,512,8,27
                f_recc, f_xtc = self.gen.enc_text(tr_imgc, None, mode='con_enc') # b,4096  b,512,8,27
                pred_xt_tr = self.rec(f_rec, tr_label, img_width=torch.from_numpy(np.array([IMG_WIDTH]*batch_size)), mode='feature')
                pred_xt_trc = self.rec(f_recc, tr_labelc, img_width=torch.from_numpy(np.array([IMG_WIDTH]*batch_size)), mode='feature')
            l_rec_tr = crit(log_softmax(pred_xt_tr.reshape(-1,vocab_size)), tr_label.reshape(-1))
            l_rec_trc = crit(log_softmax(pred_xt_trc.reshape(-1,vocab_size)), tr_labelc.reshape(-1))
            l_rec = (l_rec_tr + l_rec_trc)/2.

            cer0, cer1 = cer_func
            cer0.add(pred_xt_tr, tr_label)
            cer1.add(pred_xt_trc, tr_labelc)
            return l_rec

        elif mode == 'rec_solo_update':
            cer0, cer1 = cer_func
            pred0 = self.rec_solo(tr_img, tr_label, img_width=torch.from_numpy(np.array([IMG_WIDTH]*batch_size)), mode='image')
            pred1 = self.rec_solo(tr_imgc, tr_labelc, img_width=torch.from_numpy(np.array([IMG_WIDTH]*batch_size)), mode='image')
            l_rec0 = crit(log_softmax(pred0.reshape(-1,vocab_size)), tr_label.reshape(-1))
            l_rec1 = crit(log_softmax(pred1.reshape(-1,vocab_size)), tr_labelc.reshape(-1))
            cer0.add(pred0, tr_label)
            cer1.add(pred1, tr_labelc)
            l_rec_solo = (l_rec0 + l_rec1)/2.
            l_rec_solo.backward()
            return l_rec_solo

        elif mode =='cla_update':
            tr_img_rec0 = tr_img[:, 0:1, :, :] # 8,50,64,200 choose one channel 8,1,64,200
            tr_img_rec0 = tr_img_rec0.requires_grad_()

            tr_img_rec1 = tr_imgc
            tr_img_rec1 = tr_img_rec1.requires_grad_()
            l_cla_tr0 = self.cla(tr_img_rec0, tr_wid)
            l_cla_tr1 = self.cla(tr_img_rec1, tr_widc)
            l_cla_tr = (l_cla_tr0 + l_cla_tr1)/2.
            l_cla_tr.backward()
            return l_cla_tr

        elif mode == 'gen_update':
            self.iter_num += 1
            '''dis loss'''
            f_xs = self.gen.enc_image(tr_img) # b,512,8,27
            f_rec, f_xt = self.gen.enc_text(tr_img, None, mode='con_enc') # b,4096  b,512,8,27
            f_xsc = self.gen.enc_image(tr_imgc) # b,512,8,27
            f_recc, f_xtc = self.gen.enc_text(tr_imgc, None, mode='con_enc') # b,4096  b,512,8,27

            xg0 = self.gen.decode(f_xs, f_xt)  # translation b,1,64,128
            xg1 = self.gen.decode(f_xs, f_xtc)  # translation b,1,64,128
            xg2 = self.gen.decode(f_xsc, f_xt)  # translation b,1,64,128
            xg3 = self.gen.decode(f_xsc, f_xtc)  # translation b,1,64,128

            l_dis0 = self.dis.calc_gen_loss(xg0)
            l_dis1 = self.dis.calc_gen_loss(xg1)
            l_dis2 = self.dis.calc_gen_loss(xg2)
            l_dis3 = self.dis.calc_gen_loss(xg3)
            l_dis = (l_dis0 + l_dis1 + l_dis2 + l_dis3)/4.

            '''writer classifier loss'''
            l_cla0 = self.cla(xg0, tr_wid)
            l_cla1 = self.cla(xg1, tr_wid)
            l_cla2 = self.cla(xg2, tr_widc)
            l_cla3 = self.cla(xg3, tr_widc)

            l_cla = (l_cla0 + l_cla1 + l_cla2 + l_cla3)/4.

            '''rec loss'''
            cer_te0, cer_te1, cer_te2, cer_te3 = cer_func
            pred_xt0 = self.rec_solo(xg0, tr_label, img_width=torch.from_numpy(np.array([IMG_WIDTH]*batch_size)), mode='image')
            pred_xt1 = self.rec_solo(xg1, tr_labelc, img_width=torch.from_numpy(np.array([IMG_WIDTH]*batch_size)), mode='image')
            pred_xt2 = self.rec_solo(xg2, tr_label, img_width=torch.from_numpy(np.array([IMG_WIDTH]*batch_size)), mode='image')
            pred_xt3 = self.rec_solo(xg3, tr_labelc, img_width=torch.from_numpy(np.array([IMG_WIDTH]*batch_size)), mode='image')
            l_rec0 = crit(log_softmax(pred_xt0.reshape(-1,vocab_size)), tr_label.reshape(-1))
            l_rec1 = crit(log_softmax(pred_xt1.reshape(-1,vocab_size)), tr_labelc.reshape(-1))
            l_rec2 = crit(log_softmax(pred_xt2.reshape(-1,vocab_size)), tr_label.reshape(-1))
            l_rec3 = crit(log_softmax(pred_xt3.reshape(-1,vocab_size)), tr_labelc.reshape(-1))
            cer_te0.add(pred_xt0, tr_label)
            cer_te1.add(pred_xt1, tr_labelc)
            cer_te2.add(pred_xt2, tr_label)
            cer_te3.add(pred_xt3, tr_labelc)
            l_rec = (l_rec0 + l_rec1 + l_rec2 + l_rec3)/4.

            '''fin'''
            l_total = w_dis * l_dis + w_cla * l_cla + w_rec * l_rec
            l_total = gamma_2 * l_total
            l_total.backward()
            return l_total, l_dis, l_cla, l_rec

        elif mode == 'dis_update':
            sample_img1 = tr_img[:,0:1,:,:]
            sample_img1.requires_grad_()
            sample_img2 = tr_imgc
            sample_img2.requires_grad_()

            l_real0 = self.dis.calc_dis_real_loss(sample_img1)
            l_real1 = self.dis.calc_dis_real_loss(sample_img2)
            l_real = (l_real0 + l_real1)/2.
            l_real.backward(retain_graph=True)

            with torch.no_grad():
                f_xs = self.gen.enc_image(tr_img) # b,512,8,27
                f_rec, f_xt = self.gen.enc_text(tr_img, None, mode='con_enc') # b,4096  b,512,8,27
                f_xsc = self.gen.enc_image(tr_imgc) # b,512,8,27
                f_recc, f_xtc = self.gen.enc_text(tr_imgc, None, mode='con_enc') # b,4096  b,512,8,27

                xg0 = self.gen.decode(f_xs, f_xt)
                xg1 = self.gen.decode(f_xs, f_xtc)
                xg2 = self.gen.decode(f_xsc, f_xt)
                xg3 = self.gen.decode(f_xsc, f_xtc)  # translation b,1,64,128

            l_fake0 = self.dis.calc_dis_fake_loss(xg0)
            l_fake1 = self.dis.calc_dis_fake_loss(xg1)
            l_fake2 = self.dis.calc_dis_fake_loss(xg2)
            l_fake3 = self.dis.calc_dis_fake_loss(xg3)
            l_fake = (l_fake0 + l_fake1 + l_fake2 + l_fake3)/4.
            l_fake.backward()

            l_total = l_real + l_fake
            '''write images'''
            if self.iter_num % self.show_iter_num == 0:
                with torch.no_grad():
                    pred0 = self.rec_solo(xg0, tr_label, img_width=torch.from_numpy(np.array([IMG_WIDTH]*batch_size)), mode='image')
                    pred1 = self.rec_solo(xg1, tr_labelc, img_width=torch.from_numpy(np.array([IMG_WIDTH]*batch_size)), mode='image')
                    pred2 = self.rec_solo(xg2, tr_label, img_width=torch.from_numpy(np.array([IMG_WIDTH]*batch_size)), mode='image')
                    pred3 = self.rec_solo(xg3, tr_labelc, img_width=torch.from_numpy(np.array([IMG_WIDTH]*batch_size)), mode='image')
                write_image(tr_img, tr_imgc, [xg0, xg1, xg2, xg3], [pred0, pred1, pred2, pred3], [tr_label, tr_labelc, tr_label, tr_labelc], 'epoch_'+str(epoch)+'-'+str(self.iter_num))
            return l_total

        elif mode =='eval':
            with torch.no_grad():
                f_xs = self.gen.enc_image(tr_img) # b,512,8,27
                f_rec, f_xt = self.gen.enc_text(tr_img, None, mode='con_enc') # b,4096  b,512,8,27
                f_xsc = self.gen.enc_image(tr_imgc) # b,512,8,27
                f_recc, f_xtc = self.gen.enc_text(tr_imgc, None, mode='con_enc') # b,4096  b,512,8,27

                xg0 = self.gen.decode(f_xs, f_xt)
                xg1 = self.gen.decode(f_xs, f_xtc)
                xg2 = self.gen.decode(f_xsc, f_xt)
                xg3 = self.gen.decode(f_xsc, f_xtc)  # translation b,1,64,128
                '''write images'''
                pred0 = self.rec_solo(xg0, tr_label, img_width=torch.from_numpy(np.array([IMG_WIDTH]*batch_size)), mode='image')
                pred1 = self.rec_solo(xg1, tr_labelc, img_width=torch.from_numpy(np.array([IMG_WIDTH]*batch_size)), mode='image')
                pred2 = self.rec_solo(xg2, tr_label, img_width=torch.from_numpy(np.array([IMG_WIDTH]*batch_size)), mode='image')
                pred3 = self.rec_solo(xg3, tr_labelc, img_width=torch.from_numpy(np.array([IMG_WIDTH]*batch_size)), mode='image')
                write_image(tr_img, tr_imgc, [xg0, xg1, xg2, xg3], [pred0, pred1, pred2, pred3], [tr_label, tr_labelc, tr_label, tr_labelc], 'eval_'+str(epoch)+'-'+str(self.iter_num))
                self.iter_num += 1
                '''dis loss'''
                l_dis0 = self.dis.calc_gen_loss(xg0)
                l_dis1 = self.dis.calc_gen_loss(xg1)
                l_dis2 = self.dis.calc_gen_loss(xg2)
                l_dis3 = self.dis.calc_gen_loss(xg3)
                l_dis = (l_dis0 + l_dis1 + l_dis2 + l_dis3)/4.

                '''rec loss'''
                cer_te0, cer_te1, cer_te2, cer_te3 = cer_func
                pred_xt0 = self.rec_solo(xg0, tr_label, img_width=torch.from_numpy(np.array([IMG_WIDTH]*batch_size)), mode='image')
                pred_xt1 = self.rec_solo(xg1, tr_labelc, img_width=torch.from_numpy(np.array([IMG_WIDTH]*batch_size)), mode='image')
                pred_xt2 = self.rec_solo(xg2, tr_label, img_width=torch.from_numpy(np.array([IMG_WIDTH]*batch_size)), mode='image')
                pred_xt3 = self.rec_solo(xg3, tr_labelc, img_width=torch.from_numpy(np.array([IMG_WIDTH]*batch_size)), mode='image')
                l_rec0 = crit(log_softmax(pred_xt0.reshape(-1,vocab_size)), tr_label.reshape(-1))
                l_rec1 = crit(log_softmax(pred_xt1.reshape(-1,vocab_size)), tr_labelc.reshape(-1))
                l_rec2 = crit(log_softmax(pred_xt2.reshape(-1,vocab_size)), tr_label.reshape(-1))
                l_rec3 = crit(log_softmax(pred_xt3.reshape(-1,vocab_size)), tr_labelc.reshape(-1))
                cer_te0.add(pred_xt0, tr_label)
                cer_te1.add(pred_xt1, tr_labelc)
                cer_te2.add(pred_xt2, tr_label)
                cer_te3.add(pred_xt3, tr_labelc)
                l_rec = (l_rec0 + l_rec1 + l_rec2 + l_rec3)/4.

                '''writer classifier loss'''
                l_cla0 = self.cla(xg0, tr_wid)
                l_cla1 = self.cla(xg1, tr_wid)
                l_cla2 = self.cla(xg2, tr_widc)
                l_cla3 = self.cla(xg3, tr_widc)
                l_cla = (l_cla0 + l_cla1 + l_cla2 + l_cla3)/4.

            return l_dis, l_cla, l_rec
