import os.path as osp
import torch
import torch.nn.functional as F
from torch.autograd import Variable

import datasets
import settings
from utils import compress, calculate_top_map, calculate_map, p_topK, pr_Curve
from models import ImgNet, TxtNet
import scipy.io as sio
import numpy as np

torch.manual_seed(1)
torch.cuda.manual_seed_all(1)

class Session:
    def __init__(self):
        self.logger = settings.logger

        torch.cuda.set_device(settings.GPU_ID)

        if settings.DATASET == "MIRFlickr":
            self.train_dataset = datasets.MIRFlickr(train=True, transform=datasets.mir_train_transform)
            self.test_dataset = datasets.MIRFlickr(train=False, database=False, transform=datasets.mir_test_transform)
            self.database_dataset = datasets.MIRFlickr(train=False, database=True,
                                                       transform=datasets.mir_test_transform)

        if settings.DATASET == "NUSWIDE":
            self.train_dataset = datasets.NUSWIDE(train=True, transform=datasets.nus_train_transform)
            self.test_dataset = datasets.NUSWIDE(train=False, database=False, transform=datasets.nus_test_transform)
            self.database_dataset = datasets.NUSWIDE(train=False, database=True, transform=datasets.nus_test_transform)

        # Data Loader (Input Pipeline)
        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset,
                                                        batch_size=settings.BATCH_SIZE,
                                                        shuffle=True,
                                                        num_workers=settings.NUM_WORKERS,
                                                        drop_last=True)

        self.test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset,
                                                       batch_size=settings.BATCH_SIZE,
                                                       shuffle=False,
                                                       num_workers=settings.NUM_WORKERS)

        self.database_loader = torch.utils.data.DataLoader(dataset=self.database_dataset,
                                                           batch_size=settings.BATCH_SIZE,
                                                           shuffle=False,
                                                           num_workers=settings.NUM_WORKERS)
        txt_feat_len = datasets.txt_feat_len
        self.CodeNet_I = ImgNet(code_len=settings.CODE_LEN, txt_feat_len=txt_feat_len)
        self.FeatNet_I = ImgNet(code_len=settings.CODE_LEN, txt_feat_len=txt_feat_len)

        self.CodeNet_T = TxtNet(code_len=settings.CODE_LEN, txt_feat_len=txt_feat_len)
        self.maxfunc = torch.nn.ReLU()

        self.opt_I = torch.optim.SGD(self.CodeNet_I.parameters(), lr=settings.LR_IMG, momentum=settings.MOMENTUM,
                                     weight_decay=settings.WEIGHT_DECAY, nesterov=True)

        self.opt_T = torch.optim.SGD(self.CodeNet_T.parameters(), lr=settings.LR_TXT, momentum=settings.MOMENTUM,
                                     weight_decay=settings.WEIGHT_DECAY, nesterov=True)
        self.best = 0
        
    def set_LR(self):
        self.opt_I = torch.optim.SGD(self.CodeNet_I.parameters(), lr=settings.LR_IMG * 0.2, momentum=settings.MOMENTUM,
                                     weight_decay=settings.WEIGHT_DECAY, nesterov=True)

        self.opt_T = torch.optim.SGD(self.CodeNet_T.parameters(), lr=settings.LR_TXT * 0.2, momentum=settings.MOMENTUM,
                                     weight_decay=settings.WEIGHT_DECAY, nesterov=True)

    def train(self, epoch):

        self.CodeNet_I.cuda().train()
        self.CodeNet_T.cuda().train()

        self.CodeNet_I.set_alpha(epoch)
        self.CodeNet_T.set_alpha(epoch)

        self.logger.info('Epoch [%d/%d], alpha for ImgNet: %.4f, alpha for TxtNet: %.4f' % (
            epoch + 1, settings.NUM_EPOCH, self.CodeNet_I.alpha, self.CodeNet_T.alpha))

        for idx, (img, txt, labels, _) in enumerate(self.train_loader):
            img = Variable(img.cuda())
            txt = Variable(torch.FloatTensor(txt.numpy()).cuda())
            labels = Variable(torch.FloatTensor(labels.numpy()).cuda())
            self.opt_I.zero_grad()
            self.opt_T.zero_grad()

            _, _, B_I = self.CodeNet_I(img)
            _, _, B_T = self.CodeNet_T(txt)

            L = F.normalize(labels).mm(F.normalize(labels).t())
            
            k = torch.tensor(settings.CODE_LEN, dtype=torch.float32)
            thresh = (1 - L) * k / 2
            width = 3
            up_thresh =  thresh
            low_thresh =  thresh - width
            low_thresh[low_thresh <= 0] = 0
            low_thresh[L == 0] = settings.CODE_LEN / 2
            
            low_flag = torch.ones(settings.BATCH_SIZE, settings.BATCH_SIZE).cuda()
            up_flag = torch.ones(settings.BATCH_SIZE, settings.BATCH_SIZE).cuda()
            low_flag[L == 1] = 0
            low_flag[L == 0] = settings.BETA
            up_flag[L == 0] = 0
            up_flag[L == 1] = settings.ALPHA

            BI_BI = (settings.CODE_LEN - B_I.mm(B_I.t())) / 2
            BT_BT = (settings.CODE_LEN - B_T.mm(B_T.t())) / 2
            BI_BT = (settings.CODE_LEN - B_I.mm(B_T.t())) / 2
            BT_BI = (settings.CODE_LEN - B_T.mm(B_I.t())) / 2

            # # lower bound
            loss1 = (torch.norm(self.maxfunc(low_thresh - BI_BI) * low_flag) \
                    + torch.norm(self.maxfunc(low_thresh - BT_BT) * low_flag) \
                    + torch.norm(self.maxfunc(low_thresh - BT_BI) * low_flag) \
                    + torch.norm(self.maxfunc(low_thresh - BI_BT) * low_flag)) / (settings.BATCH_SIZE * settings.BATCH_SIZE)
            
            # upper bound
            loss2 = (torch.norm(self.maxfunc(BI_BI - up_thresh) * up_flag) \
                    + torch.norm(self.maxfunc(BT_BT - up_thresh) * up_flag) \
                    + torch.norm(self.maxfunc(BT_BI - up_thresh) * up_flag) \
                    + torch.norm(self.maxfunc(BI_BT - up_thresh) * up_flag)) / (settings.BATCH_SIZE * settings.BATCH_SIZE)
            
            loss = loss1 + loss2
        
            loss.backward()
            self.opt_I.step()
            self.opt_T.step()
            if (idx + 1) % (len(self.train_dataset) // settings.BATCH_SIZE / settings.EPOCH_INTERVAL) == 0:
                self.logger.info(
                    'Epoch [%d/%d], Iter [%d/%d] '
                    'Loss1: %.4f Loss2: %.4f'
                    'Total Loss: %.4f'
                    % (
                        epoch + 1, settings.NUM_EPOCH, idx + 1,
                        len(self.train_dataset) // settings.BATCH_SIZE,
                        loss1.item(), loss2.item(),
                        loss.item()))

    def eval(self, step=0, last=False):

        # Change model to 'eval' mode (BN uses moving mean/var).
        self.CodeNet_I.eval().cuda()
        self.CodeNet_T.eval().cuda()

        
        re_BI, re_BT, re_L, qu_BI, qu_BT, qu_L = compress(self.database_loader, self.test_loader, self.CodeNet_I,
                                                              self.CodeNet_T, self.database_dataset, self.test_dataset)
        K1 = [1, 200, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
            
        if settings.DATASET == "MIRFlickr":
            K2 = [50, 100, 300, 500, 1000] + [i for i in range(2000, 20015, 2000)] #+ [20015]
        if settings.DATASET == "NUSWIDE":
            K2 = [50, 100, 500, 1000, 2000, 5000] + [i for i in range(5000, 186577, 10000)] #+ [186577]
            
       
        if settings.EVAL:
            MAP_I2T1, R_I2T = calculate_map(qu_B=qu_BI, re_B=re_BT, qu_L=qu_L, re_L=re_L)
            MAP_T2I1, R_T2I = calculate_map(qu_B=qu_BT, re_B=re_BI, qu_L=qu_L, re_L=re_L)
            file_name = '%s_%d_order.mat' % (settings.DATASET, settings.CODE_LEN)
            # sio.savemat(file_name, {'Ri2t':R_I2T,
            #                         'Rt2i':R_T2I,
            #                         'Lqu': qu_L,
            #                         'Lre': re_L,
            #                         'BI': re_BI,
            #                         'BT': re_BT})
            self.logger.info('--------------------Evaluation: Calculate top MAP-------------------')
            self.logger.info('MAP of Image to Text: %.4f, MAP of Text to Image: %.4f' % (MAP_I2T1, MAP_T2I1))
            self.logger.info('--------------------------------------------------------------------')
            retI2T = p_topK(qu_BI, re_BT, qu_L, re_L, K1)
            retT2I = p_topK(qu_BT, re_BI, qu_L, re_L, K1)
            self.logger.info(retI2T)
            self.logger.info(retT2I)
            pI2T, rI2T = pr_Curve(qu_BI, re_BT, qu_L, re_L, K2)
            pT2I, rT2I = pr_Curve(qu_BT, re_BI, qu_L, re_L, K2)
            self.logger.info(pI2T)
            self.logger.info(rI2T)
            self.logger.info(pT2I)
            self.logger.info(rT2I)
            MAP_I2T = calculate_top_map(qu_B=qu_BI, re_B=re_BT, qu_L=qu_L, re_L=re_L, topk=50)
            MAP_T2I = calculate_top_map(qu_B=qu_BT, re_B=re_BI, qu_L=qu_L, re_L=re_L, topk=50)
            self.logger.info('--------------------Evaluation: Calculate top MAP-------------------')
            self.logger.info('MAP of Image to Text: %.4f, MAP of Text to Image: %.4f' % (MAP_I2T, MAP_T2I))
            self.logger.info('--------------------------------------------------------------------')
            file_name = '%s_%d_result_%d.mat' % (settings.DATASET, settings.CODE_LEN, step)
            sio.savemat(file_name, {'mapall':[MAP_I2T1, MAP_T2I1],
                                    # 'map50':[MAP_I2T, MAP_T2I],
                                    'topki2t': retI2T.numpy(),
                                    'topkt2i': retT2I.numpy(),
                                    'prpi2t': pI2T.numpy(),
                                    'prri2t': rI2T.numpy(),
                                    'prpt2i': pT2I.numpy(),
                                    'prrt2i': rT2I.numpy()})
            
        else:
            MAP_I2T, _ = calculate_map(qu_B=qu_BI, re_B=re_BT, qu_L=qu_L, re_L=re_L)
            MAP_T2I, _ = calculate_map(qu_B=qu_BT, re_B=re_BI, qu_L=qu_L, re_L=re_L)
            self.logger.info('--------------------Evaluation: Calculate top MAP-------------------')
            self.logger.info('MAP of Image to Text: %.4f, MAP of Text to Image: %.4f' % (MAP_I2T, MAP_T2I))
            self.logger.info('--------------------------------------------------------------------')
            
        
        if MAP_I2T + MAP_T2I > self.best and not settings.EVAL:
            self.save_checkpoints(step=step, best=True)
            self.best = MAP_T2I + MAP_I2T
            self.logger.info("#########is best:%.4f #########" % self.best)

    def save_checkpoints(self, step, file_name='%s_%d_bit_latest.pth' % (settings.DATASET, settings.CODE_LEN),
                         best=False):
        if best:
            file_name = '%s_%d_bit_best_epoch.pth' % (settings.DATASET, settings.CODE_LEN)
        ckp_path = osp.join(settings.MODEL_DIR, file_name)
        obj = {
            'ImgNet': self.CodeNet_I.state_dict(),
            'TxtNet': self.CodeNet_T.state_dict(),
            'step': step,
        }
        torch.save(obj, ckp_path)
        self.logger.info('**********Save the trained model successfully.**********')

    def load_checkpoints(self, file_name='%s_%d_bit_best_epoch.pth' % (settings.DATASET, settings.CODE_LEN)):
        ckp_path = osp.join(settings.MODEL_DIR, file_name)
        try:
            obj = torch.load(ckp_path, map_location=lambda storage, loc: storage.cuda())
            self.logger.info('**************** Load checkpoint %s ****************' % ckp_path)
        except IOError:
            self.logger.error('********** No checkpoint %s!*********' % ckp_path)
            return
        self.CodeNet_I.load_state_dict(obj['ImgNet'])
        self.CodeNet_T.load_state_dict(obj['TxtNet'])
        self.logger.info('********** The loaded model has been trained for %d epochs.*********' % obj['step'])


def main():    
    sess = Session()

    if settings.EVAL == True:
        sess.load_checkpoints()
        sess.eval()

    else:
        sess.logger.info('---------------------------Size------------------------')
        sess.logger.info('Training size: %d, Test size: %d' % (len(datasets.indexTrain), len(datasets.indexTest)))
        
        # settings.EVAL = True
        for epoch in range(settings.NUM_EPOCH):
            sess.train(epoch)
            # eval the Model
            if (epoch + 1) % settings.EVAL_INTERVAL == 0:
                sess.eval(step=epoch + 1)
            # save the model
        settings.EVAL = True
        sess.logger.info('---------------------------Test------------------------')
        sess.load_checkpoints()
        sess.eval()


if __name__ == '__main__':
    main()
