import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.update import BasicUpdateBlock, SmallUpdateBlock, BasicUpdateBlock_laplace
from model.extractor import BasicEncoder, SmallEncoder
from model.corr import CorrBlock, AlternateCorrBlock
from utils.utils import bilinear_sampler, coords_grid, upflow8

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass


class RAFT(nn.Module):
    def __init__(self, args):
        super(RAFT, self).__init__()
        self.args = args

        if args.small:
            self.hidden_dim = hdim = 96
            self.context_dim = cdim = 64
            args.corr_levels = 4
            args.corr_radius = 3
        
        else:
            self.hidden_dim = hdim = 128
            self.context_dim = cdim = 128
            args.corr_levels = 4
            args.corr_radius = 4

        if 'dropout' not in self.args:
            self.args.dropout = 0

        if 'alternate_corr' not in self.args:
            self.args.alternate_corr = False

        # feature network, context network, and update block
        if args.small:
            # add input dim
            self.fnet = SmallEncoder(input_dim=1, output_dim=128, norm_fn='instance', dropout=args.dropout)        
            self.cnet = SmallEncoder(input_dim=1, output_dim=hdim+cdim, norm_fn='none', dropout=args.dropout)
            self.update_block = SmallUpdateBlock(self.args, hidden_dim=hdim)

        else:
            self.fnet = BasicEncoder(input_dim=1, output_dim=256, norm_fn='instance', dropout=args.dropout)        
            self.cnet = BasicEncoder(input_dim=1, output_dim=hdim+cdim, norm_fn='batch', dropout=args.dropout) # hdim + cdim
            self.update_block = BasicUpdateBlock_laplace(self.args, hidden_dim=hdim)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
    # initialize the flow
    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//8, W//8, device=img.device) # output is a tensor of shape [N, 2, H//8, W//8]
        coords1 = coords_grid(N, H//8, W//8, device=img.device) # and this is the same as coords0

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, info, mask):
        # input flow is of shape [N, 2, H//8, W//8]
        # mask is what?
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)
        # laplace
        up_info = F.unfold(info, [3, 3], padding=1)
        up_info = up_info.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        # laplace
        up_info = torch.sum(mask * up_info, dim=2)
        up_info = up_info.permute(0, 1, 4, 2, 5, 3)
        # return up_flow.reshape(N, 2, 8*H, 8*W)
        # laplace
        return up_flow.reshape(N, 2, 8*H, 8*W), up_info.reshape(N, 2, 8*H, 8*W)


    def forward(self, image1, image2, iters=12, flow_init=None, upsample=True, test_mode=False):
        """ Estimate optical flow between pair of frames """
        # normalize the image
        # image1 = 2 * (image1 / 255.0) - 1.0 # TODO change the 255
        # image2 = 2 * (image2 / 255.0) - 1.0

        # contiguous to improve the performance
        image1 = image1.contiguous()
        image2 = image2.contiguous()

        # hyper-parameters
        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        with autocast(enabled=self.args.mixed_precision):
            fmap1, fmap2 = self.fnet([image1, image2])        # get features. both fmap1 and fmap2
        # force to be float32 (default)
        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
        if self.args.alternate_corr: # alternate correlation
            corr_fn = AlternateCorrBlock(fmap1, fmap2, radius=self.args.corr_radius)
        else:
            corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius) # get a corr_fn, based on fmap1 and fmap2

        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            cnet = self.cnet(image1) # run context block in image1. Still from basic encoder
            net, inp = torch.split(cnet, [hdim, cdim], dim=1) # split the channel dimension
            net = torch.tanh(net) # activation, tanh
            inp = torch.relu(inp) # activation, relu

        # 
        coords0, coords1 = self.initialize_flow(image1)

        if flow_init is not None:
            coords1 = coords1 + flow_init # directly add the flow_init to coords1

        flow_predictions = []
        # laplace
        info_predictions = []
        info = torch.zeros_like(coords1)
        for itr in range(iters): # go over iterations
            coords1 = coords1.detach() # no update on the initial flow of image2.
            corr = corr_fn(coords1) # index correlation volume based on the coords of image2. If identical img1 and img2, corr should be 1

            flow = coords1 - coords0 # calcualte the flow. this is multi-level 
            with autocast(enabled=self.args.mixed_precision):
                net, up_mask, delta_data = self.update_block(net, inp, corr, flow, info) # do update block. require 4 things

            delta_flow = delta_data[:,:2]
            delta_info = delta_data[:,2:]
            
            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow # get new coords1
            info = info + delta_info


            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0) # up 8 times.
            else:
                flow_up, info_up = self.upsample_flow(coords1 - coords0, info, up_mask)
            
            flow_predictions.append(flow_up)
            info_predictions.append(info_up)

        if test_mode:
            return flow_up, info_up
            
        return flow_predictions, info_predictions # return the flow predictions
