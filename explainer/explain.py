""" explain.py

    Implementation of the explainer.
"""

import math
import time
import os
from scipy.stats import wasserstein_distance
import matplotlib
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import tensorboardX.utils
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.neighbors import NearestNeighbors
import sklearn.metrics as metrics
from sklearn.metrics import roc_auc_score, recall_score, precision_score, roc_auc_score, precision_recall_curve
from sklearn.cluster import DBSCAN

import pdb

import utils.io_utils as io_utils
import utils.train_utils as train_utils
import utils.graph_utils as graph_utils

def pdist(sample_1, sample_2, norm=2, eps=1e-5):
    """Compute the matrix of all squared pairwise distances.
    Arguments
    ---------
    sample_1 : torch.Tensor or Variable
        The first sample, should be of shape ``(n_1, d)``.
    sample_2 : torch.Tensor or Variable
        The second sample, should be of shape ``(n_2, d)``.
    norm : float
        The l_p norm to be used.
    Returns
    -------
    torch.Tensor or Variable
        Matrix of shape (n_1, n_2). The [i, j]-th entry is equal to
        ``|| sample_1[i, :] - sample_2[j, :] ||_p``."""
    n_1, n_2 = sample_1.size(0), sample_2.size(0)
    norm = float(norm)
    if norm == 2.:
        norms_1 = torch.sum(sample_1**2, dim=1, keepdim=True)
        norms_2 = torch.sum(sample_2**2, dim=1, keepdim=True)
        norms = (norms_1.expand(n_1, n_2) +
                 norms_2.transpose(0, 1).expand(n_1, n_2))
        distances_squared = norms - 2 * sample_1.mm(sample_2.t())
        return torch.sqrt(eps + torch.abs(distances_squared))
    else:
        dim = sample_1.size(1)
        expanded_1 = sample_1.unsqueeze(1).expand(n_1, n_2, dim)
        expanded_2 = sample_2.unsqueeze(0).expand(n_1, n_2, dim)
        differences = torch.abs(expanded_1 - expanded_2) ** norm
        inner = torch.sum(differences, dim=2, keepdim=False)
        return (eps + inner) ** (1. / norm)


def wasserstein(x, y, p=0.5, lam=10, its=10, sq=False, backpropT=False, cuda=False):
    """return W dist between x and y"""
    '''distance matrix M'''

    if len(x.shape)<2:
        x=x.unsqueeze(0)
    if len(y.shape)<2:
        y=y.unsqueeze(0)

    nx = x.shape[0]
    ny = y.shape[0]

    #x = x.squeeze()
    #y = y.squeeze()

    #    pdist = torch.nn.PairwiseDistance(p=2)


    M = pdist(x, y)  # distance_matrix(x,y,p=2)

    '''estimate lambda and delta'''
    M_mean = torch.mean(M)
    #M_drop = F.dropout(M, 10.0 / (nx * ny))
    M_drop = F.dropout(M, 0.0 / (nx * ny))
    delta = torch.max(M_drop).cpu().detach()
    eff_lam = (lam / M_mean).cpu().detach()

    '''compute new distance matrix'''
    Mt = M
    row = delta * torch.ones(M[0:1, :].shape)
    col = torch.cat([delta * torch.ones(M[:, 0:1].shape), torch.zeros((1, 1))], 0)
    if cuda:
        row = row.cuda()
        col = col.cuda()

    Mt = torch.cat([M, row], 0)
    Mt = torch.cat([Mt, col], 1)

    '''compute marginal'''
    a = torch.cat([p * torch.ones((nx, 1)) / nx, (1 - p) * torch.ones((1, 1))], 0)
    b = torch.cat([(1 - p) * torch.ones((ny, 1)) / ny, p * torch.ones((1, 1))], 0)

    '''compute kernel'''
    Mlam = eff_lam * Mt
    temp_term = torch.ones(1) * 1e-6
    if cuda:
        temp_term = temp_term.cuda()
        a = a.cuda()
        b = b.cuda()

    K = torch.exp(-Mlam) + temp_term
    U = K * Mt
    ainvK = K / a

    u = a

    for i in range(its):
        u = 1.0 / (ainvK.matmul(b / torch.t(torch.t(u).matmul(K))))
        if cuda:
            u = u.cuda()

    v = b / (torch.t(torch.t(u).matmul(K)))
    if cuda:
        v = v.cuda()


    upper_t = u * (torch.t(v) * K).detach()

    E = upper_t * Mt
    D = 2 * torch.sum(E)

    if cuda:
        D = D.cuda()

    return D, Mlam

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
Tensor = FloatTensor

class Explainer:
    def __init__(
            self,
            model,
            adj,
            feat,
            label,
            pred,
            train_idx,
            args,
            writer=None,
            print_training=True,
            graph_mode=False,
            graph_idx=False,
    ):
        self.model = model
        self.model.eval()
        self.adj = adj
        self.feat = feat
        self.label = label
        self.pred = pred
        self.train_idx = train_idx
        self.n_hops = args.num_gc_layers
        self.graph_mode = graph_mode
        self.graph_idx = graph_idx
        self.neighborhoods = None if self.graph_mode else graph_utils.neighborhoods(adj=self.adj, n_hops=self.n_hops, use_cuda=use_cuda)
        self.args = args
        self.writer = writer
        self.print_training = print_training

        self.use_wass=True
        self.use_KL=True

        self.explain_true=[]
        self.explain_true_wass=[]
        self.explain_true_minuswass=[]
        self.wass_dis=[]
        self.wass_dis_ori=[]
        self.wass_dis_att=[]
        self.start_wass_dis=[]
        self.start_wass_dis_ori=[]
        self.start_wass_dis_att=[]
        self.start_wass_dis_unfair=[]
        self.wass_dis_unfair=[]


        self.folder_name=self.args.dataset+'_explain_neighbors_both5_newnew'

        os.makedirs('log/'+self.folder_name,exist_ok=True)



        self.tensor_pred=torch.tensor(self.pred).cuda().softmax(-1)




    # Main method
    def explain(
            self, node_idx, graph_idx=0, graph_mode=False, unconstrained=False, model="exp"
    ):
        """Explain a single node prediction
        """
        # index of the query node in the new adj
        if graph_mode:
            node_idx_new = node_idx
            sub_adj = self.adj[graph_idx]
            sub_feat = self.feat[graph_idx, :]
            sub_label = self.label[graph_idx]
            neighbors = np.asarray(range(self.adj.shape[0]))
        else:
            print("node label: ", self.label[graph_idx][node_idx])
            node_idx_new, sub_adj, sub_feat, sub_label, neighbors = self.extract_neighborhood(
                node_idx, graph_idx
            )
            print("neigh graph idx: ", node_idx, node_idx_new)
            sub_label = np.expand_dims(sub_label, axis=0)

        sub_adj = np.expand_dims(sub_adj, axis=0)
        sub_feat = np.expand_dims(sub_feat, axis=0)

        adj   = torch.tensor(sub_adj, dtype=torch.float)
        x     = torch.tensor(sub_feat, requires_grad=True, dtype=torch.float)
        label = torch.tensor(sub_label, dtype=torch.long)

        if self.graph_mode:
            pred_label = np.argmax(self.pred[0][graph_idx], axis=0)
            print("Graph predicted label: ", pred_label)
        else:
            pred_label = np.argmax(self.pred[graph_idx][neighbors], axis=1)
            print("Node predicted label: ", pred_label[node_idx_new])

        explainer = ExplainModule(
            adj=adj,
            x=x,
            model=self.model,
            label=label,
            args=self.args,
            writer=self.writer,
            graph_idx=self.graph_idx,
            graph_mode=self.graph_mode,
        )
        if self.args.gpu:
            explainer = explainer.cuda()

        explainer_fair = ExplainModule(
            adj=adj,
            x=x,
            model=self.model,
            label=label,
            args=self.args,
            writer=self.writer,
            graph_idx=self.graph_idx,
            graph_mode=self.graph_mode,
        )
        if self.args.gpu:
            explainer_fair = explainer_fair.cuda()

        explainer_unfair = ExplainModule(
            adj=adj,
            x=x,
            model=self.model,
            label=label,
            args=self.args,
            writer=self.writer,
            graph_idx=self.graph_idx,
            graph_mode=self.graph_mode,
        )
        if self.args.gpu:
            explainer_unfair = explainer_unfair.cuda()



        self.model.eval()


        # gradient baseline
        if model == "grad":
            explainer.zero_grad()
            # pdb.set_trace()
            adj_grad = torch.abs(
                explainer.adj_feat_grad(node_idx_new, pred_label[node_idx_new])[0]
            )[graph_idx]
            masked_adj = adj_grad + adj_grad.t()
            masked_adj = nn.functional.sigmoid(masked_adj)
            masked_adj = masked_adj.cpu().detach().numpy() * sub_adj.squeeze()

            ori_mask=masked_adj.cpu().detach().numpy()
            index=np.argsort(ori_mask.flatten())[-threshold:]
            ori_mask_new=np.zeros(ori_mask.shape).flatten()
            ori_mask_new[index]=1
            #ori_mask=np.argwhere(ori_mask_new.reshape(ori_mask.shape)>0)

            new_adj=adj.cuda()*torch.tensor(ori_mask_new.reshape(ori_mask.shape),dtype=torch.float32).cuda()

            ypred, adj_att = self.model(x, new_adj)
            node_pred = ypred[self.graph_idx, node_idx, :]
            ypred = nn.Softmax(dim=0)(node_pred)


            index=self.feat[0][:,-1]==1-self.feat[0][node_idx][-1]
            new_tensor_pred=self.tensor_pred[graph_idx].clone()
            if epoch!=0: new_tensor_pred[node_idx]=ypred
            new_index=self.feat[0][:,-1]==self.feat[0][node_idx][-1]
            #wass_dis=wasserstein(self.tensor_pred[graph_idx][index],new_tensor_pred[new_index],cuda=True)[0]*100000
            #wass_dis=wasserstein(self.tensor_pred[graph_idx][index][indices[0]],new_tensor_pred[index_same][indices_same_class[0]],cuda=True)[0]*100000
            wass_dis=wasserstein_distance(self.tensor_pred[graph_idx][index][:,0].cpu().detach().numpy(),new_tensor_pred[new_index][:,0].cpu().detach().numpy())*100000
            if epoch==0:
                self.start_wass_dis_grad.append(wass_dis)
            else:
                self.wass_dis_grad.append(wass_dis)


        else:
            explainer.train()
            begin_time = time.time()
            for epoch in range(self.args.num_epochs):
                explainer.zero_grad()
                explainer.optimizer.zero_grad()
                ypred_ori, adj_atts_ori = explainer(node_idx_new, unconstrained=unconstrained)
                loss = explainer.loss(ypred_ori, pred_label, node_idx_new, epoch)


                k_neighbors=5
                threshold=500
                index=self.feat[0][:,-1]==1-self.feat[0][node_idx][-1]
                nbrs = NearestNeighbors(n_neighbors=min(self.feat[0][index,:-1].shape[0],k_neighbors), algorithm='ball_tree').fit(self.feat[0][index,:-1])
                distances, indices = nbrs.kneighbors(self.feat[0][node_idx,:-1].reshape(1, -1))
                #print(self.tensor_pred[0][indices[0]].shape)
                #wass_dis=wasserstein(self.tensor_pred[graph_idx][indices[0]],ypred,cuda=True)[0]


                #取出属于本类的neighbors的点
                index_same=self.feat[0][:,-1]==self.feat[0][node_idx][-1]
                nbrs = NearestNeighbors(n_neighbors=min(self.feat[0][index_same,:-1].shape[0],k_neighbors), algorithm='ball_tree').fit(self.feat[0][index_same,:-1])
                distances, indices_same_class = nbrs.kneighbors(self.feat[0][node_idx,:-1].reshape(1, -1))



                #取出属于本类的全部pred，并替换其中一个
                index=self.feat[0][:,-1]==1-self.feat[0][node_idx][-1]
                new_tensor_pred=self.tensor_pred[graph_idx].clone()
                new_tensor_pred[node_idx]=ypred_ori
                new_index=self.feat[0][:,-1]==self.feat[0][node_idx][-1]
                wass_dis=wasserstein(self.tensor_pred[graph_idx][index][indices[0]],new_tensor_pred[index_same][indices_same_class[0]],cuda=True)[0]

                loss+=wass_dis*100


                loss.backward()


                if epoch==0 or epoch+1==self.args.num_epochs:
                    ypred,adj_atts = explainer.cal_WD_ypred(node_idx_new,threshold=threshold)

                    index=self.feat[0][:,-1]==1-self.feat[0][node_idx][-1]
                    new_tensor_pred=self.tensor_pred[graph_idx].clone()


                    if epoch!=0: new_tensor_pred[node_idx]=ypred


                    new_index=self.feat[0][:,-1]==self.feat[0][node_idx][-1]
                    #wass_dis=wasserstein(self.tensor_pred[graph_idx][index],new_tensor_pred[new_index],cuda=True)[0]*100000
                    #wass_dis=wasserstein(self.tensor_pred[graph_idx][index][indices[0]],new_tensor_pred[index_same][indices_same_class[0]],cuda=True)[0]*100000
                    wass_dis=wasserstein_distance(self.tensor_pred[graph_idx][index][:,0].cpu().detach().numpy(),new_tensor_pred[new_index][:,0].cpu().detach().numpy())*100000
                    if epoch==0:
                        self.start_wass_dis_ori.append(wass_dis)
                    else:
                        self.wass_dis_ori.append(wass_dis)






                explainer.optimizer.step()
                if explainer.scheduler is not None:
                    explainer.scheduler.step()

                if self.use_wass:

                    k_neighbors=5#100



                    explainer_unfair.zero_grad()
                    explainer_unfair.optimizer.zero_grad()



                    ypred_unfair, adj_atts = explainer_unfair(node_idx_new, unconstrained=unconstrained)
                    loss = explainer_unfair.loss(ypred_unfair, pred_label, node_idx_new, epoch)
                    #loss=0

                    #取出不同sens feature的点？
                    index=self.feat[0][:,-1]==1-self.feat[0][node_idx][-1]
                    nbrs = NearestNeighbors(n_neighbors=min(self.feat[0][index,:-1].shape[0],k_neighbors), algorithm='ball_tree').fit(self.feat[0][index,:-1])
                    distances, indices = nbrs.kneighbors(self.feat[0][node_idx,:-1].reshape(1, -1))
                    #print(self.tensor_pred[0][indices[0]].shape)
                    #wass_dis=wasserstein(self.tensor_pred[graph_idx][indices[0]],ypred,cuda=True)[0]


                    #取出属于本类的neighbors的点
                    index_same=self.feat[0][:,-1]==self.feat[0][node_idx][-1]
                    nbrs = NearestNeighbors(n_neighbors=min(self.feat[0][index_same,:-1].shape[0],k_neighbors), algorithm='ball_tree').fit(self.feat[0][index_same,:-1])
                    distances, indices_same_class = nbrs.kneighbors(self.feat[0][node_idx,:-1].reshape(1, -1))



                    #取出属于本类的全部pred，并替换其中一个
                    index=self.feat[0][:,-1]==1-self.feat[0][node_idx][-1]
                    new_tensor_pred=self.tensor_pred[graph_idx].clone()
                    new_tensor_pred[node_idx]=ypred_unfair
                    new_index=self.feat[0][:,-1]==self.feat[0][node_idx][-1]





                    #wass_dis=wasserstein(self.tensor_pred[graph_idx][index],new_tensor_pred[new_index],cuda=True)[0]
                    #wass_dis=wasserstein(self.tensor_pred[graph_idx][indices[0]],ypred_unfair,cuda=True)[0]
                    wass_dis=wasserstein(self.tensor_pred[graph_idx][index][indices[0]],new_tensor_pred[index_same][indices_same_class[0]],cuda=True)[0]






                    loss-=wass_dis*10000

                    loss.backward(retain_graph=True)
                    explainer_unfair.optimizer.step()
                    if explainer_unfair.scheduler is not None:
                        explainer_unfair.scheduler.step()





                    threshold=500

                    if epoch==0 or epoch+1==self.args.num_epochs:
                        ypred_unfair,adj_atts = explainer_unfair.cal_WD_ypred(node_idx_new,threshold=threshold)

                        index=self.feat[0][:,-1]==1-self.feat[0][node_idx][-1]
                        new_tensor_pred=self.tensor_pred[graph_idx].clone()
                        new_tensor_pred[node_idx]=ypred_unfair
                        new_index=self.feat[0][:,-1]==self.feat[0][node_idx][-1]
                        #wass_dis=(wasserstein(self.tensor_pred[graph_idx][index],new_tensor_pred[new_index],cuda=True)[0]*100000).cpu().detach().numpy()
                        wass_dis=wasserstein_distance(self.tensor_pred[graph_idx][index][:,0].cpu().detach().numpy(),new_tensor_pred[new_index][:,0].cpu().detach().numpy())*100000
                        if epoch==0:
                            self.start_wass_dis_unfair.append(wass_dis)
                        else:
                            self.wass_dis_unfair.append(wass_dis)




                    explainer_fair.zero_grad()
                    explainer_fair.optimizer.zero_grad()


                    ypred, adj_atts = explainer_fair(node_idx_new, unconstrained=unconstrained)
                    loss = explainer_fair.loss(ypred, pred_label, node_idx_new, epoch)
                    #loss=0




                    index=self.feat[0][:,-1]==1-self.feat[0][node_idx][-1]
                    new_tensor_pred=self.tensor_pred[graph_idx].clone()
                    new_tensor_pred[node_idx]=ypred
                    new_index=self.feat[0][:,-1]==self.feat[0][node_idx][-1]



                    #wass_dis=wasserstein(self.tensor_pred[graph_idx][index],new_tensor_pred[new_index],cuda=True)[0]
                    #wass_dis=wasserstein(self.tensor_pred[graph_idx][indices[0]],ypred,cuda=True)[0]
                    wass_dis=wasserstein(self.tensor_pred[graph_idx][index][indices[0]],new_tensor_pred[index_same][indices_same_class[0]],cuda=True)[0]



                    WD_loss=wass_dis*10000

                    loss+=wass_dis*10000
                    #loss+=wasserstein(self.tensor_pred[graph_idx][indices[0]],ypred,cuda=True)[0]*10


                    if epoch==0 or epoch+1==self.args.num_epochs:
                        ypred_fair,adj_atts = explainer_fair.cal_WD_ypred(node_idx_new,threshold=threshold)

                        index=self.feat[0][:,-1]==1-self.feat[0][node_idx][-1]
                        new_tensor_pred=self.tensor_pred[graph_idx].clone()
                        new_tensor_pred[node_idx]=ypred_fair
                        new_index=self.feat[0][:,-1]==self.feat[0][node_idx][-1]
                        #wass_dis=(wasserstein(self.tensor_pred[graph_idx][index]+1e-9,new_tensor_pred[new_index]+1e-9,cuda=True)[0]*100000).cpu().detach().numpy()
                        wass_dis=wasserstein_distance(self.tensor_pred[graph_idx][index][:,0].cpu().detach().numpy(),new_tensor_pred[new_index][:,0].cpu().detach().numpy())*100000



                        if epoch==0:
                            self.start_wass_dis.append(wass_dis)
                        else:
                            self.wass_dis.append(wass_dis)


                    if self.use_KL:
                        #print(F.kl_div(explainer_fair.mask.flatten()+1e-9,explainer_unfair.mask.flatten()+1e-9))
                        #print(F.kl_div(torch.log(explainer_fair.mask.flatten().softmax(-1)),explainer_unfair.mask.flatten().softmax(-1)))


                        KL_loss=(0.5*F.kl_div(torch.log(explainer_fair.masked_adj.flatten().softmax(-1)),explainer_unfair.masked_adj.flatten().softmax(-1))*+0.5*F.kl_div(torch.log(explainer_unfair.masked_adj.flatten().softmax(-1)),explainer_fair.masked_adj.flatten().softmax(-1)))*1e20


                        loss-=KL_loss
                        #loss-=wasserstein(explainer_fair.mask,explainer_unfair.mask,cuda=True)[0]*10

                    #loss+=wasserstein(self.tensor_pred[graph_idx][neighbors][node_idx_new],ypred)[0]
                    loss.backward()

                    explainer_fair.optimizer.step()
                    if explainer_fair.scheduler is not None:
                        explainer_fair.scheduler.step()










                mask_density = explainer.mask_density()
                if self.print_training and epoch%20==0:
                    print(
                        "epoch: ",
                        epoch,
                        "; loss: {:.4f}".format(
                            loss.item()/1000),
                        "; WD loss: {:.4f}".format(
                            WD_loss.item()/1000),
                        "; KL loss: {:.4f}".format(
                            KL_loss.item()/1000),
                        "; mask density: {:.4f} ".format(
                            mask_density.item())
                    )
                single_subgraph_label = sub_label.squeeze()

                if self.writer is not None:
                    self.writer.add_scalar("mask/density", mask_density, epoch)
                    self.writer.add_scalar(
                        "optimization/lr",
                        explainer.optimizer.param_groups[0]["lr"],
                        epoch,
                    )
                    #if epoch % 25 == 0:
                    if False:
                        explainer.log_mask(epoch)
                        explainer.log_masked_adj(
                            node_idx_new, epoch, label=single_subgraph_label
                        )
                        explainer.log_adj_grad(
                            node_idx_new, pred_label, epoch, label=single_subgraph_label
                        )

                    if epoch == 0:
                        if self.model.att:
                            # explain node
                            print("adj att size: ", adj_atts_ori.size())
                            adj_att = torch.sum(adj_atts_ori[0], dim=2)
                            # adj_att = adj_att[neighbors][:, neighbors]
                            node_adj_att = adj_att * adj.float().cuda()
                            io_utils.log_matrix(
                                self.writer, node_adj_att[0], "att/matrix", epoch
                            )
                            node_adj_att = node_adj_att[0].cpu().detach().numpy()
                            G = io_utils.denoise_graph(
                                node_adj_att,
                                node_idx_new,
                                threshold=3.8,  # threshold_num=20,
                                max_component=True,
                            )
                            io_utils.log_graph(
                                self.writer,
                                G,
                                name="att/graph",
                                identify_self=not self.graph_mode,
                                nodecolor="label",
                                edge_vmax=None,
                                args=self.args,
                            )
                if model != "exp":
                    break

            print("finished training in ", time.time() - begin_time)
            if model == "exp":
                masked_adj = (
                        explainer.masked_adj[0].cpu().detach().numpy() * sub_adj.squeeze()
                )
                masked_adj_fair = (
                        explainer_fair.masked_adj[0].cpu().detach().numpy() * sub_adj.squeeze()
                )
                masked_adj_unfair = (
                        explainer_unfair.masked_adj[0].cpu().detach().numpy() * sub_adj.squeeze()
                )

                #print(masked_adj[:10,:10])
                #print(sub_adj.squeeze()[:10,:10])
                #print(self.adj[0][neighbors[:10],:][:,neighbors[:10]])



            else:
                adj_atts_ori = nn.functional.sigmoid(adj_atts_ori).squeeze()
                masked_adj = adj_atts_ori.cpu().detach().numpy() * sub_adj.squeeze()



                ori_mask=masked_adj.cpu().detach().numpy()
                index=np.argsort(ori_mask.flatten())[-threshold:]
                ori_mask_new=np.zeros(ori_mask.shape).flatten()
                ori_mask_new[index]=1
                #ori_mask=np.argwhere(ori_mask_new.reshape(ori_mask.shape)>0)

                new_adj=adj.cuda()*torch.tensor(ori_mask_new.reshape(ori_mask.shape),dtype=torch.float32).cuda()

                ypred, adj_att = self.model(x, new_adj)
                node_pred = ypred[self.graph_idx, node_idx, :]
                ypred = nn.Softmax(dim=0)(node_pred)


                index=self.feat[0][:,-1]==1-self.feat[0][node_idx][-1]
                new_tensor_pred=self.tensor_pred[graph_idx].clone()
                if epoch!=0: new_tensor_pred[node_idx]=ypred
                new_index=self.feat[0][:,-1]==self.feat[0][node_idx][-1]
                #wass_dis=wasserstein(self.tensor_pred[graph_idx][index],new_tensor_pred[new_index],cuda=True)[0]*100000
                #wass_dis=wasserstein(self.tensor_pred[graph_idx][index][indices[0]],new_tensor_pred[index_same][indices_same_class[0]],cuda=True)[0]*100000
                wass_dis=wasserstein_distance(self.tensor_pred[graph_idx][index][:,0].cpu().detach().numpy(),new_tensor_pred[new_index][:,0].cpu().detach().numpy())*100000
                if epoch==0:
                    self.start_wass_dis_att.append(wass_dis)
                else:
                    self.wass_dis_att.append(wass_dis)




        if np.argmax(self.pred[graph_idx][neighbors][node_idx_new])==np.argmax(ypred_ori.cpu().detach().numpy()):
            self.explain_true.append(1)
        else:
            self.explain_true.append(0)

        if np.argmax(self.pred[graph_idx][neighbors][node_idx_new])==np.argmax(ypred.cpu().detach().numpy()):
            self.explain_true_wass.append(1)
        else:
            self.explain_true_wass.append(0)

        if np.argmax(self.pred[graph_idx][neighbors][node_idx_new])==np.argmax(ypred_unfair.cpu().detach().numpy()):
            self.explain_true_minuswass.append(1)
        else:
            self.explain_true_minuswass.append(0)








        fname = 'neighbors_' + io_utils.gen_explainer_prefix(self.args) + (
                'node_idx_'+str(node_idx)+'graph_idx_'+str(self.graph_idx)+'.npy')
        with open(os.path.join(self.args.logdir, self.folder_name,fname), 'wb') as outfile:
            np.save(outfile, np.asarray(neighbors.tolist()+[node_idx_new]))
            print("Saved adjacency matrix to ", fname)


        fname = 'masked_adj_' + io_utils.gen_explainer_prefix(self.args) + (
                'node_idx_'+str(node_idx)+'graph_idx_'+str(self.graph_idx)+'KL'+str(self.use_KL)+'.npy')
        with open(os.path.join(self.args.logdir, self.folder_name,fname), 'wb') as outfile:
            np.save(outfile, np.asarray([masked_adj.copy(),masked_adj_fair.copy(),masked_adj_unfair.copy()]))
            print("Saved adjacency matrix to ", fname)




        return masked_adj


    # NODE EXPLAINER
    def explain_nodes(self, node_indices, args, graph_idx=0):
        """
        Explain nodes

        Args:
            - node_indices  :  Indices of the nodes to be explained
            - args          :  Program arguments (mainly for logging paths)
            - graph_idx     :  Index of the graph to explain the nodes from (if multiple).
        """
        masked_adjs = [
            self.explain(node_idx, graph_idx=graph_idx) for node_idx in node_indices
        ]
        ref_idx = node_indices[0]
        ref_adj = masked_adjs[0]
        curr_idx = node_indices[1]
        curr_adj = masked_adjs[1]
        new_ref_idx, _, ref_feat, _, _ = self.extract_neighborhood(ref_idx)
        new_curr_idx, _, curr_feat, _, _ = self.extract_neighborhood(curr_idx)

        G_ref = io_utils.denoise_graph(ref_adj, new_ref_idx, ref_feat, threshold=0.1)
        denoised_ref_feat = np.array(
            [G_ref.nodes[node]["feat"] for node in G_ref.nodes()]
        )
        denoised_ref_adj = nx.to_numpy_matrix(G_ref)
        # ref center node
        ref_node_idx = list(G_ref.nodes()).index(new_ref_idx)

        G_curr = io_utils.denoise_graph(
            curr_adj, new_curr_idx, curr_feat, threshold=0.1
        )
        denoised_curr_feat = np.array(
            [G_curr.nodes[node]["feat"] for node in G_curr.nodes()]
        )
        denoised_curr_adj = nx.to_numpy_matrix(G_curr)
        # curr center node
        curr_node_idx = list(G_curr.nodes()).index(new_curr_idx)

        P, aligned_adj, aligned_feat = self.align(
            denoised_ref_feat,
            denoised_ref_adj,
            ref_node_idx,
            denoised_curr_feat,
            denoised_curr_adj,
            curr_node_idx,
            args=args,
        )
        io_utils.log_matrix(self.writer, P, "align/P", 0)

        G_ref = nx.convert_node_labels_to_integers(G_ref)
        io_utils.log_graph(self.writer, G_ref, "align/ref")
        G_curr = nx.convert_node_labels_to_integers(G_curr)
        io_utils.log_graph(self.writer, G_curr, "align/before")

        P = P.cpu().detach().numpy()
        aligned_adj = aligned_adj.cpu().detach().numpy()
        aligned_feat = aligned_feat.cpu().detach().numpy()

        aligned_idx = np.argmax(P[:, curr_node_idx])
        print("aligned self: ", aligned_idx)
        G_aligned = io_utils.denoise_graph(
            aligned_adj, aligned_idx, aligned_feat, threshold=0.5
        )
        io_utils.log_graph(self.writer, G_aligned, "mask/aligned")

        # io_utils.log_graph(self.writer, aligned_adj.cpu().detach().numpy(), new_curr_idx,
        #        'align/aligned', epoch=1)

        return masked_adjs


    def explain_nodes_gnn_stats(self, node_indices, args, graph_idx=0, model="exp"):
        masked_adjs = [
            self.explain(node_idx, graph_idx=graph_idx, model=model)
            for node_idx in node_indices
        ]
        # pdb.set_trace()

        if not self.use_wass:
            self.args.dataset+='_no_wass'


        graphs = []
        feats = []
        adjs = []
        pred_all = []
        real_all = []
        for i, idx in enumerate(node_indices):
            new_idx, _, feat, _, _ = self.extract_neighborhood(idx)
            G = io_utils.denoise_graph(masked_adjs[i], new_idx, feat, threshold_num=20)
            #pred, real = self.make_pred_real(masked_adjs[i], new_idx)
            #pred_all.append(pred)
            #real_all.append(real)
            denoised_feat = np.array([G.nodes[node]["feat"] for node in G.nodes()])
            denoised_adj = nx.to_numpy_matrix(G)
            graphs.append(G)
            feats.append(denoised_feat)
            adjs.append(denoised_adj)
            io_utils.log_graph(
                self.writer,
                G,
                "graph/{}_{}_{}".format(self.args.dataset, model, i),
                identify_self=True,
                args=self.args
            )


        if model=='att':
            explain_acc=np.array([self.explain_true,self.explain_true_wass,self.explain_true_minuswass,self.wass_dis_att,self.wass_dis,self.wass_dis_unfair,
                                  self.start_wass_dis_att,self.start_wass_dis,self.start_wass_dis_unfair])

        else:


            explain_acc=np.array([self.explain_true,self.explain_true_wass,self.explain_true_minuswass,self.wass_dis_ori,self.wass_dis,self.wass_dis_unfair,
                                  self.start_wass_dis_ori,self.start_wass_dis,self.start_wass_dis_unfair])

        fname = 'explain_acc_' + io_utils.gen_explainer_prefix(self.args)+'graph_idx_'+str(self.graph_idx)+'.npy'
        with open(os.path.join(self.args.logdir, self.folder_name,fname), 'wb') as outfile:
            np.save(outfile, np.asarray(explain_acc.copy()))
            print("Saved explain_acc matrix to ", fname)


            #pred_all = np.concatenate((pred_all), axis=0)
        #real_all = np.concatenate((real_all), axis=0)
        #
        #auc_all = roc_auc_score(real_all, pred_all)
        #precision, recall, thresholds = precision_recall_curve(real_all, pred_all)
        #
        #plt.switch_backend("agg")
        #plt.plot(recall, precision)
        #plt.savefig("log/pr/pr_" + self.args.dataset + "_" + model + ".png")
        #
        #plt.close()
        #
        #auc_all = roc_auc_score(real_all, pred_all)
        #precision, recall, thresholds = precision_recall_curve(real_all, pred_all)
        #
        #plt.switch_backend("agg")
        #plt.plot(recall, precision)
        #plt.savefig("log/pr/pr_" + self.args.dataset + "_" + model + ".png")
        #
        #plt.close()
        #
        #with open("log/pr/auc_" + self.args.dataset + "_" + model + ".txt", "w") as f:
        #    f.write(
        #        "dataset: {}, model: {}, auc: {}\n".format(
        #            self.args.dataset, "exp", str(auc_all)
        #        )
        #    )

        return masked_adjs

    # GRAPH EXPLAINER
    def explain_graphs(self, graph_indices):
        """
        Explain graphs.
        """
        masked_adjs = []

        for graph_idx in graph_indices:
            masked_adj = self.explain(node_idx=0, graph_idx=graph_idx, graph_mode=True)
            G_denoised = io_utils.denoise_graph(
                masked_adj,
                0,
                threshold_num=20,
                feat=self.feat[graph_idx],
                max_component=False,
            )
            label = self.label[graph_idx]
            io_utils.log_graph(
                self.writer,
                G_denoised,
                "graph/graphidx_{}_label={}".format(graph_idx, label),
                identify_self=False,
                nodecolor="feat",
                args=self.args
            )
            masked_adjs.append(masked_adj)

            G_orig = io_utils.denoise_graph(
                self.adj[graph_idx],
                0,
                feat=self.feat[graph_idx],
                threshold=None,
                max_component=False,
            )

            io_utils.log_graph(
                self.writer,
                G_orig,
                "graph/graphidx_{}".format(graph_idx),
                identify_self=False,
                nodecolor="feat",
                args=self.args
            )

        # plot cmap for graphs' node features
        io_utils.plot_cmap_tb(self.writer, "tab20", 20, "tab20_cmap")

        return masked_adjs

    def log_representer(self, rep_val, sim_val, alpha, graph_idx=0):
        """ visualize output of representer instances. """
        rep_val = rep_val.cpu().detach().numpy()
        sim_val = sim_val.cpu().detach().numpy()
        alpha = alpha.cpu().detach().numpy()
        sorted_rep = sorted(range(len(rep_val)), key=lambda k: rep_val[k])
        print(sorted_rep)
        topk = 5
        most_neg_idx = [sorted_rep[i] for i in range(topk)]
        most_pos_idx = [sorted_rep[-i - 1] for i in range(topk)]
        rep_idx = [most_pos_idx, most_neg_idx]

        if self.graph_mode:
            pred = np.argmax(self.pred[0][graph_idx], axis=0)
        else:
            pred = np.argmax(self.pred[graph_idx][self.train_idx], axis=1)
        print(metrics.confusion_matrix(self.label[graph_idx][self.train_idx], pred))
        plt.switch_backend("agg")
        fig = plt.figure(figsize=(5, 3), dpi=600)
        for i in range(2):
            for j in range(topk):
                idx = self.train_idx[rep_idx[i][j]]
                print(
                    "node idx: ",
                    idx,
                    "; node label: ",
                    self.label[graph_idx][idx],
                    "; pred: ",
                    pred,
                )

                idx_new, sub_adj, sub_feat, sub_label, neighbors = self.extract_neighborhood(
                    idx, graph_idx
                )
                G = nx.from_numpy_matrix(sub_adj)
                node_colors = [1 for i in range(G.number_of_nodes())]
                node_colors[idx_new] = 0
                # node_color='#336699',

                ax = plt.subplot(2, topk, i * topk + j + 1)
                nx.draw(
                    G,
                    pos=nx.spring_layout(G),
                    with_labels=True,
                    font_size=4,
                    node_color=node_colors,
                    cmap=plt.get_cmap("Set1"),
                    vmin=0,
                    vmax=8,
                    edge_vmin=0.0,
                    edge_vmax=1.0,
                    width=0.5,
                    node_size=25,
                    alpha=0.7,
                )
                ax.xaxis.set_visible(False)
        fig.canvas.draw()
        self.writer.add_image(
            "local/representer_neigh", tensorboardX.utils.figure_to_image(fig), 0
        )

    def representer(self):
        """
        experiment using representer theorem for finding supporting instances.
        https://papers.nips.cc/paper/8141-representer-point-selection-for-explaining-deep-neural-networks.pdf
        """
        self.model.train()
        self.model.zero_grad()
        adj = torch.tensor(self.adj, dtype=torch.float)
        x = torch.tensor(self.feat, requires_grad=True, dtype=torch.float)
        label = torch.tensor(self.label, dtype=torch.long)
        if self.args.gpu:
            adj, x, label = adj.cuda(), x.cuda(), label.cuda()

        preds, _ = self.model(x, adj)
        preds.retain_grad()
        self.embedding = self.model.embedding_tensor
        loss = self.model.loss(preds, label)
        loss.backward()
        self.preds_grad = preds.grad
        pred_idx = np.expand_dims(np.argmax(self.pred, axis=2), axis=2)
        pred_idx = torch.LongTensor(pred_idx)
        if self.args.gpu:
            pred_idx = pred_idx.cuda()
        self.alpha = self.preds_grad


    # Utilities
    def extract_neighborhood(self, node_idx, graph_idx=0):
        """Returns the neighborhood of a given ndoe."""
        neighbors_adj_row = self.neighborhoods[graph_idx][node_idx, :]
        # index of the query node in the new adj
        node_idx_new = sum(neighbors_adj_row[:node_idx])
        neighbors = np.nonzero(neighbors_adj_row)[0]
        sub_adj = self.adj[graph_idx][neighbors][:, neighbors]
        sub_feat = self.feat[graph_idx, neighbors]
        sub_label = self.label[graph_idx][neighbors]
        return node_idx_new, sub_adj, sub_feat, sub_label, neighbors

    def align(
            self, ref_feat, ref_adj, ref_node_idx, curr_feat, curr_adj, curr_node_idx, args
    ):
        """ Tries to find an alignment between two graphs.
        """
        ref_adj = torch.FloatTensor(ref_adj)
        curr_adj = torch.FloatTensor(curr_adj)

        ref_feat = torch.FloatTensor(ref_feat)
        curr_feat = torch.FloatTensor(curr_feat)

        P = nn.Parameter(torch.FloatTensor(ref_adj.shape[0], curr_adj.shape[0]))
        with torch.no_grad():
            nn.init.constant_(P, 1.0 / ref_adj.shape[0])
            P[ref_node_idx, :] = 0.0
            P[:, curr_node_idx] = 0.0
            P[ref_node_idx, curr_node_idx] = 1.0
        opt = torch.optim.Adam([P], lr=0.01, betas=(0.5, 0.999))
        for i in range(args.align_steps):
            opt.zero_grad()
            feat_loss = torch.norm(P @ curr_feat - ref_feat)

            aligned_adj = P @ curr_adj @ torch.transpose(P, 0, 1)
            align_loss = torch.norm(aligned_adj - ref_adj)
            loss = feat_loss + align_loss
            loss.backward()  # Calculate gradients
            self.writer.add_scalar("optimization/align_loss", loss, i)
            print("iter: ", i, "; loss: ", loss)
            opt.step()

        return P, aligned_adj, P @ curr_feat

    def make_pred_real(self, adj, start):
        # house graph
        return 0,0
        if True:
            # num_pred = max(G.number_of_edges(), 6)
            pred = adj[np.triu(adj) > 0]
            real = adj.copy()

            if real[start][start + 1] > 0:
                real[start][start + 1] = 10
            if real[start + 1][start + 2] > 0:
                real[start + 1][start + 2] = 10
            if real[start + 2][start + 3] > 0:
                real[start + 2][start + 3] = 10
            if real[start][start + 3] > 0:
                real[start][start + 3] = 10
            if real[start][start + 4] > 0:
                real[start][start + 4] = 10
            if real[start + 1][start + 4]:
                real[start + 1][start + 4] = 10
            real = real[np.triu(real) > 0]
            real[real != 10] = 0
            real[real == 10] = 1

        # cycle graph
        elif self.args.dataset == "syn4":
            pred = adj[np.triu(adj) > 0]
            real = adj.copy()
            # pdb.set_trace()
            if real[start][start + 1] > 0:
                real[start][start + 1] = 10
            if real[start + 1][start + 2] > 0:
                real[start + 1][start + 2] = 10
            if real[start + 2][start + 3] > 0:
                real[start + 2][start + 3] = 10
            if real[start + 3][start + 4] > 0:
                real[start + 3][start + 4] = 10
            if real[start + 4][start + 5] > 0:
                real[start + 4][start + 5] = 10
            if real[start][start + 5]:
                real[start][start + 5] = 10
            real = real[np.triu(real) > 0]
            real[real != 10] = 0
            real[real == 10] = 1

        return pred, real


class ExplainModule(nn.Module):
    def __init__(
            self,
            adj,
            x,
            model,
            label,
            args,
            graph_idx=0,
            writer=None,
            use_sigmoid=True,
            graph_mode=False,
    ):
        super(ExplainModule, self).__init__()
        self.adj = adj
        self.x = x
        self.model = model
        self.label = label
        self.graph_idx = graph_idx
        self.args = args
        self.writer = writer
        self.mask_act = args.mask_act
        self.use_sigmoid = use_sigmoid
        self.graph_mode = graph_mode

        init_strategy = "normal"
        num_nodes = adj.size()[1]
        self.mask, self.mask_bias = self.construct_edge_mask(
            num_nodes, init_strategy=init_strategy
        )

        self.feat_mask = self.construct_feat_mask(x.size(-1), init_strategy="constant")
        params = [self.mask, self.feat_mask]
        if self.mask_bias is not None:
            params.append(self.mask_bias)
        # For masking diagonal entries
        self.diag_mask = torch.ones(num_nodes, num_nodes) - torch.eye(num_nodes)
        if args.gpu:
            self.diag_mask = self.diag_mask.cuda()

        self.scheduler, self.optimizer = train_utils.build_optimizer(args, params)

        self.coeffs = {
            "size": 0.005,
            "feat_size": 1.0,
            "ent": 1.0,
            "feat_ent": 0.1,
            "grad": 0,
            "lap": 1.0,
        }

    def construct_feat_mask(self, feat_dim, init_strategy="normal"):
        mask = nn.Parameter(torch.FloatTensor(feat_dim))
        if init_strategy == "normal":
            std = 0.1
            with torch.no_grad():
                mask.normal_(1.0, std)
        elif init_strategy == "constant":
            with torch.no_grad():
                nn.init.constant_(mask, 0.0)
                # mask[0] = 2
        return mask

    def construct_edge_mask(self, num_nodes, init_strategy="normal", const_val=1.0):
        mask = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes))
        if init_strategy == "normal":
            std = nn.init.calculate_gain("relu") * math.sqrt(
                2.0 / (num_nodes + num_nodes)
            )
            with torch.no_grad():
                mask.normal_(1.0, std)
                # mask.clamp_(0.0, 1.0)
        elif init_strategy == "const":
            nn.init.constant_(mask, const_val)

        if self.args.mask_bias:
            mask_bias = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes))
            nn.init.constant_(mask_bias, 0.0)
        else:
            mask_bias = None

        return mask, mask_bias

    def _masked_adj(self):
        sym_mask = self.mask
        if self.mask_act == "sigmoid":
            sym_mask = torch.sigmoid(self.mask)
        elif self.mask_act == "ReLU":
            sym_mask = nn.ReLU()(self.mask)
        sym_mask = (sym_mask + sym_mask.t()) / 2
        adj = self.adj.cuda() if self.args.gpu else self.adj
        masked_adj = adj * sym_mask
        if self.args.mask_bias:
            bias = (self.mask_bias + self.mask_bias.t()) / 2
            bias = nn.ReLU6()(bias * 6) / 6
            masked_adj += (bias + bias.t()) / 2
        return masked_adj * self.diag_mask

    def mask_density(self):
        mask_sum = torch.sum(self._masked_adj()).cpu()
        adj_sum = torch.sum(self.adj)
        return mask_sum / adj_sum

    def forward(self, node_idx, unconstrained=False, mask_features=False, marginalize=False):
        x = self.x.cuda() if self.args.gpu else self.x

        if unconstrained:
            sym_mask = torch.sigmoid(self.mask) if self.use_sigmoid else self.mask
            self.masked_adj = (
                    torch.unsqueeze((sym_mask + sym_mask.t()) / 2, 0) * self.diag_mask
            )
        else:
            self.masked_adj = self._masked_adj()
            if mask_features:
                feat_mask = (
                    torch.sigmoid(self.feat_mask)
                    if self.use_sigmoid
                    else self.feat_mask
                )
                if marginalize:
                    std_tensor = torch.ones_like(x, dtype=torch.float) / 2
                    mean_tensor = torch.zeros_like(x, dtype=torch.float) - x
                    z = torch.normal(mean=mean_tensor, std=std_tensor)
                    x = x + z * (1 - feat_mask)
                else:
                    x = x * feat_mask

        ypred, adj_att = self.model(x, self.masked_adj)
        if self.graph_mode:
            res = nn.Softmax(dim=0)(ypred[0])
        else:
            node_pred = ypred[self.graph_idx, node_idx, :]
            res = nn.Softmax(dim=0)(node_pred)
        return res, adj_att


    def cal_WD_ypred(self, node_idx,threshold):
        x = self.x.cuda() if self.args.gpu else self.x
        self.masked_adj = self._masked_adj()

        if threshold<1:

            ypred, adj_att = self.model(x, self.masked_adj*(self.masked_adj>=threshold))
            node_pred = ypred[self.graph_idx, node_idx, :]
            res = nn.Softmax(dim=0)(node_pred)
            return res, adj_att
        else:
            ori_mask=self.masked_adj.cpu().detach().numpy()
            index=np.argsort(ori_mask.flatten())[-threshold:]
            ori_mask_new=np.zeros(ori_mask.shape).flatten()
            ori_mask_new[index]=1
            #ori_mask=np.argwhere(ori_mask_new.reshape(ori_mask.shape)>0)

            new_adj=self.adj.cuda()*torch.tensor(ori_mask_new.reshape(ori_mask.shape),dtype=torch.float32).cuda()

            ypred, adj_att = self.model(x, new_adj)
            node_pred = ypred[self.graph_idx, node_idx, :]
            res = nn.Softmax(dim=0)(node_pred)

            return res, adj_att


    def adj_feat_grad(self, node_idx, pred_label_node):
        self.model.zero_grad()
        self.adj.requires_grad = True
        self.x.requires_grad = True
        if self.adj.grad is not None:
            self.adj.grad.zero_()
            self.x.grad.zero_()
        if self.args.gpu:
            adj = self.adj.cuda()
            x = self.x.cuda()
            label = self.label.cuda()
        else:
            x, adj = self.x, self.adj
        ypred, _ = self.model(x, adj)
        if self.graph_mode:
            logit = nn.Softmax(dim=0)(ypred[0])
        else:
            logit = nn.Softmax(dim=0)(ypred[self.graph_idx, node_idx, :])
        logit = logit[pred_label_node]
        loss = -torch.log(logit)
        loss.backward()
        return self.adj.grad, self.x.grad

    def loss(self, pred, pred_label, node_idx, epoch):
        """
        Args:
            pred: prediction made by current model
            pred_label: the label predicted by the original model.
        """
        mi_obj = False
        if mi_obj:
            pred_loss = -torch.sum(pred * torch.log(pred))
        else:
            pred_label_node = pred_label if self.graph_mode else pred_label[node_idx]
            gt_label_node = self.label if self.graph_mode else self.label[0][node_idx]
            logit = pred[gt_label_node]
            pred_loss = -torch.log(logit)
        # size
        mask = self.mask
        if self.mask_act == "sigmoid":
            mask = torch.sigmoid(self.mask)
        elif self.mask_act == "ReLU":
            mask = nn.ReLU()(self.mask)
        size_loss = self.coeffs["size"] * (torch.sum(mask)-50)*10

        # pre_mask_sum = torch.sum(self.feat_mask)
        feat_mask = (
            torch.sigmoid(self.feat_mask) if self.use_sigmoid else self.feat_mask
        )
        feat_size_loss = self.coeffs["feat_size"] * torch.mean(feat_mask)

        # entropy
        mask_ent = -mask * torch.log(mask) - (1 - mask) * torch.log(1 - mask)
        mask_ent_loss = self.coeffs["ent"] * torch.mean(mask_ent)

        feat_mask_ent = - feat_mask \
                        * torch.log(feat_mask) \
                        - (1 - feat_mask) \
                        * torch.log(1 - feat_mask)

        feat_mask_ent_loss = self.coeffs["feat_ent"] * torch.mean(feat_mask_ent)

        # laplacian
        D = torch.diag(torch.sum(self.masked_adj[0], 0))
        m_adj = self.masked_adj if self.graph_mode else self.masked_adj[self.graph_idx]
        L = D - m_adj
        pred_label_t = torch.tensor(pred_label, dtype=torch.float)
        if self.args.gpu:
            pred_label_t = pred_label_t.cuda()
            L = L.cuda()
        if self.graph_mode:
            lap_loss = 0
        else:
            lap_loss = (self.coeffs["lap"]
                        * (pred_label_t @ L @ pred_label_t)
                        / self.adj.numel()
                        )

        # grad                              
        # adj
        # adj_grad, x_grad = self.adj_feat_grad(node_idx, pred_label_node)
        # adj_grad = adj_grad[self.graph_idx]
        # x_grad = x_grad[self.graph_idx]
        # if self.args.gpu:
        #    adj_grad = adj_grad.cuda()
        # grad_loss = self.coeffs['grad'] * -torch.mean(torch.abs(adj_grad) * mask)

        # feat
        # x_grad_sum = torch.sum(x_grad, 1)
        # grad_feat_loss = self.coeffs['featgrad'] * -torch.mean(x_grad_sum * mask)

        loss = pred_loss + size_loss + lap_loss + mask_ent_loss + feat_size_loss
        if self.writer is not None:
            self.writer.add_scalar("optimization/size_loss", size_loss, epoch)
            self.writer.add_scalar("optimization/feat_size_loss", feat_size_loss, epoch)
            self.writer.add_scalar("optimization/mask_ent_loss", mask_ent_loss, epoch)
            self.writer.add_scalar(
                "optimization/feat_mask_ent_loss", mask_ent_loss, epoch
            )
            # self.writer.add_scalar('optimization/grad_loss', grad_loss, epoch)
            self.writer.add_scalar("optimization/pred_loss", pred_loss, epoch)
            self.writer.add_scalar("optimization/lap_loss", lap_loss, epoch)
            self.writer.add_scalar("optimization/overall_loss", loss, epoch)
        return loss

    def log_mask(self, epoch):
        plt.switch_backend("agg")
        fig = plt.figure(figsize=(4, 3), dpi=400)
        plt.imshow(self.mask.cpu().detach().numpy(), cmap=plt.get_cmap("BuPu"))
        cbar = plt.colorbar()
        cbar.solids.set_edgecolor("face")

        plt.tight_layout()
        fig.canvas.draw()
        self.writer.add_image(
            "mask/mask", tensorboardX.utils.figure_to_image(fig), epoch
        )

        # fig = plt.figure(figsize=(4,3), dpi=400)
        # plt.imshow(self.feat_mask.cpu().detach().numpy()[:,np.newaxis], cmap=plt.get_cmap('BuPu'))
        # cbar = plt.colorbar()
        # cbar.solids.set_edgecolor("face")

        # plt.tight_layout()
        # fig.canvas.draw()
        # self.writer.add_image('mask/feat_mask', tensorboardX.utils.figure_to_image(fig), epoch)
        io_utils.log_matrix(
            self.writer, torch.sigmoid(self.feat_mask), "mask/feat_mask", epoch
        )

        fig = plt.figure(figsize=(4, 3), dpi=400)
        # use [0] to remove the batch dim
        plt.imshow(self.masked_adj[0].cpu().detach().numpy(), cmap=plt.get_cmap("BuPu"))
        cbar = plt.colorbar()
        cbar.solids.set_edgecolor("face")

        plt.tight_layout()
        fig.canvas.draw()
        self.writer.add_image(
            "mask/adj", tensorboardX.utils.figure_to_image(fig), epoch
        )

        if self.args.mask_bias:
            fig = plt.figure(figsize=(4, 3), dpi=400)
            # use [0] to remove the batch dim
            plt.imshow(self.mask_bias.cpu().detach().numpy(), cmap=plt.get_cmap("BuPu"))
            cbar = plt.colorbar()
            cbar.solids.set_edgecolor("face")

            plt.tight_layout()
            fig.canvas.draw()
            self.writer.add_image(
                "mask/bias", tensorboardX.utils.figure_to_image(fig), epoch
            )

    def log_adj_grad(self, node_idx, pred_label, epoch, label=None):
        log_adj = False

        if self.graph_mode:
            predicted_label = pred_label
            # adj_grad, x_grad = torch.abs(self.adj_feat_grad(node_idx, predicted_label)[0])[0]
            adj_grad, x_grad = self.adj_feat_grad(node_idx, predicted_label)
            adj_grad = torch.abs(adj_grad)[0]
            x_grad = torch.sum(x_grad[0], 0, keepdim=True).t()
        else:
            predicted_label = pred_label[node_idx]
            # adj_grad = torch.abs(self.adj_feat_grad(node_idx, predicted_label)[0])[self.graph_idx]
            adj_grad, x_grad = self.adj_feat_grad(node_idx, predicted_label)
            adj_grad = torch.abs(adj_grad)[self.graph_idx]
            x_grad = x_grad[self.graph_idx][node_idx][:, np.newaxis]
            # x_grad = torch.sum(x_grad[self.graph_idx], 0, keepdim=True).t()
        adj_grad = (adj_grad + adj_grad.t()) / 2
        adj_grad = (adj_grad * self.adj).squeeze()
        if log_adj:
            io_utils.log_matrix(self.writer, adj_grad, "grad/adj_masked", epoch)
            self.adj.requires_grad = False
            io_utils.log_matrix(self.writer, self.adj.squeeze(), "grad/adj_orig", epoch)

        masked_adj = self.masked_adj[0].cpu().detach().numpy()

        # only for graph mode since many node neighborhoods for syn tasks are relatively large for
        # visualization
        if self.graph_mode:
            G = io_utils.denoise_graph(
                masked_adj, node_idx, feat=self.x[0], threshold=None, max_component=False
            )
            io_utils.log_graph(
                self.writer,
                G,
                name="grad/graph_orig",
                epoch=epoch,
                identify_self=False,
                label_node_feat=True,
                nodecolor="feat",
                edge_vmax=None,
                args=self.args,
            )
        io_utils.log_matrix(self.writer, x_grad, "grad/feat", epoch)

        adj_grad = adj_grad.detach().numpy()
        if self.graph_mode:
            print("GRAPH model")
            G = io_utils.denoise_graph(
                adj_grad,
                node_idx,
                feat=self.x[0],
                threshold=0.0003,  # threshold_num=20,
                max_component=True,
            )
            io_utils.log_graph(
                self.writer,
                G,
                name="grad/graph",
                epoch=epoch,
                identify_self=False,
                label_node_feat=True,
                nodecolor="feat",
                edge_vmax=None,
                args=self.args,
            )
        else:
            # G = io_utils.denoise_graph(adj_grad, node_idx, label=label, threshold=0.5)
            G = io_utils.denoise_graph(adj_grad, node_idx, threshold_num=12)
            io_utils.log_graph(
                self.writer, G, name="grad/graph", epoch=epoch, args=self.args
            )

        # if graph attention, also visualize att

    def log_masked_adj(self, node_idx, epoch, name="mask/graph", label=None):
        # use [0] to remove the batch dim
        masked_adj = self.masked_adj[0].cpu().detach().numpy()
        if self.graph_mode:
            G = io_utils.denoise_graph(
                masked_adj,
                node_idx,
                feat=self.x[0],
                threshold=0.2,  # threshold_num=20,
                max_component=True,
            )
            io_utils.log_graph(
                self.writer,
                G,
                name=name,
                identify_self=False,
                nodecolor="feat",
                epoch=epoch,
                label_node_feat=True,
                edge_vmax=None,
                args=self.args,
            )
        else:
            G = io_utils.denoise_graph(
                masked_adj, node_idx, threshold_num=12, max_component=True
            )
            io_utils.log_graph(
                self.writer,
                G,
                name=name,
                identify_self=True,
                nodecolor="label",
                epoch=epoch,
                edge_vmax=None,
                args=self.args,
            )

