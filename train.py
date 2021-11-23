""" train.py

    Main interface to train the GNNs that will be later explained.
"""
import argparse
import os
import pickle
import random
import shutil
import time

import matplotlib
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

import networkx as nx
import numpy as np
import sklearn.metrics as metrics
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
import scipy.sparse as sp
from tensorboardX import SummaryWriter

import configs
import gengraph

import utils.math_utils as math_utils
import utils.io_utils as io_utils
import utils.parser_utils as parser_utils
import utils.train_utils as train_utils
import utils.featgen as featgen
import utils.graph_utils as graph_utils

import models
import warnings

warnings.filterwarnings("ignore")

#############################
#
# Prepare Data
#
#############################
def prepare_data(graphs, args, test_graphs=None, max_nodes=0):

    random.shuffle(graphs)
    if test_graphs is None:
        train_idx = int(len(graphs) * args.train_ratio)
        test_idx = int(len(graphs) * (1 - args.test_ratio))
        train_graphs = graphs[:train_idx]
        val_graphs = graphs[train_idx:test_idx]
        test_graphs = graphs[test_idx:]
    else:
        train_idx = int(len(graphs) * args.train_ratio)
        train_graphs = graphs[:train_idx]
        val_graphs = graph[train_idx:]
    print(
        "Num training graphs: ",
        len(train_graphs),
        "; Num validation graphs: ",
        len(val_graphs),
        "; Num testing graphs: ",
        len(test_graphs),
    )

    print("Number of graphs: ", len(graphs))
    print("Number of edges: ", sum([G.number_of_edges() for G in graphs]))
    print(
        "Max, avg, std of graph size: ",
        max([G.number_of_nodes() for G in graphs]),
        ", " "{0:.2f}".format(np.mean([G.number_of_nodes() for G in graphs])),
        ", " "{0:.2f}".format(np.std([G.number_of_nodes() for G in graphs])),
    )

    # minibatch
    dataset_sampler = graph_utils.GraphSampler(
        train_graphs,
        normalize=False,
        max_num_nodes=max_nodes,
        features=args.feature_type,
    )
    train_dataset_loader = torch.utils.data.DataLoader(
        dataset_sampler,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    dataset_sampler = graph_utils.GraphSampler(
        val_graphs,
        normalize=False,
        max_num_nodes=max_nodes,
        features=args.feature_type
    )
    val_dataset_loader = torch.utils.data.DataLoader(
        dataset_sampler,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    dataset_sampler = graph_utils.GraphSampler(
        test_graphs,
        normalize=False,
        max_num_nodes=max_nodes,
        features=args.feature_type,
    )
    test_dataset_loader = torch.utils.data.DataLoader(
        dataset_sampler,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    return (
        train_dataset_loader,
        val_dataset_loader,
        test_dataset_loader,
        dataset_sampler.max_num_nodes,
        dataset_sampler.feat_dim,
        dataset_sampler.assign_feat_dim,
    )


#############################
#
# Training
#
#############################
def train(
        dataset,
        model,
        args,
        same_feat=True,
        val_dataset=None,
        test_dataset=None,
        writer=None,
        mask_nodes=True,
):
    writer_batch_idx = [0, 3, 6, 9]

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=0.001
    )
    iter = 0
    best_val_result = {"epoch": 0, "loss": 0, "acc": 0}
    test_result = {"epoch": 0, "loss": 0, "acc": 0}
    train_accs = []
    train_epochs = []
    best_val_accs = []
    best_val_epochs = []
    test_accs = []
    test_epochs = []
    val_accs = []

    for epoch in range(args.num_epochs):
        begin_time = time.time()
        avg_loss = 0.0
        model.train()
        predictions = []
        print("Epoch: ", epoch)
        for batch_idx, data in enumerate(dataset):
            model.zero_grad()
            if batch_idx == 0:
                prev_adjs = data["adj"]
                prev_feats = data["feats"]
                prev_labels = data["label"]
                all_adjs = prev_adjs
                all_feats = prev_feats
                all_labels = prev_labels
            elif batch_idx < 20:
                prev_adjs = data["adj"]
                prev_feats = data["feats"]
                prev_labels = data["label"]
                all_adjs = torch.cat((all_adjs, prev_adjs), dim=0)
                all_feats = torch.cat((all_feats, prev_feats), dim=0)
                all_labels = torch.cat((all_labels, prev_labels), dim=0)
            adj = Variable(data["adj"].float(), requires_grad=False).cuda()
            h0 = Variable(data["feats"].float(), requires_grad=False).cuda()
            label = Variable(data["label"].long()).cuda()
            batch_num_nodes = data["num_nodes"].int().numpy() if mask_nodes else None
            assign_input = Variable(
                data["assign_feats"].float(), requires_grad=False
            ).cuda()

            ypred, att_adj = model(h0, adj, batch_num_nodes, assign_x=assign_input)
            if batch_idx < 5:
                predictions += ypred.cpu().detach().numpy().tolist()

            if not args.method == "soft-assign" or not args.linkpred:
                loss = model.loss(ypred, label)
            else:
                loss = model.loss(ypred, label, adj, batch_num_nodes)
            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), args.clip)
            optimizer.step()
            iter += 1
            avg_loss += loss

        avg_loss /= batch_idx + 1
        elapsed = time.time() - begin_time
        if writer is not None:
            writer.add_scalar("loss/avg_loss", avg_loss, epoch)
            if args.linkpred:
                writer.add_scalar("loss/linkpred_loss", model.link_loss, epoch)
        print("Avg loss: ", avg_loss, "; epoch time: ", elapsed)
        result = evaluate(dataset, model, args, name="Train", max_num_examples=100)
        train_accs.append(result["acc"])
        train_epochs.append(epoch)
        if val_dataset is not None:
            val_result = evaluate(val_dataset, model, args, name="Validation")
            val_accs.append(val_result["acc"])
        if val_result["acc"] > best_val_result["acc"] - 1e-7:
            best_val_result["acc"] = val_result["acc"]
            best_val_result["epoch"] = epoch
            best_val_result["loss"] = avg_loss
        if test_dataset is not None:
            test_result = evaluate(test_dataset, model, args, name="Test")
            test_result["epoch"] = epoch
        if writer is not None:
            writer.add_scalar("acc/train_acc", result["acc"], epoch)
            writer.add_scalar("acc/val_acc", val_result["acc"], epoch)
            writer.add_scalar("loss/best_val_loss", best_val_result["loss"], epoch)
            if test_dataset is not None:
                writer.add_scalar("acc/test_acc", test_result["acc"], epoch)

        print("Best val result: ", best_val_result)
        best_val_epochs.append(best_val_result["epoch"])
        best_val_accs.append(best_val_result["acc"])
        if test_dataset is not None:
            print("Test result: ", test_result)
            test_epochs.append(test_result["epoch"])
            test_accs.append(test_result["acc"])

    matplotlib.style.use("seaborn")
    plt.switch_backend("agg")
    plt.figure()
    plt.plot(train_epochs, math_utils.exp_moving_avg(train_accs, 0.85), "-", lw=1)
    if test_dataset is not None:
        plt.plot(best_val_epochs, best_val_accs, "bo", test_epochs, test_accs, "go")
        plt.legend(["train", "val", "test"])
    else:
        plt.plot(best_val_epochs, best_val_accs, "bo")
        plt.legend(["train", "val"])
    plt.savefig(io_utils.gen_train_plt_name(args), dpi=600)
    plt.close()
    matplotlib.style.use("default")

    print(all_adjs.shape, all_feats.shape, all_labels.shape)

    cg_data = {
        "adj": all_adjs,
        "feat": all_feats,
        "label": all_labels,
        "pred": np.expand_dims(predictions, axis=0),
        "train_idx": list(range(len(dataset))),
    }
    io_utils.save_checkpoint(model, optimizer, args, num_epochs=-1, cg_dict=cg_data)
    return model, val_accs


def fair_metric(output,x,labels,idx):
    sens=x[:,-1].squeeze().detach()
    val_y = labels.cpu().numpy()
    idx_s0 = sens.cpu().numpy()[idx]==0
    idx_s1 = sens.cpu().numpy()[idx]==1

    idx_s0_y1 = np.bitwise_and(idx_s0,val_y==1)
    idx_s1_y1 = np.bitwise_and(idx_s1,val_y==1)

    #pred_y = (output[idx].squeeze()>0).type_as(labels).cpu().numpy()
    pred_y=torch.argmax(output,-1).type_as(labels).cpu().numpy()

    parity = abs(sum(pred_y[idx_s0])/sum(idx_s0)-sum(pred_y[idx_s1])/sum(idx_s1))
    equality = abs(sum(pred_y[idx_s0_y1])/sum(idx_s0_y1)-sum(pred_y[idx_s1_y1])/sum(idx_s1_y1))

    return [parity,equality]

def train_node_classifier(G, labels, model, args, writer=None):
    # train/test split only for nodes
    num_nodes = G.number_of_nodes()
    num_train = int(num_nodes * args.train_ratio)
    idx = [i for i in range(num_nodes)]

    np.random.shuffle(idx)
    train_idx = idx[:num_train]
    test_idx = idx[num_train:]

    data = gengraph.preprocess_input_graph(G, labels)
    labels_train = torch.tensor(data["labels"][:, train_idx], dtype=torch.long)

    print(data['feat'].shape)

    def feature_norm(features):
        min_values = features.min(axis=0,keepdims=True)
        max_values = features.max(axis=0,keepdims=True)
        return 2*(features - min_values)/(max_values-min_values) - 1


    #data['feat']/=data['feat'].max(1,keepdims=True)
    #data['feat']=np.random.randn(data['feat'].shape[0],data['feat'].shape[1],data['feat'].shape[2])
    data['feat'][0][:,:-1]=feature_norm(data['feat'][0][:,:-1])

    print(data['feat'])

    data['adj'][0]=data['adj'][0]+np.identity(data['adj'][0].shape[-1])


    adj = torch.tensor(data["adj"], dtype=torch.float)



    x = torch.tensor(data["feat"], requires_grad=True, dtype=torch.float)
    scheduler, optimizer = train_utils.build_optimizer(
        args, model.parameters(), weight_decay=args.weight_decay
    )


    test_acc=0
    model.train()
    ypred = None
    for epoch in range(args.num_epochs):
        begin_time = time.time()
        model.zero_grad()

        if args.gpu:
            ypred, adj_att = model(x.cuda(), adj.cuda())
        else:
            ypred, adj_att = model(x, adj)






        ypred_train = ypred[:, train_idx, :]
        if args.gpu:
            loss = model.loss(ypred_train, labels_train.cuda())
        else:
            loss = model.loss(ypred_train, labels_train)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        optimizer.step()
        #for param_group in optimizer.param_groups:
        #    print(param_group["lr"])
        elapsed = time.time() - begin_time

        result_train, result_test = evaluate_node(
            ypred.cpu(), data["labels"], train_idx, test_idx
        )
        if writer is not None:
            writer.add_scalar("loss/avg_loss", loss, epoch)
            writer.add_scalars(
                "prec",
                {"train": result_train["prec"], "test": result_test["prec"]},
                epoch,
            )
            writer.add_scalars(
                "recall",
                {"train": result_train["recall"], "test": result_test["recall"]},
                epoch,
            )
            writer.add_scalars(
                "acc", {"train": result_train["acc"], "test": result_test["acc"]}, epoch
            )

        if epoch % 50 == 0:
            print(
                "epoch: ",
                epoch,
                "; loss:{:.4f} ".format(
                    loss.item()),
                "; train_acc: ",
                result_train["acc"],
                "; test_acc: ",
                result_test["acc"],
                "; train_prec: ",
                result_train["prec"],
                "; test_prec: ",
                result_test["prec"],
                "; epoch time: ",
                "{0:0.2f}".format(elapsed),
            )
        if result_test["acc"]>test_acc:
            test_acc=result_test["acc"]

        if scheduler is not None:
            scheduler.step()
    print(result_train["conf_mat"])
    print(result_test["conf_mat"])
    a=result_test["conf_mat"]
    print('f1',a[0,0]/(a[0,0]+0.5*a[0,1]+0.5*a[1,0]))
    print('f1',2/(1/result_test['prec']+1/result_test['recall']))

    print(fair_metric(ypred_train.squeeze(),x[0],labels_train[0],train_idx),test_acc)


    # computation graph
    model.eval()
    if args.gpu:
        ypred, _ = model(x.cuda(), adj.cuda())
    else:
        ypred, _ = model(x, adj)
    cg_data = {
        "adj": data["adj"],
        "feat": data["feat"],
        "label": data["labels"],
        "pred": ypred.cpu().detach().numpy(),
        "train_idx": train_idx,
    }
    # import pdb
    # pdb.set_trace()
    io_utils.save_checkpoint(model, optimizer, args, num_epochs=-1, cg_dict=cg_data)

    return [fair_metric(ypred_train.squeeze(),x[0],labels_train[0],train_idx)+[test_acc.item()]]



#############################
#
# Evaluate Trained Model
#
#############################
def evaluate(dataset, model, args, name="Validation", max_num_examples=None):
    model.eval()

    labels = []
    preds = []
    for batch_idx, data in enumerate(dataset):
        adj = Variable(data["adj"].float(), requires_grad=False).cuda()
        h0 = Variable(data["feats"].float()).cuda()
        labels.append(data["label"].long().numpy())
        batch_num_nodes = data["num_nodes"].int().numpy()
        assign_input = Variable(
            data["assign_feats"].float(), requires_grad=False
        ).cuda()

        ypred, att_adj = model(h0, adj, batch_num_nodes, assign_x=assign_input)
        _, indices = torch.max(ypred, 1)
        preds.append(indices.cpu().data.numpy())

        if max_num_examples is not None:
            if (batch_idx + 1) * args.batch_size > max_num_examples:
                break

    labels = np.hstack(labels)
    preds = np.hstack(preds)

    result = {
        "prec": metrics.precision_score(labels, preds, average="macro"),
        "recall": metrics.recall_score(labels, preds, average="macro"),
        "acc": metrics.accuracy_score(labels, preds),
    }
    print(name, " accuracy:", result["acc"])
    return result


def evaluate_node(ypred, labels, train_idx, test_idx):
    _, pred_labels = torch.max(ypred, 2)
    pred_labels = pred_labels.numpy()

    pred_train = np.ravel(pred_labels[:, train_idx])
    pred_test = np.ravel(pred_labels[:, test_idx])
    labels_train = np.ravel(labels[:, train_idx])
    labels_test = np.ravel(labels[:, test_idx])

    result_train = {
        "prec": metrics.precision_score(labels_train, pred_train, average="macro"),
        "recall": metrics.recall_score(labels_train, pred_train, average="macro"),
        "acc": metrics.accuracy_score(labels_train, pred_train),
        "conf_mat": metrics.confusion_matrix(labels_train, pred_train),
    }
    result_test = {
        "prec": metrics.precision_score(labels_test, pred_test, average="macro"),
        "recall": metrics.recall_score(labels_test, pred_test, average="macro"),
        "acc": metrics.accuracy_score(labels_test, pred_test),
        "conf_mat": metrics.confusion_matrix(labels_test, pred_test),
    }
    return result_train, result_test



#############################
#
# Run Experiments
#
#############################
def ppi_essential_task(args, writer=None):
    feat_file = "G-MtfPathways_gene-motifs.csv"
    # G = io_utils.read_biosnap('data/ppi_essential', 'PP-Pathways_ppi.csv', 'G-HumanEssential.tsv',
    #        feat_file=feat_file)
    G = io_utils.read_biosnap(
        "data/ppi_essential",
        "hi-union-ppi.tsv",
        "G-HumanEssential.tsv",
        feat_file=feat_file,
    )
    labels = np.array([G.nodes[u]["label"] for u in G.nodes()])
    num_classes = max(labels) + 1
    input_dim = G.nodes[next(iter(G.nodes()))]["feat"].shape[0]

    if args.method == "attn":
        print("Method: attn")
    else:
        print("Method:", args.method)
        args.loss_weight = torch.tensor([1, 5.0], dtype=torch.float).cuda()
        model = models.GcnEncoderNode(
            input_dim,
            args.hidden_dim,
            args.output_dim,
            num_classes,
            args.num_gc_layers,
            bn=args.bn,
            args=args,
        )
        if args.gpu:
            model = model.cuda()

    train_node_classifier(G, labels, model, args, writer=writer)

def load_credit(dataset, sens_attr="Age", predict_attr="NoDefaultNextMonth", path="./credit/", label_number=1000):
    from scipy.spatial import distance_matrix
    def build_relationship(x, thresh=0.25):
        df_euclid = pd.DataFrame(1 / (1 + distance_matrix(x.T.T, x.T.T)), columns=x.T.columns, index=x.T.columns)
        df_euclid = df_euclid.to_numpy()
        idx_map = []
        for ind in range(df_euclid.shape[0]):
            max_sim = np.sort(df_euclid[ind, :])[-2]
            neig_id = np.where(df_euclid[ind, :] > thresh*max_sim)[0]
            import random
            random.seed(912)
            random.shuffle(neig_id)
            for neig in neig_id:
                if neig != ind:
                    idx_map.append([ind, neig])
        # print('building edge relationship complete')
        idx_map =  np.array(idx_map)

        return idx_map


    # print('Loading {} dataset from {}'.format(dataset, path))
    idx_features_labels = pd.read_csv(os.path.join(path,"{}.csv".format(dataset)))
    header = list(idx_features_labels.columns)
    header.remove(predict_attr)
    header.remove('Single')

    ##################################
    header.remove(sens_attr)
    header.append(sens_attr)
    ##################################



    # build relationship
    if os.path.exists(f'{path}/{dataset}_edges.txt'):
        edges_unordered = np.genfromtxt(f'{path}/{dataset}_edges.txt').astype('int')
    else:
        edges_unordered = build_relationship(idx_features_labels[header], thresh=0.7)
        np.savetxt(f'{path}/{dataset}_edges.txt', edges_unordered)

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels[predict_attr].values
    idx = np.arange(features.shape[0])
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=int).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    import random
    random.seed(20)
    label_idx_0 = np.where(labels==0)[0]
    label_idx_1 = np.where(labels==1)[0]
    random.shuffle(label_idx_0)
    random.shuffle(label_idx_1)

    idx_train = np.append(label_idx_0[:min(int(0.5 * len(label_idx_0)), label_number//2)], label_idx_1[:min(int(0.5 * len(label_idx_1)), label_number//2)])
    idx_val = np.append(label_idx_0[int(0.5 * len(label_idx_0)):int(0.75 * len(label_idx_0))], label_idx_1[int(0.5 * len(label_idx_1)):int(0.75 * len(label_idx_1))])
    idx_test = np.append(label_idx_0[int(0.75 * len(label_idx_0)):], label_idx_1[int(0.75 * len(label_idx_1)):])

    sens = idx_features_labels[sens_attr].values.astype(int)
    sens = torch.FloatTensor(sens)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, edges, sens


def load_bail(dataset, sens_attr="WHITE", predict_attr="RECID", path="./bail/", label_number=1000):
    # print('Loading {} dataset from {}'.format(dataset, path))
    idx_features_labels = pd.read_csv(os.path.join(path,"{}.csv".format(dataset)))
    header = list(idx_features_labels.columns)
    header.remove(predict_attr)

    ##################################
    header.remove(sens_attr)
    header.append(sens_attr)
    ##################################



    # build relationship
    if os.path.exists(f'{path}/{dataset}_edges.txt'):
        edges_unordered = np.genfromtxt(f'{path}/{dataset}_edges.txt').astype('int')
    else:
        edges_unordered = build_relationship(idx_features_labels[header], thresh=0.6)
        np.savetxt(f'{path}/{dataset}_edges.txt', edges_unordered)

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels[predict_attr].values

    idx = np.arange(features.shape[0])
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=int).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # features = normalize(features)
    adj = adj + sp.eye(adj.shape[0])

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    import random
    random.seed(20)
    label_idx_0 = np.where(labels==0)[0]
    label_idx_1 = np.where(labels==1)[0]
    random.shuffle(label_idx_0)
    random.shuffle(label_idx_1)
    idx_train = np.append(label_idx_0[:min(int(0.5 * len(label_idx_0)), label_number//2)], label_idx_1[:min(int(0.5 * len(label_idx_1)), label_number//2)])
    idx_val = np.append(label_idx_0[int(0.5 * len(label_idx_0)):int(0.75 * len(label_idx_0))], label_idx_1[int(0.5 * len(label_idx_1)):int(0.75 * len(label_idx_1))])
    idx_test = np.append(label_idx_0[int(0.75 * len(label_idx_0)):], label_idx_1[int(0.75 * len(label_idx_1)):])

    sens = idx_features_labels[sens_attr].values.astype(int)
    sens = torch.FloatTensor(sens)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, edges, sens








def load_german(dataset, sens_attr="Gender", predict_attr="GoodCustomer", path="./german/", label_number=1000):
    # print('Loading {} dataset from {}'.format(dataset, path))
    idx_features_labels = pd.read_csv(os.path.join(path,"{}.csv".format(dataset)))
    header = list(idx_features_labels.columns)
    header.remove(predict_attr)
    header.remove('OtherLoansAtStore')
    header.remove('PurposeOfLoan')


    ##################################
    header.remove('Gender')
    header.append('Gender')
    ##################################

    # Sensitive Attribute
    idx_features_labels['Gender'][idx_features_labels['Gender'] == 'Female'] = 1
    idx_features_labels['Gender'][idx_features_labels['Gender'] == 'Male'] = 0

    if os.path.exists(f'{path}/{dataset}_edges.txt'):
        edges_unordered = np.genfromtxt(f'{path}/{dataset}_edges.txt').astype('int')
    else:
        edges_unordered = build_relationship(idx_features_labels[header], thresh=0.8)
        np.savetxt(f'{path}/{dataset}_edges.txt', edges_unordered)

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels[predict_attr].values
    labels[labels == -1] = 0

    idx = np.arange(features.shape[0])
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=int).reshape(edges_unordered.shape)


    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    adj = adj + sp.eye(adj.shape[0])

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    import random
    random.seed(20)
    label_idx_0 = np.where(labels==0)[0]
    label_idx_1 = np.where(labels==1)[0]
    random.shuffle(label_idx_0)
    random.shuffle(label_idx_1)

    idx_train = np.append(label_idx_0[:min(int(0.5 * len(label_idx_0)), label_number//2)], label_idx_1[:min(int(0.5 * len(label_idx_1)), label_number//2)])
    idx_val = np.append(label_idx_0[int(0.5 * len(label_idx_0)):int(0.75 * len(label_idx_0))], label_idx_1[int(0.5 * len(label_idx_1)):int(0.75 * len(label_idx_1))])
    idx_test = np.append(label_idx_0[int(0.75 * len(label_idx_0)):], label_idx_1[int(0.75 * len(label_idx_1)):])

    sens = idx_features_labels[sens_attr].values.astype(int)
    sens = torch.FloatTensor(sens)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, edges, sens

def task_real_data(args,writer=None):


    label_number = 100

    if args.dataset=='german':
        adj, features, labels, edges, sens = load_german('german', label_number=label_number)
    elif args.dataset=='credit':
        adj, features, labels, edges, sens = load_credit('credit',label_number=label_number)
    elif args.dataset=='bail':
        adj, features, labels, edges, sens = load_bail('bail',label_number=label_number)



    #G = nx.Graph()

    G=nx.from_numpy_matrix(adj.todense())


    #G.add_nodes_from(list(range(1000)))

    #G.add_edges_from([edges[i].tolist() for i in range(edges.shape[0])])

    feat_dict = {i:{'feat': np.array(features[i], dtype=np.float32)} for i in G.nodes()}



    print(len(G.edges()))

    show_graph=False
    if show_graph:
        select=200
        node_num=200
        node_list=list(range(1000))
        total_unfair_edge=set()
        count=0
        for j,i in enumerate(node_list[:node_num]):

            try:
                neighbors=np.load('./log/{}_explain_neighbors_both5_newnew/neighbors_{}_base_h20_o20_explainnode_idx_{}graph_idx_-1.npy'.format(
                    show_graph,show_graph,i))
            except:
                continue

            count+=1
            if count>100:
                break


            new_idx=neighbors[-1]
            neighbors=neighbors[:-1]

            if 'ori' not in show_graph:
                masks=np.load('./log/{}_explain_neighbors_both5_newnew/masked_adj_{}_base_h20_o20_explainnode_idx_{}graph_idx_-1KLTrue.npy'.format(
                    show_graph,show_graph,i))

                ori_mask=masks[0]
                wass_mask=masks[1]
                unwass_mask=masks[2]
            else:
                masks=np.load('./log/{}_explain_neighbors_both5_newnew/masked_adj_{}_base_h20_o20_explainnode_idx_{}graph_idx_-1KLTrue.npy'.format(
                    show_graph,show_graph,i))
                wass_mask=masks[0]
                masks=np.load('./log/{}_explain_neighbors_both5_newnew/masked_adj_{}_base_h20_o20_explainnode_idx_{}graph_idx_-1KLTrue.npy'.format(
                    show_graph,show_graph,i))
                unwass_mask=masks[0]

            wass_mask=np.triu(wass_mask)
            unwass_mask=np.triu(unwass_mask)


            #print(wass_mask[:10,:10])
            #print(adj.todense()[neighbors[:10],:][:,neighbors[:10]])
            #print(adj.todense()[:10,:10])



            num=ori_mask.flatten().shape[0]

            index=np.argsort(ori_mask.flatten())[-select:]
            ori_mask_new=np.zeros(ori_mask.shape).flatten()
            ori_mask_new[index]=1



            index=np.argsort(wass_mask.flatten())[-select:]
            wass_mask_new=np.zeros(wass_mask.shape).flatten()
            wass_mask_new[index]=1

            index=np.argsort(unwass_mask.flatten())[-select:]
            unwass_mask_new=np.zeros(unwass_mask.shape).flatten()
            unwass_mask_new[index]=1

            ori_mask=np.argwhere(ori_mask_new.reshape(ori_mask.shape)>0).tolist()
            wass_mask=np.argwhere(wass_mask_new.reshape(wass_mask.shape)>0).tolist()
            unwass_mask=np.argwhere(unwass_mask_new.reshape(unwass_mask.shape)>0).tolist()




            temp=set()
            for one in unwass_mask:
                a=(neighbors[one[0]],neighbors[one[1]])
                temp.add(a)

            temp_wass=set()
            for one in wass_mask:
                a=(neighbors[one[0]],neighbors[one[1]])
                temp_wass.add(a)

            unwass_mask=set(temp).difference(set(temp_wass))



            total_unfair_edge.update(unwass_mask)


        print('unfair edge',len(total_unfair_edge))

        #print(total_unfair_edge)
        #print(list(G.edges())[:2000])

        for edge in total_unfair_edge:
            try:

                G.remove_edge(edge[0],edge[1])
            except:
                continue



    print(len(G.edges()))

    nx.set_node_attributes(G, feat_dict)


    num_classes = 2
    args.input_dim=features.shape[-1]

    print(features.shape)



    model = models.GcnEncoderNode(
        args.input_dim,
        args.hidden_dim,
        args.output_dim,
        num_classes,
        args.num_gc_layers,
        bn=args.bn,
        args=args,)

    if args.gpu:
        model = model.cuda()



    return train_node_classifier(G, labels, model, args, writer=writer)


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot
def dif(a,b):
    if a>500 and b<500:
        return True
    if a<500 and b>500:
        return True
    return False
def load_synthetic():
    # ************************ Feature Matrix **********************************
    import numpy as np
    import scipy.sparse as sp
    import torch
    import os
    import pandas as pd
    # import dgl
    import networkx as nx
    import random
    from math import sqrt
    import networkx as nx
    import matplotlib.pyplot as plt
    import random
    random.seed(20)

    num, dim = 500, 2
    confusing_dim = 10-dim
    np.random.seed(0)
    # male = np.random.uniform(0, 1, (num, dim))
    # female = np.random.uniform(0, 1, (num, dim))
    male = np.random.randn(num, dim)
    female = np.random.randn(num, dim)

    male_sum = np.sum(male, axis=1)
    male_index = np.argsort(male_sum)  # 小的在前面
    male = male[male_index[::-1], :]  # male的二维数组，sum大的在前面，小的在后面

    female_sum = np.sum(female, axis=1)
    female_index = np.argsort(female_sum)  # 小的在前面
    female = female[female_index[::-1], :]  # female的二维数组，sum大的在前面，小的在后面

    X = np.concatenate((male, female), axis=0)
    X = np.concatenate((X, np.random.uniform(0, 1, (num * 2, confusing_dim))), axis=1)

    # to_be_deleted = np.concatenate((np.ones((num, 1)), np.zeros((num, 1))), axis=0)  # 这是相反的原因
    to_be_deleted = np.concatenate((np.zeros((num, 1)), np.ones((num, 1))), axis=0)
    X = np.concatenate((X, to_be_deleted), axis=1)

    print("Shape of feature matrix X: ", X.shape)
    # print(X)

    # plt.scatter(male[:, 0], male[:, 1])
    # plt.scatter(female[:, 0], female[:, 1])
    # plt.show()

    # **************************************************************************

    # ************************ Adjacency Matrix & Graph **********************************

    A = np.zeros((num * 2, num * 2))
    p_within_male = 0.05
    p_within_female = 0.05
    p_within_ordinary = 0.01 *5
    p_between_male_ordinary = 0.0002 *25
    p_between_female_ordinary = 0.0002 *25

    # p_within_male = 0.05
    # p_within_female = 0.05
    # p_within_ordinary = 0.005
    # p_between_male_ordinary = 0.0002
    # p_between_female_ordinary = 0.0002

    the_threshold = 180

    threshold_number = the_threshold * the_threshold

    A_male = np.random.binomial(1, p_within_male, threshold_number)
    A_male = A_male.reshape(int(sqrt(threshold_number)), int(sqrt(threshold_number)))

    A_female = np.random.binomial(1, p_within_female, threshold_number)
    A_female = A_female.reshape(int(sqrt(threshold_number)), int(sqrt(threshold_number)))

    A_ordinary = np.random.binomial(1, p_within_ordinary, int(num * 2 - 2 * sqrt(threshold_number)) ** 2)
    A_ordinary = A_ordinary.reshape(int(num * 2 - 2 * sqrt(threshold_number)),
                                    int(num * 2 - 2 * sqrt(threshold_number)))

    A_between_male_ordinary = np.zeros((int(sqrt(threshold_number)), int((num * 2 - 2 * sqrt(threshold_number)))))

    while np.where(A_between_male_ordinary == 1)[0].shape[0] == 0:
        A_between_male_ordinary = np.random.binomial(1, p_between_male_ordinary,
                                                     int((num * 2 - 2 * sqrt(threshold_number)) * sqrt(
                                                         threshold_number)))
        A_between_male_ordinary = A_between_male_ordinary.reshape(int(sqrt(threshold_number)),
                                                                  int((num * 2 - 2 * sqrt(threshold_number))))

    A_between_female_ordinary = np.zeros((int(sqrt(threshold_number)), int((num * 2 - 2 * sqrt(threshold_number)))))

    while np.where(A_between_female_ordinary == 1)[0].shape[0] == 0:
        A_between_female_ordinary = np.random.binomial(1, p_between_female_ordinary,
                                                       int((num * 2 - 2 * sqrt(threshold_number)) * sqrt(
                                                           threshold_number)))
        A_between_female_ordinary = A_between_female_ordinary.reshape(int(sqrt(threshold_number)),
                                                                      int((num * 2 - 2 * sqrt(threshold_number))))

    print(1)

    A[:the_threshold, :the_threshold] = A_male
    A[-the_threshold:, -the_threshold:] = A_female
    A[the_threshold:(num * 2 - the_threshold), the_threshold:(num * 2 - the_threshold)] = A_ordinary
    A[:the_threshold, the_threshold:(num * 2 - the_threshold)] = A_between_male_ordinary
    A[the_threshold:(num * 2 - the_threshold), -the_threshold:] = A_between_female_ordinary.T

    print(2)

    # Make it symmetircal
    A = np.triu(A)

    # A += A.T - np.diag(A.diagonal())  # diagonal != 0
    A += A.T - 2 * np.diag(A.diagonal())  # diagonal = 0

    print(3)

    A = A + np.diag([1] * 2 * num)
    # D_sqrt = np.diag(1 / np.sqrt(np.sum(A, axis=1)))
    # D_sqrt[np.isnan(D_sqrt)] = 0
    # A = D_sqrt.dot(A).dot(D_sqrt)




    feature = X


    # final_sum = np.sum(feature[:, 3:5] + 0.0 * np.random.randn(2*num, dim), axis=1)
    final_sum = np.sum(feature[:, 2:4] + 0.01 * np.random.randn(2*num, dim), axis=1)

    # print(np.random.randn(2*num, 1).shape)
    # print((feature[:, 2] + 0.05 * np.random.randn(2*num, 1)).shape)


    # adj
    adj = sp.coo_matrix(A, dtype=np.float32)
    # adj = sparse_mx_to_torch_sparse_tensor(adj)

    # feature
    features = sp.csr_matrix(feature, dtype=np.float32)
    # features = normalize(features)
    features = torch.FloatTensor(np.array(features.todense()))




    # train, val, test
    #node_perm = np.random.permutation(A.shape[0])
    node_perm=np.random.permutation(np.array(list(range(A.shape[0]))))
    # print(node_perm)
    num_train = int(0.5 * A.shape[0])
    num_val = int(0.4 * A.shape[0])
    idx_train = node_perm[:num_train]
    idx_val = node_perm[num_train:num_train + num_val]
    idx_test = node_perm[num_train + num_val:]

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    # labels

    # labels = np.around(final_sum)
    labels = np.floor(final_sum)
    print(labels)
    labels[labels >= 1] = 1
    labels[labels <= 0] = 0
    #labels+=1
    print(set(labels))
    print(labels)

    labels = encode_onehot(labels)
    labels = torch.LongTensor(np.where(labels)[1])
    sens = torch.LongTensor(np.array([0] * num + [1] * num))





    return adj, features, labels, idx_train, idx_val, idx_test, node_perm, sens

def task_synthetic(args,writer=None):
    adj, features, labels, idx_train, idx_val, idx_test, node_perm, sens=load_synthetic()



    G = nx.Graph()

    G.add_edges_from([[adj.row[i],adj.col[i]] for i in range(len(adj.row))])

    feat_dict = {i:{'feat': np.array(features[i], dtype=np.float32)} for i in G.nodes()}

    nx.set_node_attributes(G, feat_dict)

    idx_male=set(list(range(180)))
    idx_ordinary=set(list(range(180,1000-180)))
    idx_female=set(list(range(1000-180,1000)))
    count_same_gender=0
    count_cross_gender=0
    count_cross_gender_community=0
    count_cross_gender_ordinary_community=0

    for one in G.edges():
        a=one[0]
        b=one[1]

        if dif(a,b):
            count_cross_gender+=1
        else:
            count_same_gender+=1
        if dif(a,b) and a in idx_ordinary and b not in idx_ordinary:
            count_cross_gender_community+=1
        if dif(a,b) and b in idx_ordinary and a not in idx_ordinary:
            count_cross_gender_community+=1
        if dif(a,b) and a in idx_ordinary and b in idx_ordinary:
            count_cross_gender_ordinary_community+=1

    print('{:.2f}%'.format(count_cross_gender_ordinary_community*100/(count_same_gender+count_cross_gender)))
    print('{:.2f}%'.format(count_cross_gender_community*100/(count_same_gender+count_cross_gender)))
    print('{:.2f}%'.format(count_same_gender*100/(count_same_gender+count_cross_gender)))

    print(G.nodes())
    print(len(G.nodes()))






    num_classes = 2
    args.input_dim=11

    model = models.GcnEncoderNode(
        args.input_dim,
        args.hidden_dim,
        args.output_dim,
        num_classes,
        args.num_gc_layers,
        bn=args.bn,
        args=args,)

    if args.gpu:
        model = model.cuda()

    train_node_classifier(G, labels, model, args, writer=writer)




def load_synthetic2():
    # ************************ Feature Matrix **********************************
    import random
    random.seed(20)

    num, dim = 500, 2
    confusing_dim = 8
    np.random.seed(0)
    # male = np.random.uniform(0, 1, (num, dim))
    # female = np.random.uniform(0, 1, (num, dim))
    male = np.random.randn(num, dim) + 1.5
    female = np.random.randn(num, dim) - 1.5

    male_sum = np.sum(male, axis=1)
    male_index = np.argsort(male_sum)  # 小的在前面
    male = male[male_index[::-1], :]  # male的二维数组，sum大的在前面，小的在后面

    female_sum = np.sum(female, axis=1)
    female_index = np.argsort(female_sum)  # 小的在前面
    female = female[female_index[::-1], :]  # female的二维数组，sum大的在前面，小的在后面

    X = np.concatenate((male, female), axis=0)
    X = np.concatenate((X, np.random.uniform(0, 1, (num * 2, confusing_dim))), axis=1)

    # to_be_deleted = np.concatenate((np.ones((num, 1)), np.zeros((num, 1))), axis=0)  # 这是相反的原因
    to_be_deleted = np.concatenate((np.zeros((num, 1)), np.ones((num, 1))), axis=0)
    X = np.concatenate((X, to_be_deleted), axis=1)

    print("Shape of feature matrix X: ", X.shape)
    # print(X)

    # plt.scatter(male[:, 0], male[:, 1])
    # plt.scatter(female[:, 0], female[:, 1])
    # plt.show()

    # **************************************************************************

    # ************************ Adjacency Matrix & Graph **********************************


    p_link = 0.002 *20
    A = np.random.binomial(1, p_link, num*dim*num*dim).reshape(num*dim, num*dim)


    print(2)

    # Make it symmetircal
    A = np.triu(A)

    # A += A.T - np.diag(A.diagonal())  # diagonal != 0
    A += A.T - 2 * np.diag(A.diagonal())  # diagonal = 0

    print(3)

    A = A + np.diag([1] * 2 * num)
    # D_sqrt = np.diag(1 / np.sqrt(np.sum(A, axis=1)))
    # D_sqrt[np.isnan(D_sqrt)] = 0
    # A = D_sqrt.dot(A).dot(D_sqrt)



    # ************************ Graph visualization **********************************

    print(A.shape)

    G = nx.from_numpy_matrix(A)

    color_map = []
    for node in G:
        if node < num:
            color_map.append('blue')
        else:
            color_map.append('red')




    feature = X

    print(feature[:, 2])


    # final_sum = np.sum(feature[:, 3:5] + 0.0 * np.random.randn(2*num, dim), axis=1)
    final_sum = np.sum(feature[:, 0:2] + 0.01 * np.random.randn(2*num, dim), axis=1)

    # print(np.random.randn(2*num, 1).shape)
    # print((feature[:, 2] + 0.05 * np.random.randn(2*num, 1)).shape)




    # adj
    adj = sp.coo_matrix(A, dtype=np.float32)
    # adj = sparse_mx_to_torch_sparse_tensor(adj)



    # feature
    features = sp.csr_matrix(feature, dtype=np.float32)
    # features = normalize(features)
    features = torch.FloatTensor(np.array(features.todense()))




    # train, val, test
    #node_perm = np.random.permutation(A.shape[0])
    node_perm=np.random.permutation(np.array(list(range(A.shape[0]))))
    # print(node_perm)
    num_train = int(0.5 * A.shape[0])
    num_val = int(0.4 * A.shape[0])
    idx_train = node_perm[:num_train]
    idx_val = node_perm[num_train:num_train + num_val]
    idx_test = node_perm[num_train + num_val:]

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    # labels

    # labels = np.around(final_sum)
    labels = np.floor(final_sum)
    print(labels)
    labels[labels >= 1] = 1
    labels[labels <= 0] = 0
    print(set(labels))
    print(labels)

    labels = encode_onehot(labels)
    labels = torch.LongTensor(np.where(labels)[1])
    sens = torch.LongTensor(np.array([0] * num + [1] * num))




    return adj, features, labels, idx_train, idx_val, idx_test, node_perm, sens


def task_synthetic2(args,writer=None):
    adj, features, labels, idx_train, idx_val, idx_test, node_perm, sens=load_synthetic2()


    G = nx.Graph()

    G.add_edges_from([[adj.row[i],adj.col[i]] for i in range(len(adj.row))])

    feat_dict = {i:{'feat': np.array(features[i], dtype=np.float32)} for i in G.nodes()}

    nx.set_node_attributes(G, feat_dict)


    print(G.nodes())
    print(len(G.nodes()))

    num_classes = 2
    args.input_dim=11

    model = models.GcnEncoderNode(
        args.input_dim,
        args.hidden_dim,
        args.output_dim,
        num_classes,
        args.num_gc_layers,
        bn=args.bn,
        args=args,)

    if args.gpu:
        model = model.cuda()

    train_node_classifier(G, labels, model, args, writer=writer)


def syn_task1(args, writer=None):
    # data
    G, labels, name = gengraph.gen_syn1(
        feature_generator=featgen.ConstFeatureGen(np.ones(args.input_dim, dtype=float))
    )
    num_classes = max(labels) + 1

    if args.method == "att":
        print("Method: att")
        model = models.GcnEncoderNode(
            args.input_dim,
            args.hidden_dim,
            args.output_dim,
            num_classes,
            args.num_gc_layers,
            bn=args.bn,
            args=args,
        )
    else:
        print("Method:", args.method)
        model = models.GcnEncoderNode(
            args.input_dim,
            args.hidden_dim,
            args.output_dim,
            num_classes,
            args.num_gc_layers,
            bn=args.bn,
            args=args,
        )
    if args.gpu:
        model = model.cuda()

    train_node_classifier(G, labels, model, args, writer=writer)


def syn_task2(args, writer=None):
    # data
    G, labels, name = gengraph.gen_syn2()
    input_dim = len(G.nodes[0]["feat"])
    num_classes = max(labels) + 1

    if args.method == "attn":
        print("Method: attn")
    else:
        print("Method:", args.method)
        model = models.GcnEncoderNode(
            input_dim,
            args.hidden_dim,
            args.output_dim,
            num_classes,
            args.num_gc_layers,
            bn=args.bn,
            args=args,
        )
        if args.gpu:
            model = model.cuda()

    train_node_classifier(G, labels, model, args, writer=writer)


def syn_task3(args, writer=None):
    # data
    G, labels, name = gengraph.gen_syn3(
        feature_generator=featgen.ConstFeatureGen(np.ones(args.input_dim, dtype=float))
    )
    print(labels)
    num_classes = max(labels) + 1

    if args.method == "attn":
        print("Method: attn")
    else:
        print("Method:", args.method)
        model = models.GcnEncoderNode(
            args.input_dim,
            args.hidden_dim,
            args.output_dim,
            num_classes,
            args.num_gc_layers,
            bn=args.bn,
            args=args,
        )
        if args.gpu:
            model = model.cuda()

    train_node_classifier(G, labels, model, args, writer=writer)


def syn_task4(args, writer=None):
    # data
    G, labels, name = gengraph.gen_syn4(
        feature_generator=featgen.ConstFeatureGen(np.ones(args.input_dim, dtype=float))
    )
    print(labels)
    num_classes = max(labels) + 1

    if args.method == "attn":
        print("Method: attn")
    else:
        print("Method:", args.method)
        model = models.GcnEncoderNode(
            args.input_dim,
            args.hidden_dim,
            args.output_dim,
            num_classes,
            args.num_gc_layers,
            bn=args.bn,
            args=args,
        )

        if args.gpu:
            model = model.cuda()

    train_node_classifier(G, labels, model, args, writer=writer)


def syn_task5(args, writer=None):
    # data
    G, labels, name = gengraph.gen_syn5(
        feature_generator=featgen.ConstFeatureGen(np.ones(args.input_dim, dtype=float))
    )
    print(labels)
    print("Number of nodes: ", G.number_of_nodes())
    num_classes = max(labels) + 1

    if args.method == "attn":
        print("Method: attn")
    else:
        print("Method: base")
        model = models.GcnEncoderNode(
            args.input_dim,
            args.hidden_dim,
            args.output_dim,
            num_classes,
            args.num_gc_layers,
            bn=args.bn,
            args=args,
        )

        if args.gpu:
            model = model.cuda()

    train_node_classifier(G, labels, model, args, writer=writer)


def pkl_task(args, feat=None):
    with open(os.path.join(args.datadir, args.pkl_fname), "rb") as pkl_file:
        data = pickle.load(pkl_file)
    graphs = data[0]
    labels = data[1]
    test_graphs = data[2]
    test_labels = data[3]

    for i in range(len(graphs)):
        graphs[i].graph["label"] = labels[i]
    for i in range(len(test_graphs)):
        test_graphs[i].graph["label"] = test_labels[i]

    if feat is None:
        featgen_const = featgen.ConstFeatureGen(np.ones(args.input_dim, dtype=float))
        for G in graphs:
            featgen_const.gen_node_features(G)
        for G in test_graphs:
            featgen_const.gen_node_features(G)

    train_dataset, test_dataset, max_num_nodes = prepare_data(
        graphs, args, test_graphs=test_graphs
    )
    model = models.GcnEncoderGraph(
        args.input_dim,
        args.hidden_dim,
        args.output_dim,
        args.num_classes,
        args.num_gc_layers,
        bn=args.bn,
    ).cuda()
    train(train_dataset, model, args, test_dataset=test_dataset)
    evaluate(test_dataset, model, args, "Validation")


def enron_task_multigraph(args, idx=None, writer=None):
    labels_dict = {
        "None": 5,
        "Employee": 0,
        "Vice President": 1,
        "Manager": 2,
        "Trader": 3,
        "CEO+Managing Director+Director+President": 4,
    }
    max_enron_id = 183
    if idx is None:
        G_list = []
        labels_list = []
        for i in range(10):
            net = pickle.load(
                open("data/gnn-explainer-enron/enron_slice_{}.pkl".format(i), "rb")
            )
            net.add_nodes_from(range(max_enron_id))
            labels = [n[1].get("role", "None") for n in net.nodes(data=True)]
            labels_num = [labels_dict[l] for l in labels]
            featgen_const = featgen.ConstFeatureGen(
                np.ones(args.input_dim, dtype=float)
            )
            featgen_const.gen_node_features(net)
            G_list.append(net)
            labels_list.append(labels_num)
        # train_dataset, test_dataset, max_num_nodes = prepare_data(G_list, args)
        model = models.GcnEncoderNode(
            args.input_dim,
            args.hidden_dim,
            args.output_dim,
            args.num_classes,
            args.num_gc_layers,
            bn=args.bn,
            args=args,
        )
        if args.gpu:
            model = model.cuda()
        print(labels_num)
        train_node_classifier_multigraph(
            G_list, labels_list, model, args, writer=writer
        )
    else:
        print("Running Enron full task")


def enron_task(args, idx=None, writer=None):
    labels_dict = {
        "None": 5,
        "Employee": 0,
        "Vice President": 1,
        "Manager": 2,
        "Trader": 3,
        "CEO+Managing Director+Director+President": 4,
    }
    max_enron_id = 183
    if idx is None:
        G_list = []
        labels_list = []
        for i in range(10):
            net = pickle.load(
                open("data/gnn-explainer-enron/enron_slice_{}.pkl".format(i), "rb")
            )
            # net.add_nodes_from(range(max_enron_id))
            # labels=[n[1].get('role', 'None') for n in net.nodes(data=True)]
            # labels_num = [labels_dict[l] for l in labels]
            featgen_const = featgen.ConstFeatureGen(
                np.ones(args.input_dim, dtype=float)
            )
            featgen_const.gen_node_features(net)
            G_list.append(net)
            print(net.number_of_nodes())
            # labels_list.append(labels_num)

        G = nx.disjoint_union_all(G_list)
        model = models.GcnEncoderNode(
            args.input_dim,
            args.hidden_dim,
            args.output_dim,
            len(labels_dict),
            args.num_gc_layers,
            bn=args.bn,
            args=args,
        )
        labels = [n[1].get("role", "None") for n in G.nodes(data=True)]
        labels_num = [labels_dict[l] for l in labels]
        for i in range(5):
            print("Label ", i, ": ", labels_num.count(i))

        print("Total num nodes: ", len(labels_num))
        print(labels_num)

        if args.gpu:
            model = model.cuda()
        train_node_classifier(G, labels_num, model, args, writer=writer)
    else:
        print("Running Enron full task")


def benchmark_task(args, writer=None, feat="node-label"):
    graphs = io_utils.read_graphfile(
        args.datadir, args.bmname, max_nodes=args.max_nodes
    )
    print(max([G.graph["label"] for G in graphs]))

    if feat == "node-feat" and "feat_dim" in graphs[0].graph:
        print("Using node features")
        input_dim = graphs[0].graph["feat_dim"]
    elif feat == "node-label" and "label" in graphs[0].nodes[0]:
        print("Using node labels")
        for G in graphs:
            for u in G.nodes():
                G.nodes[u]["feat"] = np.array(G.nodes[u]["label"])
                # make it -1/1 instead of 0/1
                # feat = np.array(G.nodes[u]['label'])
                # G.nodes[u]['feat'] = feat * 2 - 1
    else:
        print("Using constant labels")
        featgen_const = featgen.ConstFeatureGen(np.ones(args.input_dim, dtype=float))
        for G in graphs:
            featgen_const.gen_node_features(G)

    train_dataset, val_dataset, test_dataset, max_num_nodes, input_dim, assign_input_dim = prepare_data(
        graphs, args, max_nodes=args.max_nodes
    )
    if args.method == "soft-assign":
        print("Method: soft-assign")
        model = models.SoftPoolingGcnEncoder(
            max_num_nodes,
            input_dim,
            args.hidden_dim,
            args.output_dim,
            args.num_classes,
            args.num_gc_layers,
            args.hidden_dim,
            assign_ratio=args.assign_ratio,
            num_pooling=args.num_pool,
            bn=args.bn,
            dropout=args.dropout,
            linkpred=args.linkpred,
            args=args,
            assign_input_dim=assign_input_dim,
        ).cuda()
    else:
        print("Method: base")
        model = models.GcnEncoderGraph(
            input_dim,
            args.hidden_dim,
            args.output_dim,
            args.num_classes,
            args.num_gc_layers,
            bn=args.bn,
            dropout=args.dropout,
            args=args,
        ).cuda()

    train(
        train_dataset,
        model,
        args,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        writer=writer,
    )
    evaluate(test_dataset, model, args, "Validation")


def benchmark_task_val(args, writer=None, feat="node-label"):
    all_vals = []
    graphs = io_utils.read_graphfile(
        args.datadir, args.bmname, max_nodes=args.max_nodes
    )

    if feat == "node-feat" and "feat_dim" in graphs[0].graph:
        print("Using node features")
        input_dim = graphs[0].graph["feat_dim"]
    elif feat == "node-label" and "label" in graphs[0].nodes[0]:
        print("Using node labels")
        for G in graphs:
            for u in G.nodes():
                G.nodes[u]["feat"] = np.array(G.nodes[u]["label"])
    else:
        print("Using constant labels")
        featgen_const = featgen.ConstFeatureGen(np.ones(args.input_dim, dtype=float))
        for G in graphs:
            featgen_const.gen_node_features(G)

    # 10 splits
    for i in range(10):
        train_dataset, val_dataset, max_num_nodes, input_dim, assign_input_dim = cross_val.prepare_val_data(
            graphs, args, i, max_nodes=args.max_nodes
        )
        print("Method: base")
        model = models.GcnEncoderGraph(
            input_dim,
            args.hidden_dim,
            args.output_dim,
            args.num_classes,
            args.num_gc_layers,
            bn=args.bn,
            dropout=args.dropout,
            args=args,
        ).cuda()

        _, val_accs = train(
            train_dataset,
            model,
            args,
            val_dataset=val_dataset,
            test_dataset=None,
            writer=writer,
        )
        all_vals.append(np.array(val_accs))
    all_vals = np.vstack(all_vals)
    all_vals = np.mean(all_vals, axis=0)
    print(all_vals)
    print(np.max(all_vals))
    print(np.argmax(all_vals))


def arg_parse():
    parser = argparse.ArgumentParser(description="GraphPool arguments.")
    io_parser = parser.add_mutually_exclusive_group(required=False)
    io_parser.add_argument("--dataset", dest="dataset", help="Input dataset.")
    benchmark_parser = io_parser.add_argument_group()
    benchmark_parser.add_argument(
        "--bmname", dest="bmname", help="Name of the benchmark dataset"
    )
    io_parser.add_argument("--pkl", dest="pkl_fname", help="Name of the pkl data file")

    softpool_parser = parser.add_argument_group()
    softpool_parser.add_argument(
        "--assign-ratio",
        dest="assign_ratio",
        type=float,
        help="ratio of number of nodes in consecutive layers",
    )
    softpool_parser.add_argument(
        "--num-pool", dest="num_pool", type=int, help="number of pooling layers"
    )
    parser.add_argument(
        "--linkpred",
        dest="linkpred",
        action="store_const",
        const=True,
        default=False,
        help="Whether link prediction side objective is used",
    )

    parser_utils.parse_optimizer(parser)

    parser.add_argument(
        "--datadir", dest="datadir", help="Directory where benchmark is located"
    )
    parser.add_argument("--logdir", dest="logdir", help="Tensorboard log directory")
    parser.add_argument("--ckptdir", dest="ckptdir", help="Model checkpoint directory")
    parser.add_argument("--cuda", dest="cuda", help="CUDA.")
    parser.add_argument(
        "--gpu",
        dest="gpu",
        action="store_const",
        const=True,
        default=False,
        help="whether to use GPU.",
    )
    parser.add_argument(
        "--max-nodes",
        dest="max_nodes",
        type=int,
        help="Maximum number of nodes (ignore graghs with nodes exceeding the number.",
    )
    parser.add_argument("--batch-size", dest="batch_size", type=int, help="Batch size.")
    parser.add_argument(
        "--epochs", dest="num_epochs", type=int, help="Number of epochs to train."
    )
    parser.add_argument(
        "--train-ratio",
        dest="train_ratio",
        type=float,
        help="Ratio of number of graphs training set to all graphs.",
    )
    parser.add_argument(
        "--num_workers",
        dest="num_workers",
        type=int,
        help="Number of workers to load data.",
    )
    parser.add_argument(
        "--feature",
        dest="feature_type",
        help="Feature used for encoder. Can be: id, deg",
    )
    parser.add_argument(
        "--input-dim", dest="input_dim", type=int, help="Input feature dimension"
    )
    parser.add_argument(
        "--hidden-dim", dest="hidden_dim", type=int, help="Hidden dimension"
    )
    parser.add_argument(
        "--output-dim", dest="output_dim", type=int, help="Output dimension"
    )
    parser.add_argument(
        "--num-classes", dest="num_classes", type=int, help="Number of label classes"
    )
    parser.add_argument(
        "--num-gc-layers",
        dest="num_gc_layers",
        type=int,
        help="Number of graph convolution layers before each pooling",
    )
    parser.add_argument(
        "--bn",
        dest="bn",
        action="store_const",
        const=True,
        default=False,
        help="Whether batch normalization is used",
    )
    parser.add_argument("--dropout", dest="dropout", type=float, help="Dropout rate.")
    parser.add_argument(
        "--nobias",
        dest="bias",
        action="store_const",
        const=False,
        default=True,
        help="Whether to add bias. Default to True.",
    )
    parser.add_argument(
        "--weight-decay",
        dest="weight_decay",
        type=float,
        help="Weight decay regularization constant.",
    )

    parser.add_argument(
        "--method", dest="method", help="Method. Possible values: base, "
    )
    parser.add_argument(
        "--name-suffix", dest="name_suffix", help="suffix added to the output filename"
    )

    parser.set_defaults(
        datadir="data",  # io_parser
        logdir="log",
        ckptdir="ckpt",
        dataset="german",
        opt="adam",  # opt_parser
        opt_scheduler="none",
        max_nodes=100,
        cuda="1",
        feature_type="default",
        lr=0.001,
        clip=2.0,
        batch_size=20,
        num_epochs=1000,
        train_ratio=0.8,
        test_ratio=0.1,
        num_workers=1,
        input_dim=27,
        hidden_dim=20,
        output_dim=20,
        num_classes=2,
        num_gc_layers=3,
        dropout=0.0,
        weight_decay=0.005,
        method="base",
        name_suffix="",
        assign_ratio=0.1,
    )
    return parser.parse_args()


def main():
    prog_args = configs.arg_parse()

    path = os.path.join(prog_args.logdir, io_utils.gen_prefix(prog_args))
    writer = SummaryWriter(path)

    prog_args.gpu=True

    if prog_args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = prog_args.cuda
        print("CUDA", prog_args.cuda)
    else:
        print("Using CPU")

    prog_args.dataset='german'
    prog_args.num_classes=2
    prog_args.method='base'

    prog_args.num_epochs=1000

    if prog_args.dataset=='bail' or prog_args.dataset=='credit':
        prog_args.num_epochs=300



    # use --bmname=[dataset_name] for Reddit-Binary, Mutagenicity
    if prog_args.bmname is not None:
        benchmark_task(prog_args, writer=writer)
    elif prog_args.pkl_fname is not None:
        pkl_task(prog_args)
    elif prog_args.dataset is not None:
        if prog_args.dataset == "syn1":
            syn_task1(prog_args, writer=writer)
        elif prog_args.dataset == "syn2":
            syn_task2(prog_args, writer=writer)
        elif prog_args.dataset == "syn3":
            syn_task3(prog_args, writer=writer)
        elif prog_args.dataset == "syn4":
            syn_task4(prog_args, writer=writer)
        elif prog_args.dataset == "syn5":
            syn_task5(prog_args, writer=writer)
        elif prog_args.dataset == "enron":
            enron_task(prog_args, writer=writer)
        elif prog_args.dataset == "ppi_essential":
            ppi_essential_task(prog_args, writer=writer)
        elif prog_args.dataset == 'german' or prog_args.dataset == 'bail' or prog_args.dataset == 'credit':
            result=[]
            result.append(task_real_data(prog_args,writer=writer))
        elif prog_args.dataset == 'synthetic':
            task_synthetic(prog_args,writer=writer)
        elif prog_args.dataset == 'synthetic2':
            task_synthetic2(prog_args,writer=writer)

    writer.close()


if __name__ == "__main__":
    main()

