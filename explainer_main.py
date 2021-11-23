""" explainer_main.py

     Main user interface for the explainer module.
"""
import argparse
import os

import sklearn.metrics as metrics
import pandas as pd
from tensorboardX import SummaryWriter
import gengraph
import pickle
import shutil
import torch
import numpy as np
import models
import utils.io_utils as io_utils
import utils.parser_utils as parser_utils
from explainer import explain
import scipy.sparse as sp
import networkx as nx
import warnings

warnings.filterwarnings("ignore")

def load_credit(dataset, sens_attr="Age", predict_attr="NoDefaultNextMonth", path="./dataset/credit/", label_number=1000):
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

def arg_parse():
    parser = argparse.ArgumentParser(description="GNN Explainer arguments.")
    io_parser = parser.add_mutually_exclusive_group(required=False)
    io_parser.add_argument("--dataset", dest="dataset", help="Input dataset.")
    benchmark_parser = io_parser.add_argument_group()
    benchmark_parser.add_argument(
        "--bmname", dest="bmname", help="Name of the benchmark dataset"
    )
    io_parser.add_argument("--pkl", dest="pkl_fname", help="Name of the pkl data file")

    parser_utils.parse_optimizer(parser)

    parser.add_argument("--clean-log", action="store_true", help="If true, cleans the specified log directory before running.")
    parser.add_argument("--logdir", dest="logdir", help="Tensorboard log directory")
    parser.add_argument("--ckptdir", dest="ckptdir", help="Model checkpoint directory")
    parser.add_argument("--cuda", dest="cuda", help="CUDA.")
    parser.add_argument(
        "--gpu",
        dest="gpu",
        action="store_const",
        const=True,
        default=True,
        help="whether to use GPU.",
    )
    parser.add_argument(
        "--epochs", dest="num_epochs", type=int, help="Number of epochs to train."
    )
    parser.add_argument(
        "--hidden-dim", dest="hidden_dim", type=int, help="Hidden dimension"
    )
    parser.add_argument(
        "--output-dim", dest="output_dim", type=int, help="Output dimension"
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
        "--no-writer",
        dest="writer",
        action="store_const",
        const=False,
        default=True,
        help="Whether to add bias. Default to True.",
    )
    # Explainer
    parser.add_argument("--mask-act", dest="mask_act", type=str, help="sigmoid, ReLU.")
    parser.add_argument(
        "--mask-bias",
        dest="mask_bias",
        action="store_const",
        const=True,
        default=False,
        help="Whether to add bias. Default to True.",
    )
    parser.add_argument(
        "--explain-node", dest="explain_node", type=int, help="Node to explain."
    )
    parser.add_argument(
        "--graph-idx", dest="graph_idx", type=int, help="Graph to explain."
    )
    parser.add_argument(
        "--graph-mode",
        dest="graph_mode",
        action="store_const",
        const=True,
        default=False,
        help="whether to run Explainer on Graph Classification task.",
    )
    parser.add_argument(
        "--multigraph-class",
        dest="multigraph_class",
        type=int,
        help="whether to run Explainer on multiple Graphs from the Classification task for examples in the same class.",
    )
    parser.add_argument(
        "--multinode-class",
        dest="multinode_class",
        type=int,
        help="whether to run Explainer on multiple nodes from the Classification task for examples in the same class.",
    )
    parser.add_argument(
        "--align-steps",
        dest="align_steps",
        type=int,
        help="Number of iterations to find P, the alignment matrix.",
    )

    parser.add_argument(
        "--method", dest="method", type=str, help="Method. Possible values: base, att."
    )
    parser.add_argument(
        "--name-suffix", dest="name_suffix", help="suffix added to the output filename"
    )
    parser.add_argument(
        "--explainer-suffix",
        dest="explainer_suffix",
        help="suffix added to the explainer log",
    )

    # TODO: Check argument usage
    parser.set_defaults(
        logdir="log",
        ckptdir="ckpt",
        dataset="german",
        opt="adam",
        opt_scheduler="none",
        cuda="0",
        lr=0.1,
        clip=2.0,
        batch_size=20,
        num_epochs=200,
        hidden_dim=20,
        output_dim=20,
        num_gc_layers=3,
        dropout=0.0,
        method="base",
        name_suffix="",
        explainer_suffix="",
        align_steps=1000,
        explain_node=None,
        graph_idx=-1,
        mask_act="sigmoid",
        multigraph_class=-1,
        multinode_class=-1,
    )
    return parser.parse_args()


def main():

    np.random.seed(1)


    # Load a configuration
    prog_args = arg_parse()

    if prog_args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = prog_args.cuda
        print("CUDA", prog_args.cuda)
    else:
        print("Using CPU")
        print(1/0)

    # Configure the logging directory
    if prog_args.writer:
        path = os.path.join(prog_args.logdir, io_utils.gen_explainer_prefix(prog_args))
        if os.path.isdir(path) and prog_args.clean_log:
            print('Removing existing log dir: ', path)
            if not input("Are you sure you want to remove this directory? (y/n): ").lower().strip()[:1] == "y": sys.exit(1)
            shutil.rmtree(path)
        writer = SummaryWriter(path)
    else:
        writer = None

    # Load a model checkpoint
    ckpt = io_utils.load_ckpt(prog_args)
    cg_dict = ckpt["cg"] # get computation graph


    if prog_args.dataset=='credit':

        if prog_args.dataset=='credit':
            path = "../nifty/dataset/credit"
            adj, features, labels, edges, sens = load_credit('credit', path=path)
        elif prog_args.dataset=='bail':
            path = "../nifty/dataset/bail"
            adj, features, labels, edges, sens = load_bail('bail', path=path)



        G = nx.Graph()

        G.add_edges_from([edges[i].tolist() for i in range(edges.shape[0])])

        feat_dict = {i:{'feat': np.array(features[i], dtype=np.float32)} for i in G.nodes()}

        nx.set_node_attributes(G, feat_dict)




        data = gengraph.preprocess_input_graph(G, labels)
        def feature_norm(features):
            min_values = features.min(axis=0,keepdims=True)
            max_values = features.max(axis=0,keepdims=True)
            return 2*(features - min_values)/(max_values-min_values) - 1


        #data['feat']/=data['feat'].max(1,keepdims=True)
        #data['feat']=np.random.randn(data['feat'].shape[0],data['feat'].shape[1],data['feat'].shape[2])
        data['feat'][0][:,:-1]=feature_norm(data['feat'][0][:,:-1])


        data['adj'][0]=data['adj'][0]+np.identity(data['adj'][0].shape[-1])


        cg_dict['adj']=data['adj']
        cg_dict['feat']=data['feat']


    input_dim = cg_dict["feat"].shape[2]
    num_classes = cg_dict["pred"].shape[2]
    print("Loaded model from {}".format(prog_args.ckptdir))
    print("input dim: ", input_dim, "; num classes: ", num_classes)

    # Determine explainer mode
    graph_mode = (
            prog_args.graph_mode
            or prog_args.multigraph_class >= 0
            or prog_args.graph_idx >= 0
    )

    # build model
    print("Method: ", prog_args.method)
    if graph_mode:
        # Explain Graph prediction
        model = models.GcnEncoderGraph(
            input_dim=input_dim,
            hidden_dim=prog_args.hidden_dim,
            embedding_dim=prog_args.output_dim,
            label_dim=num_classes,
            num_layers=prog_args.num_gc_layers,
            bn=prog_args.bn,
            args=prog_args,
        )
    else:
        if prog_args.dataset == "ppi_essential":
            # class weight in CE loss for handling imbalanced label classes
            prog_args.loss_weight = torch.tensor([1.0, 5.0], dtype=torch.float).cuda()
            # Explain Node prediction
        model = models.GcnEncoderNode(
            input_dim=input_dim,
            hidden_dim=prog_args.hidden_dim,
            embedding_dim=prog_args.output_dim,
            label_dim=num_classes,
            num_layers=prog_args.num_gc_layers,
            bn=prog_args.bn,
            args=prog_args,
        )
    if prog_args.gpu:
        model = model.cuda()
    # load state_dict (obtained by model.state_dict() when saving checkpoint)
    model.load_state_dict(ckpt["model_state"])

    # Create explainer
    explainer = explain.Explainer(
        model=model,
        adj=cg_dict["adj"],
        feat=cg_dict["feat"],
        label=cg_dict["label"],
        pred=cg_dict["pred"],
        train_idx=cg_dict["train_idx"],
        args=prog_args,
        writer=writer,
        print_training=True,
        graph_mode=graph_mode,
        graph_idx=prog_args.graph_idx,
    )

    # TODO: API should definitely be cleaner
    # Let's define exactly which modes we support
    # We could even move each mode to a different method (even file)
    if prog_args.explain_node is not None:
        explainer.explain(prog_args.explain_node, unconstrained=False)
    elif graph_mode:
        if prog_args.multigraph_class >= 0:
            print(cg_dict["label"])
            # only run for graphs with label specified by multigraph_class
            labels = cg_dict["label"].numpy()
            graph_indices = []
            for i, l in enumerate(labels):
                if l == prog_args.multigraph_class:
                    graph_indices.append(i)
                if len(graph_indices) > 30:
                    break
            print(
                "Graph indices for label ",
                prog_args.multigraph_class,
                " : ",
                graph_indices,
            )
            explainer.explain_graphs(graph_indices=graph_indices)

        elif prog_args.graph_idx == -1:
            # just run for a customized set of indices
            explainer.explain_graphs(graph_indices=[1, 2, 3, 4])
        else:
            explainer.explain(
                node_idx=0,
                graph_idx=prog_args.graph_idx,
                graph_mode=True,
                unconstrained=False,
            )
            io_utils.plot_cmap_tb(writer, "tab20", 20, "tab20_cmap")
    else:
        if prog_args.multinode_class >= 0:
            print(cg_dict["label"])
            # only run for nodes with label specified by multinode_class
            labels = cg_dict["label"][0]  # already numpy matrix

            node_indices = []
            for i, l in enumerate(labels):
                if len(node_indices) > 4:
                    break
                if l == prog_args.multinode_class:
                    node_indices.append(i)
            print(
                "Node indices for label ",
                prog_args.multinode_class,
                " : ",
                node_indices,
            )
            explainer.explain_nodes(node_indices, prog_args)

        else:
            if prog_args.dataset=='synthetic':
                idx_map={i:one for i,one in enumerate([0, 8, 27, 100, 107, 135, 136, 1, 2, 56, 58, 72, 79, 92, 125, 145, 173, 20, 29, 33, 78, 89, 98, 115, 126, 142, 156, 175, 3, 10, 17, 74, 90, 105, 109, 128, 155, 162, 171, 4, 5, 30, 49, 61, 95, 112, 119, 127, 154, 23, 120, 148, 6, 18, 37, 63, 73, 77, 163, 169, 172, 7, 19, 86, 94, 139, 15, 53, 91, 122, 161, 9, 44, 132, 144, 110, 113, 11, 116, 130, 141, 158, 12, 16, 41, 62, 65, 99, 123, 159, 13, 38, 57, 151, 14, 22, 59, 149, 32, 150, 157, 160, 26, 153, 40, 51, 106, 124, 137, 165, 34, 102, 170, 371, 50, 60, 143, 21, 80, 25, 85, 168, 71, 108, 114, 117, 146, 24, 45, 68, 164, 179, 35, 42, 43, 48, 75, 140, 76, 84, 167, 28, 131, 174, 314, 36, 134, 31, 55, 82, 111, 177, 87, 147, 39, 66, 97, 271, 720, 67, 69, 152, 138, 83, 178, 88, 46, 228, 47, 118, 52, 101, 104, 96, 360, 64, 54, 121, 81, 103, 530, 70, 301, 739, 93, 133, 129, 642, 660, 176, 597, 624, 370, 245, 205, 777, 166, 443, 504, 464, 391, 775, 180, 225, 389, 400, 463, 619, 181, 226, 365, 474, 515, 576, 722, 182, 274, 305, 482, 632, 745, 780, 183, 186, 281, 387, 425, 531, 609, 714, 184, 283, 329, 392, 565, 657, 679, 683, 705, 731, 742, 185, 437, 498, 510, 644, 401, 428, 509, 513, 721, 187, 206, 244, 406, 525, 188, 212, 313, 380, 419, 445, 573, 688, 708, 816, 189, 291, 478, 489, 550, 792, 190, 402, 468, 599, 647, 661, 191, 238, 321, 407, 460, 192, 335, 424, 588, 639, 687, 814, 193, 233, 354, 364, 439, 440, 559, 590, 605, 614, 696, 724, 194, 578, 758, 800, 195, 266, 340, 415, 785, 794, 196, 215, 242, 493, 548, 628, 637, 673, 682, 197, 532, 641, 645, 653, 707, 717, 789, 198, 516, 536, 577, 580, 674, 748, 765, 199, 289, 506, 527, 572, 691, 809, 200, 248, 269, 350, 678, 201, 385, 600, 634, 703, 715, 202, 293, 300, 322, 332, 495, 545, 567, 581, 591, 203, 275, 285, 497, 204, 277, 488, 492, 701, 246, 429, 783, 427, 603, 685, 757, 207, 232, 331, 677, 808, 208, 405, 455, 209, 243, 280, 403, 507, 535, 686, 210, 252, 399, 667, 211, 411, 523, 571, 620, 627, 670, 312, 337, 458, 643, 712, 762, 213, 471, 214, 261, 349, 395, 684, 273, 299, 404, 595, 764, 216, 430, 449, 542, 585, 217, 382, 802, 218, 236, 386, 453, 598, 219, 561, 589, 656, 772, 220, 381, 384, 447, 749, 221, 431, 435, 520, 759, 222, 279, 326, 608, 223, 268, 311, 612, 224, 434, 529, 592, 636, 500, 700, 363, 408, 227, 295, 375, 524, 716, 756, 270, 564, 579, 229, 241, 327, 334, 569, 761, 230, 558, 741, 774, 231, 601, 665, 812, 254, 366, 390, 817, 234, 287, 397, 235, 336, 568, 760, 347, 353, 690, 237, 613, 779, 302, 357, 649, 239, 537, 240, 259, 272, 469, 250, 517, 342, 442, 475, 543, 563, 754, 362, 514, 782, 788, 586, 746, 255, 818, 247, 799, 521, 730, 249, 566, 583, 786, 258, 494, 640, 251, 374, 466, 602, 253, 456, 459, 526, 781, 784, 372, 409, 256, 257, 260, 650, 361, 451, 633, 308, 448, 805, 341, 522, 546, 503, 680, 262, 617, 263, 318, 264, 485, 501, 582, 668, 769, 265, 383, 635, 267, 740, 422, 473, 477, 584, 306, 410, 626, 547, 778, 483, 499, 651, 771, 484, 692, 763, 278, 414, 631, 709, 344, 480, 570, 796, 276, 323, 695, 819, 593, 368, 436, 454, 551, 664, 726, 298, 552, 496, 533, 282, 444, 519, 338, 798, 284, 351, 553, 606, 505, 286, 355, 706, 728, 288, 556, 693, 446, 290, 465, 795, 394, 616, 292, 727, 396, 621, 294, 369, 378, 766, 825, 296, 304, 297, 339, 413, 418, 433, 797, 343, 438, 575, 539, 689, 303, 625, 719, 346, 658, 659, 732, 330, 307, 804, 376, 309, 310, 345, 587, 607, 961, 557, 315, 694, 316, 676, 317, 319, 320, 486, 476, 367, 324, 325, 426, 555, 560, 791, 328, 423, 753, 462, 604, 807, 725, 333, 737, 713, 450, 946, 544, 710, 787, 810, 379, 738, 736, 348, 594, 352, 420, 574, 926, 610, 356, 377, 699, 358, 562, 359, 393, 734, 421, 538, 417, 768, 755, 534, 618, 681, 747, 398, 416, 698, 623, 666, 373, 412, 461, 697, 487, 491, 743, 803, 672, 388, 793, 750, 776, 806, 702, 893, 432, 654, 770, 540, 630, 646, 733, 790, 718, 452, 752, 723, 512, 815, 662, 767, 441, 735, 508, 652, 470, 472, 541, 457, 655, 663, 467, 615, 490, 638, 851, 479, 481, 773, 813, 801, 711, 831, 502, 751, 511, 549, 518, 729, 596, 528, 811, 629, 949, 648, 554, 704, 611, 622, 842, 962, 669, 671, 675, 958, 983, 968, 922, 744, 906, 952, 900, 972, 879, 820, 822, 839, 845, 883, 886, 889, 908, 941, 955, 960, 821, 846, 854, 866, 871, 890, 932, 959, 980, 992, 994, 867, 873, 888, 933, 938, 966, 979, 991, 823, 827, 864, 865, 874, 875, 882, 912, 917, 945, 967, 976, 986, 999, 824, 834, 860, 919, 921, 940, 841, 880, 894, 902, 911, 925, 988, 826, 840, 844, 849, 868, 881, 905, 909, 974, 995, 835, 870, 887, 904, 931, 944, 964, 828, 869, 916, 923, 984, 993, 829, 884, 927, 954, 830, 850, 853, 856, 969, 878, 934, 987, 832, 977, 833, 848, 897, 863, 913, 956, 915, 989, 836, 862, 918, 971, 998, 837, 876, 951, 838, 910, 924, 981, 896, 914, 928, 939, 950, 903, 843, 929, 898, 982, 907, 947, 973, 847, 930, 943, 936, 858, 852, 857, 899, 872, 877, 942, 861, 855, 996, 997, 948, 859, 891, 892, 937, 885, 920, 985, 957, 963, 935, 901, 978, 953, 895, 970, 990, 965, 975])}

                node_list=[]
                for j in range(1000):
                    a=idx_map[j]
                    if a>=180 and a<1000-180:
                        node_list.append(j)
            elif prog_args.dataset=='synthetic2':

                node_list=list(range(cg_dict["feat"].shape[1]))
            else:
                node_list=np.random.permutation(list(range(cg_dict["feat"].shape[1])))[:5]
                print(node_list)


            # explain a set of nodes
            masked_adj = explainer.explain_nodes_gnn_stats(
                node_list, prog_args,model='exp'
            )
            print('Equal Ori Rate',np.mean(explainer.explain_true))

if __name__ == "__main__":
    main()

