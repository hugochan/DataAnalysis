#!/usr/bin/env python
#encoding=utf-8

import sys, json
from math import log
import numpy as np
from scipy import sparse
from scipy import io
import matplotlib.pyplot as plt
import pdb, time

class DataAnalysis(object):
    """do some data analysis"""
    def __init__(self, filepath):
        super(DataAnalysis, self).__init__()
        self.__filepath = filepath
        self.userset = []# user set
        self.itemset = {}# item set: {"item":index,...}
        self.__instanceSet = {}# instance data set: {user:[item,...],...}
        self.instancenum = 0# num of instance
        self._import_data()
        t0 = time.clock()
        self._create_ui_matrix("offline")
        t1 = time.clock()
        print "create_ui_matrix time costs"
        print t1-t0

    def _import_data(self):
        try:
            with open(self.__filepath, 'r') as f:
                temp_itemset = []# item set       
                templine = f.readline()
                while(templine):
                    self.instancenum += 1
                    temp = templine.split('\t')[:2]
                    user = int(temp[0])
                    item = int(temp[1][:-1])
                    temp_itemset.append(item)
                    try:
                        self.__instanceSet[user].append(item)
                    except:
                        self.__instanceSet[user] = [item]
                    templine = f.readline()
        except Exception, e:
            print "import datas error !"
            print e
            sys.exit()
        f.close()
        self.userset = self.__instanceSet.keys()
        temp_itemset = list(set(temp_itemset))# remove redundancy
        self.usernum = len(self.userset)
        self.itemnum = len(temp_itemset)
        for item_index in range(self.itemnum):
            self.itemset[temp_itemset[item_index]] = item_index

        print "user num:"
        print self.usernum
        print "item num:"
        print self.itemnum
        print "instance num:"
        print self.instancenum

    def _create_ui_matrix(self, method="online"):
        filepath = "./offline_results/ui_matrix"
        if method == "online":
            self.ui_matrix = sparse.lil_matrix((self.itemnum, self.usernum))
            user_index = 0
            for user, item in self.__instanceSet.iteritems():
                for eachitem in item:
                    self.ui_matrix[self.itemset[eachitem], user_index] = 1 # index operation is exacting !
                user_index += 1
            try:
                io.mmwrite(filepath, self.ui_matrix)
            except Exception, e:
                print e
                sys.exit()
        elif method == "offline":
            try:
                self.ui_matrix = io.mmread(filepath)
            except Exception,e:
                print e
                sys.exit()
        else:
            print "method arg error !"
            sys.exit()


    def creat_sim_matrix(self, target, method="online"):
        """Jaccard similarity
        target = 'user' means similarity among users, target = 'item' means similarity among items.
        method = 'online' means online calculation, method = 'offline' means using offline results.
        """
        filepath = "./offline_results/similarity"
        if method == "online":# online calculation
            if target == "user":# for user done
                self.ui_matrix = self.ui_matrix.tocsc()
                
                times = int(log(self.usernum, 2))
                unionsection = sparse.coo_matrix(self.ui_matrix.sum(0))
                for each in np.arange(times):
                    unionsection = sparse.vstack([unionsection, unionsection])
                unionsection = unionsection.tocsc()
                if self.usernum-2**times > 0:
                    unionsection = sparse.vstack([unionsection, unionsection[:self.usernum-2**times, :]])
                unionsection = unionsection.tocsc()

                intersection = (self.ui_matrix.transpose()).dot(self.ui_matrix)
                # sum subtracts intersection equals to unionsection,
                # no worry about zero-zero division for sparse matrix
                unionsection = unionsection + unionsection.transpose() - intersection
                
                # transfer to a upper angle matrix: demanding!!
                # intersection = intersection.tolil()
                # for c in np.arange(self.usernum):
                #         intersection[c, :c] = 0
                # intersection = intersection.tocsc()
                
                similarity = intersection/unionsection

            elif target == "item":# for item
                pdb.set_trace()
                self.ui_matrix = self.ui_matrix.tocsr()
                
                times = int(log(self.itemnum, 2))
                unionsection = sparse.coo_matrix(self.ui_matrix.sum(1))
                for each in np.arange(times):
                    unionsection = sparse.hstack([unionsection, unionsection])
                unionsection = unionsection.tocsr()
                if self.itemnum-2**times > 0:
                    unionsection = sparse.hstack([unionsection, unionsection[:, :self.itemnum-2**times]])
                unionsection = unionsection.tocsr()

                intersection = self.ui_matrix.dot(self.ui_matrix.transpose())
                # sum subtracts intersection equals to unionsection,
                # no worry about zero-zero division for sparse matrix
                unionsection = unionsection + unionsection.transpose() - intersection# unionsection
                
                # transfer to a upper angle matrix: demanding!!
                # intersection = intersection.tolil()
                # for r in np.arange(self.itemnum):
                #         intersection[r, :r] = 0
                # intersection = intersection.tocsr()
                
                similarity = intersection/unionsection

            else:
                print "target arg error !"
                sys.exit()

            # similarity = ((similarity/threshold_similarity).astype('int8').astype('float64'))\
            #         *threshold_similarity# rounding
            try:
                io.mmwrite(filepath+"_%s"%target, similarity)
            except Exception,e:
                print e
                sys.exit()
            return similarity
        elif method == "offline":# using offline results
            if target == "user" or target == "item":
                try:
                    similarity = io.mmread(filepath+"_%s"%target)
                except Exception, e:
                    print e
                    sys.exit()
                return similarity
            else:
                print "target arg error !"
                sys.exit()
        else:
            print "method arg error !"
            sys.exit()

    def degree_analysis(self, target, method="online"):
        """degree distribution
        target = 'user' means degree distribution for users,
        target = 'item' means degree distribution for items.
        method = 'online' means online calculation,
        method = 'offline' means using offlinie results."""
        filepath = "./offline_results/degree_distribution"
        if method == "online":# online calculation
            degree_distribution = {}
            if target == "user":# for users
                degree = np.asarray(self.ui_matrix.sum(0))[0].tolist()
                for eachdegree in set(degree):
                    degree_distribution[eachdegree] = degree.count(eachdegree)
            elif target == "item":# for items
                degree = np.asarray(self.ui_matrix.sum(1).transpose())[0].tolist()
                for eachdegree in set(degree):
                    degree_distribution[eachdegree] = degree.count(eachdegree)
            else:
                print "target arg error !"
                sys.exit()
            degree_distribution = dict(sorted(degree_distribution.iteritems(),\
                    key=lambda d:d[0],reverse = False))
            self.store_data(json.dumps(degree_distribution, ensure_ascii=False),\
                filepath+"_%s.json"%target)
            return degree_distribution
        elif method == "offline":# using offline results
            if target == "user" or target == "item":
                degree_distribution = self.read_data(filepath+"_%s.json"%target)
                if degree_distribution != "":
                    try:
                        degree_distribution = json.loads(degree_distribution)
                    except Exception, e:
                        print e
                        sys.exit()
                    # print degree_distribution.keys()
                    return degree_distribution
                else:
                    print "read nothing !"
                    sys.exit()
            else:
                print "target arg error !"
                sys.exit()
        else:
            print "method arg error !"
            sys.exit()

    def col_sim_analysis(self, similarity, target, method="online"):
        """collaborative similarity
        ref Empirical analysis of web-based user-object bipartite networks, Ming-Sheng Shang et al. 17 June 2010.
        target = 0 means collaborative similarity for users, target = 1 means collaborative similarity for items.
        method = 'online' means online calculation, method = 'offline' means using offline results.
        """
        filepath = "./offline_results/col_sim"
        if method == "online":# online calculation
            if target == "user":# for user
                col_sim = sparse.lil_matrix((1, self.usernum))
                self.ui_matrix = self.ui_matrix.tocsc()# fast column slicing operations
                degree = self.ui_matrix.sum(0)
                for u in np.arange(self.usernum):
                    # construct a similarity filtering matrix??????????????
                    if degree[0, u] > 1:
                        times = int(log(self.itemnum, 2))
                        temp = self.ui_matrix[:, u]
                        for each in np.arange(times):
                            temp = sparse.hstack([temp, temp])
                        temp = temp.tocsc()
                        if self.itemnum-2**times > 0:
                            temp = sparse.hstack([temp, temp[:,:self.itemnum-2**times]])
                        
                        temp = temp.tocsc()
                        similarity = similarity.tocsc()
                        similarity = (temp.multiply(temp.transpose())).multiply(similarity)

                        temp = degree[0, u]*(degree[0, u] - 1)
                        if temp == 0:
                            temp = 1
                        col_sim[0, u] = (similarity.sum()-similarity.diagonal().sum())/temp
                    else:
                        col_sim[0, u] = 0

            elif target == "item":# for item
                col_sim = sparse.lil_matrix((self.itemnum, 1))
                self.ui_matrix = self.ui_matrix.tocsr()
                degree = self.ui_matrix.sum(1)
                for o in np.arange(self.itemnum):
                    if degree[o, 0] > 1:
                        pass
                        # times = int(log(self.usernum, 2))
                        # temp = self.ui_matrix[o, :]
                        # for each in np.arange(times):
                        #     temp = sparse.vstack([temp, temp])
                        # temp = temp.tocsr()
                        # if self.usernum-2**times > 0:
                        #     temp = sparse.vstack([temp, temp[:self.usernum-2**times, :]])

                        # temp = temp.tocsr()
                        # similarity = similarity.tocsr()
                        # similarity = (temp.multiply(temp.transpose())).multiply(similarity)

                        # temp = degree[o, 0]*(degree[o, 0] - 1)
                        # if temp == 0:
                        #     temp = 1
                        # col_sim[o, 0] = (similarity.sum()-similarity.diagonal().sum())/temp
                    else:
                        col_sim[o, 0] = 0
            else:
                print "target arg error !"
                sys.exit()
            
            try:
                io.mmwrite(filepath+"_%s"%target, col_sim)
            except Exception, e:
                print e
                pdb.set_trace()
                sys.exit()
            return col_sim
        elif method == "offline":# using offline results
            if target == "user" or target == "item":
                try:
                    col_sim = io.mmread(filepath+"_%s"%target)
                except Exception, e:
                    print e
                    sys.exit()
                return col_sim
            else:
                print "target arg error !"
                sys.exit()
        else:
            print "method arg error !"
            sys.exit()

    def nearest_neighbor_degree_analysis(self, target, method="online"):
        """nearest neighbors' degree
        ref Empirical analysis of web-based user-object bipartite networks, Ming-Sheng Shang et al. 17 June 2010.
        target = 'user' means nearest neighbors' degree for users,
        target = 'item' means nearest neighbors' degree for items.        
        method = 'online' means online calculation,
        method = 'offline' means using offline results.        
        """
        filepath = "./offline_results/nn_degree"
        if method == "online":# online calculation
            tinynum = 0.00000001
            if target == "user":# for user
                self.ui_matrix = self.ui_matrix.tocsc()
                degree = sparse.csc_matrix(self.ui_matrix.sum(0))
                degree = degree + sparse.csc_matrix(np.ones([1, self.usernum]))*tinynum# to avoid zero division
                nn_degree = sparse.csc_matrix(self.ui_matrix.sum(1).transpose())\
                    .dot(self.ui_matrix)/degree
            elif target == "item":# for item
                self.ui_matrix = self.ui_matrix.tocsr()
                degree = sparse.csr_matrix(self.ui_matrix.sum(1))
                degree = degree + sparse.csr_matrix(np.ones([self.itemnum, 1]))*tinynum# to avoid zero division
                nn_degree = self.ui_matrix.dot(sparse.csr_matrix(self.ui_matrix.sum(0).transpose()))/degree
            else:
                print "target arg error !"
                sys.exit()
            if target == "user" or target == "item":
                try:
                    io.mmwrite(filepath+"_%s"%target, nn_degree)
                except Exception,e:
                    print e
                    sys.exit()
                return nn_degree
        elif method == "offline":# using offline results
            if target == "user" or target == "item":
                try:
                    nn_degree = io.mmread(filepath+"_%s"%target)
                except Exception, e:
                    print e
                    sys.exit()
                return nn_degree
            else:
                print "target arg error !"
                sys.exit()
        else:
            print "method arg error !"
            sys.exit()

    def draw_graph(self, analysis_data, data_type, target, save="off"):
        filepath = "./image/"
        plt.figure()
        if data_type == "degree_distribution":# degree
            x = analysis_data.keys()
            y = analysis_data.values()
            print x
            plt.xlabel("degree(%s)"%target)
            plt.ylabel("frequency")
            plt.title("degree distribution")
        elif data_type == "col_sim":# collaborative similarity
            degree_col_sim = {}
            if target == "user":# for user
                col_sim = analysis_data.toarray()[0].tolist()
                degree = np.asarray(self.ui_matrix.sum(0))[0].tolist()
                for index in np.arange(self.usernum):
                    try:
                        degree_col_sim[degree[index]].append(col_sim[index])
                    except:
                        degree_col_sim[degree[index]] = [col_sim[index]]
                for eachdegree, eachcol_sim in degree_col_sim.iteritems():
                    degree_col_sim[eachdegree] = np.array(eachcol_sim).mean()

            elif target == "item":# for item
                col_sim = analysis_data.transpose().toarray()[0].tolist()
                degree = np.asarray(self.ui_matrix.sum(1).transpose())[0].tolist()
                for index in np.arange(self.itemnum):
                    try:
                        degree_col_sim[degree[index]].append(col_sim[index])
                    except:
                        degree_col_sim[degree[index]] = [col_sim[index]]
                for eachdegree, eachcol_sim in degree_col_sim.iteritems():
                    degree_col_sim[eachdegree] = np.array(eachcol_sim).mean()

            else:
                print "target arg error !"
                sys.exit()
            degree_col_sim = dict(sorted(degree_col_sim.iteritems(),\
                    key=lambda d:d[0],reverse = False))
            x = degree_col_sim.keys()
            y = degree_col_sim.values()
            plt.xlabel("degree(%s)"%target)
            plt.ylabel("collaborative similarity")
            plt.title("collaborative similarity-degree distribution")
        elif data_type == "nn_degree":# nearest neighbors' degree
            degree_nndegree = {}
            if target == "user":# for user 
                nn_degree = analysis_data.toarray()[0].tolist()
                degree = np.asarray(self.ui_matrix.sum(0))[0].tolist()
                for index in np.arange(self.usernum):
                    try:
                        degree_nndegree[degree[index]].append(nn_degree[index])
                    except:
                        degree_nndegree[degree[index]] = [nn_degree[index]]
                for eachdegree, eachnn_degree in degree_nndegree.iteritems():
                    degree_nndegree[eachdegree] = np.array(eachnn_degree).mean()

            elif target == "item":# for item
                nn_degree = analysis_data.transpose().toarray()[0].tolist()
                degree = np.asarray(self.ui_matrix.sum(1).transpose())[0].tolist()
                for index in np.arange(self.itemnum):
                    try:
                        degree_nndegree[degree[index]].append(nn_degree[index])
                    except:
                        degree_nndegree[degree[index]] = [nn_degree[index]]
                for eachdegree, eachnn_degree in degree_nndegree.iteritems():
                    degree_nndegree[eachdegree] = np.array(eachnn_degree).mean()
            else:
                print "target arg error !"
                sys.exit()
            degree_nndegree = dict(sorted(degree_nndegree.iteritems(),\
                    key=lambda d:d[0],reverse = False))
            x = degree_nndegree.keys()
            y = degree_nndegree.values()
            plt.xlabel("degree(%s)"%target)
            plt.ylabel("nearest neighbors' degree")
            plt.title("nearest neighbors' degree-degree distribution")

        else:
            print "data_type arg error !"
            sys.exit()
        # print analysis_data
        plt.plot(x, y, "g-",linestyle="-")
        if save == "on":
            try:
                plt.savefig(filepath+"%s_%s.png"%(data_type, target))
            except Exception, e:
                print e
        elif save == "off":
            pass
        else:
            print "save arg error !"
        plt.show()

    def store_data(self, data, filepath):
        try:
            with open(filepath, 'w') as f:
                f.write(data)
        except Exception, e:
            print "store datas error !"
            print e
            return -1
        f.close()
        return 0

    def read_data(self, filepath):
        try:
            with open(filepath, 'r') as f:
                data = f.read()
        except Exception, e:
            print "read datas error !"
            print e
            data = ""
        f.close()
        return data

if __name__ == '__main__':
    data_analysis = DataAnalysis(filepath="../../data/k_5_2/sample_fengniao.txt")
    
    t0 = time.clock()
    similarity = data_analysis.creat_sim_matrix("item", "online")
    t1 = time.clock()
    print "creat_sim_matrix time costs"
    print t1-t0

    # t0 = time.clock()
    # dd = data_analysis.degree_analysis("user", "online")
    # t1 = time.clock()
    # print "degree_analysis time costs"
    # print t1-t0

    # t0 = time.clock()
    # col_sim = data_analysis.col_sim_analysis(similarity, "item", "online")
    # t1 = time.clock()
    # print "col_sim_analysis time costs"
    # print t1-t0
    # t0 = time.clock()

    # nndegree = data_analysis.nearest_neighbor_degree_analysis("user", "online")
    # t1 = time.clock()
    # print "nearest_neighbor_degree_analysis time costs"
    # print t1-t0
    # data_analysis.draw_graph(col_sim, "col_sim", "item", "on")

