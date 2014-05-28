#!/usr/bin/env python
#encoding=utf-8

import numpy as np
import sys, json, os
import matplotlib.pyplot as plt
import pdb, time

class DataAnalysis(object):
    """do some data analysis"""
    def __init__(self, filepath="../../data/fengniao_test.txt"):
        super(DataAnalysis, self).__init__()
        self.__filepath = filepath
        self.__userset = []
        self.__itemset = []
        self.__instanceSet = []
        self._import_data()
        self.create_ui_matrix()

    def _import_data(self):
        try:
            with open(self.__filepath, 'r') as f:
                templine = f.readline()
                while(templine):
                    temp = templine.split(' ')[:2]

                    self.__userset.append(temp[0])
                    self.__itemset.append(temp[1])
                    self.__instanceSet.append(temp)
                    templine = f.readline()
        except Exception, e:
            print "import datas error !"
            print e
            sys.exit()
        f.close()
        self.__usernum = len(self.__userset)
        self.__itemnum = len(self.__itemset)
        # print "user num:"
        # print len(self.__userset)
        # print "item num:"
        # print len(self.__itemset)
        # print "instance num:"
        # print len(self.__instanceSet)

    def create_ui_matrix(self):
        self.ui_matrix = np.zeros([self.__usernum, self.__itemnum], dtype=np.float)
        for each_instance in self.__instanceSet:
            self.ui_matrix[self.__userset.index(each_instance[0])][self.__itemset.index(each_instance[1])] = 1.0

    def creat_sim_matrix(self, target=0, method=0):
        """Jaccard similarity
        target = 0 means similarity among users, target = 1 means similarity among items.
        method = 0 means online calculation, method = 1 means using offline results.
        """
        if method == 0:# online calculation
            tinynum = 0.00000001
            if target == 0:# for user
                intersection = np.dot(np.transpose(self.ui_matrix), self.ui_matrix)# shape: (usernum, usernum)
                unionsection = np.zeros([self.__usernum, self.__usernum], dtype=np.float)
                for u in np.arange(self.__usernum):
                    temp = np.transpose(self.ui_matrix[:, u]*np.ones([self.__usernum-u, self.__itemnum]))# shape: (itemnum, usernum-u)
                    unionsection[u, u:] = (temp + self.ui_matrix[:, u:]).sum(0) - intersection[u:, u]# sum plus intersection equals to unionsection
                unionsection = unionsection + np.transpose(unionsection)# to make a full matrix using symmetry
                unionsection += np.ones([self.__usernum, self.__usernum])*tinynum# to avoid zero division
                similarity = intersection/unionsection
            elif target == 1:# for item
                intersection = np.dot(self.ui_matrix, np.transpose(self.ui_matrix))# shape: (itemnum, itemnum)
                unionsection = np.zeros([self.__itemnum, self.__itemnum], dtype=np.float)
                for o in np.arange(self.__itemnum):
                    temp = self.ui_matrix[o, :]*np.ones([self.__itemnum-o, self.__usernum])# shape: (itemnum-o, usernum)
                    unionsection[o:, o] = (temp + self.ui_matrix[o:, :]).sum(1) - intersection[o:, o]# sum plus intersection equals to unionsection
                unionsection = unionsection + np.transpose(unionsection)# to make a full matrix using symmetry
                unionsection += np.ones([self.__itemnum, self.__itemnum])*tinynum# to avoid zero division
                similarity = intersection/unionsection
            else:
                print "target arg error !"
                sys.exit()
            if target == 0 or target == 1:
                self.store_data(json.dumps(similarity.tolist(), ensure_ascii=False),\
                    "./offline_results/similarity_%s.data"%(target==0 and "user" or "item"))
                return similarity
        elif method == 1:# using offline results
            if target == 0 or target == 1:
                similarity = self.read_data("./offline_results/similarity_%s.data"%(target==0 and "user" or "item"))
                if similarity != "":
                    try:
                        similarity = json.loads(similarity)
                    except Exception, e:
                        print e
                        sys.exit()
                    similarity = np.array(similarity)
                    return similarity
                else:
                    sys.exit()
            else:
                print "target arg error !"
                sys.exit()
        else:
            print "method arg error !"
            sys.exit()

    def degree_analysis(self):
        pass
    def col_sim_analysis(self, similarity, target=0, method=0):
        """collaborative similarity
        ref Empirical analysis of web-based user-object bipartite networks, Ming-Sheng Shang et al. 17 June 2010.
        target = 0 means collaborative similarity for users, target = 1 means collaborative similarity for items.
        method = 0 means online calculation, method = 1 means using offline results.
        """
        if method == 0:# online calculation
            if target == 0:# for user
                col_sim = np.zeros([self.__usernum, ])
                degree = self.ui_matrix.sum(0)# shape: (usernum, )
                for u in np.arange(self.__usernum):
                    # construct a similarity filtering matrix
                    temp = self.ui_matrix[:, u]*np.ones([self.__itemnum, self.__itemnum])
                    filtering_matrix = temp*np.transpose(temp)# shape (itemnum, itemnum)
                    similarity = similarity*filtering_matrix

                    temp = degree[u]*(degree[u] - 1)
                    if temp == 0:
                        temp = 1
                    col_sim[u] = (np.sum(similarity)-np.trace(similarity))/temp
            elif target == 1:# for item
                col_sim = np.zeros([self.__itemnum, ])
                degree = self.ui_matrix.sum(1)# shape: (itemnum, )
                for o in np.arange(self.__itemnum):
                    # construct a similarity filtering matrix
                    temp = self.ui_matrix[o, :]*np.ones([self.__usernum, self.__usernum])
                    filtering_matrix = temp*np.transpose(temp)# shape (usernum, usernum)
                    similarity = similarity*filtering_matrix
                    
                    temp = degree[o]*(degree[o] - 1)
                    if temp == 0:
                        temp = 1
                    col_sim[o] = (np.sum(similarity)-np.trace(similarity))/temp
            else:
                print "target arg error !"
                sys.exit()
            if target == 0 or target == 1:
                self.store_data(json.dumps(col_sim.tolist(), ensure_ascii=False),\
                    "./offline_results/col_sim_%s.data"%(target==0 and "user" or "item"))
                return col_sim
        elif method == 1:# using offline results
            if target == 0 or target == 1:
                col_sim = self.read_data("./offline_results/col_sim_%s.data"%(target==0 and "user" or "item"))
                if col_sim != "":
                    try:
                        col_sim = json.loads(col_sim)
                    except Exception, e:
                        print e
                        sys.exit()
                    col_sim = np.array(col_sim)
                    return col_sim
                else:
                    sys.exit()
            else:
                print "target arg error !"
                sys.exit()
        else:
            print "method arg error !"
            sys.exit()

    def nearest_neighbor_degree_analysis(self, target=0, method=0):
        """nearest neighbors' degree
        ref Empirical analysis of web-based user-object bipartite networks, Ming-Sheng Shang et al. 17 June 2010.
        target = 0 means nearest neighbors' degree for users,
        target = 1 means nearest neighbors' degree for items.        
        method = 0 means online calculation,
        method = 1 means using offline results.        
        """
        if method == 0:# online calculation
            tinynum = 0.00000001
            if target == 0:# for user
                degree = self.ui_matrix.sum(0)
                degree += np.ones([self.__usernum, ])*tinynum# to avoid zero division
                nn_degree = (np.transpose(self.ui_matrix.sum(1)*np.ones([self.__usernum, self.__itemnum]))\
                        *self.ui_matrix).sum(0)/degree
            elif target == 1:# for item
                degree = self.ui_matrix.sum(1)
                degree += np.ones([self.__itemnum, ])*tinynum# to avoid zero division
                nn_degree = (self.ui_matrix.sum(0)*np.ones([self.__itemnum, self.__usernum])\
                        *self.ui_matrix).sum(1)/degree
            else:
                print "target arg error !"
                sys.exit()
            if target == 0 or target == 1:
                self.store_data(json.dumps(nn_degree.tolist(), ensure_ascii=False),\
                    "./offline_results/nn_degree_%s.data"%(target==0 and "user" or "item"))
                return nn_degree
        elif method == 1:# using offline results
            if target == 0 or target == 1:
                nn_degree = self.read_data("./offline_results/nn_degree_%s.data"%(target==0 and "user" or "item"))
                if nn_degree != "":
                    try:
                        nn_degree = json.loads(nn_degree)
                    except Exception, e:
                        print e
                        sys.exit()
                    nn_degree = np.array(nn_degree)
                    return nn_degree
                else:
                    sys.exit()
            else:
                print "target arg error !"
                sys.exit()
        else:
            print "method arg error !"
            sys.exit()

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
    t0 = time.clock()
    data_analysis = DataAnalysis()
    similarity = data_analysis.creat_sim_matrix(1, 1)
    # col_sim = data_analysis.col_sim_analysis(similarity, 0, 1)
    # nndegree = data_analysis.nearest_neighbor_degree_analysis(0, 1)
    t1 = time.clock()
    print t1-t0