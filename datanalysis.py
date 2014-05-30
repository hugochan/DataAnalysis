#!/usr/bin/env python
#encoding=utf-8

import numpy as np
import sys, json, os
import matplotlib.pyplot as plt
import pdb, time

class DataAnalysis(object):
    """do some data analysis"""
    def __init__(self, filepath):
        super(DataAnalysis, self).__init__()
        self.__filepath = filepath
        self.__userset = []
        self.__itemset = []
        self.__instanceSet = []
        self._import_data()
        self._create_ui_matrix()

    def _import_data(self):
        try:
            with open(self.__filepath, 'r') as f:
                templine = f.readline()
                instancenum = 1
                while(templine):
                    temp = templine.split('\t')[:2]
                    user = int(temp[0])
                    item = int(temp[1][:-1])
                    self.__userset.append(user)
                    self.__itemset.append(item)
                    self.__instanceSet.append([user, item])
                    try:    
                        templine = f.readline()
                        instancenum += 1
                    except Exception, e:
                        print 'read error'
                        print e
        except Exception, e:
            print "import datas error !"
            print e
            pdb.set_trace()
            sys.exit()
        f.close()
        self.__userset = list(set(self.__userset))# remove redundancy
        self.__itemset = list(set(self.__itemset))# remove redundancy
        self.__usernum = len(self.__userset)
        self.__itemnum = len(self.__itemset)
        print "user num:"
        print len(self.__userset)
        print "item num:"
        print len(self.__itemset)
        print "instance num:"
        print len(self.__instanceSet)

    def _create_ui_matrix(self, method="online"):
        if method == "online":
            self.ui_matrix = np.zeros([self.__usernum, self.__itemnum], dtype=np.float)
            # pdb.set_trace()
            for each_instance in self.__instanceSet:# exacting
                self.ui_matrix[self.__userset.index(each_instance[0])][self.__itemset.index(each_instance[1])] = 1.0
            self.store_data(json.dumps(self.ui_matrix.tolist(), ensure_ascii=False),\
                    "./offline_results/ui_matrix.data")
        elif method == "offline":
            self.ui_matrix = self.read_data("./offline_results/ui_matrix.data")
            if self.ui_matrix != "":
                try:
                    self.ui_matrix = json.loads(self.ui_matrix)
                except Exception, e:
                    print e
                    sys.exit()
                self.ui_matrix = np.array(self.ui_matrix)
            else:
                sys.exit()
        else:
            print "method arg error !"
            sys.exit()


    def creat_sim_matrix(self, target, method="online"):
        """Jaccard similarity
        target = 'user' means similarity among users, target = 'item' means similarity among items.
        method = 'online' means online calculation, method = 'offline' means using offline results.
        """
        if method == "online":# online calculation
            tinynum = 0.00000001
            if target == "user":# for user
                # intersection = np.dot(np.transpose(self.ui_matrix), self.ui_matrix)# shape: (usernum, usernum)
                unionsection = np.zeros([self.__usernum, self.__usernum], dtype=np.float)
                for u in np.arange(self.__usernum):
                    temp = np.transpose(self.ui_matrix[:, u]*np.ones([self.__usernum-u, self.__itemnum]))# shape: (itemnum, usernum-u)
                    # sum plus intersection equals to unionsection, then plus a tiny num to avoid zero division
                    unionsection[u, u:] = (temp + self.ui_matrix[:, u:]).sum(0)\
                        - intersection[u:, u] + np.ones([self.__usernum-u, ])*tinynum
                unionsection = unionsection + np.transpose(unionsection)# to make a full matrix using symmetry(trace to be done!!)
                # similarity = intersection/unionsection
                similarity = np.dot(np.transpose(self.ui_matrix), self.ui_matrix)/unionsection# to save space
            elif target == "item":# for item
                # intersection = np.dot(self.ui_matrix, np.transpose(self.ui_matrix))# shape: (itemnum, itemnum)
                unionsection = np.zeros([self.__itemnum, self.__itemnum], dtype=np.float)
                for o in np.arange(self.__itemnum):
                    temp = self.ui_matrix[o, :]*np.ones([self.__itemnum-o, self.__usernum])# shape: (itemnum-o, usernum)
                    unionsection[o:, o] = (temp + self.ui_matrix[o:, :]).sum(1)\
                        - intersection[o:, o] + np.ones([self.__itemnum-o, ])*tinynum
                unionsection = unionsection + np.transpose(unionsection)
                # similarity = intersection/unionsection
                similarity = np.dot(self.ui_matrix, np.transpose(self.ui_matrix))/unionsection# to save space
            else:
                print "target arg error !"
                sys.exit()
            if target == "user" or target == "item":
                self.store_data(json.dumps(similarity.tolist(), ensure_ascii=False),\
                    "./offline_results/similarity_%s.data"%target)
                return similarity
        elif method == "offline":# using offline results
            if target == "user" or target == "item":
                similarity = self.read_data("./offline_results/similarity_%s.data"%target)
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

    def degree_analysis(self, target, method="online"):
        """degree distribution
        target = 'user' means degree distribution for users,
        target = 'item' means degree distribution for items.
        method = 'online' means online calculation,
        method = 'offline' means using offlinie results."""
        if method == "online":# online calculation
            degree_distribution = {}
            if target == "user":# for users
                degree = self.ui_matrix.sum(0).tolist()
                for eachdegree in set(degree):
                    degree_distribution[eachdegree] = degree.count(eachdegree)
            elif target == "item":# for items
                degree = self.ui_matrix.sum(1).tolist()
                for eachdegree in set(degree):
                    degree_distribution[eachdegree] = degree.count(eachdegree)
            else:
                print "target arg error !"
                sys.exit()
            if target == "user" or target == "item":
                self.store_data(json.dumps(degree_distribution, ensure_ascii=False),\
                    "./offline_results/degree_distribution_%s.data"%target)
                return degree_distribution
        elif method == "offline":# using offline results
            if target == "user" or target == "item":
                degree_distribution = self.read_data("./offline_results/degree_distribution_%s.data"%target)
                if degree_distribution != "":
                    try:
                        degree_distribution = json.loads(degree_distribution)
                    except Exception, e:
                        print e
                        sys.exit()
                    print degree_distribution.keys()
                    return degree_distribution
                else:
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
        if method == "online":# online calculation
            if target == "user":# for user
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
            elif target == "item":# for item
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
            if target == "user" or target == "item":
                self.store_data(json.dumps(col_sim.tolist(), ensure_ascii=False),\
                    "./offline_results/col_sim_%s.data"%target)
                return col_sim
        elif method == "offline":# using offline results
            if target == "user" or target == "item":
                col_sim = self.read_data("./offline_results/col_sim_%s.data"%target)
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

    def nearest_neighbor_degree_analysis(self, target, method="online"):
        """nearest neighbors' degree
        ref Empirical analysis of web-based user-object bipartite networks, Ming-Sheng Shang et al. 17 June 2010.
        target = 'user' means nearest neighbors' degree for users,
        target = 'item' means nearest neighbors' degree for items.        
        method = 'online' means online calculation,
        method = 'offline' means using offline results.        
        """
        if method == "online":# online calculation
            tinynum = 0.00000001
            if target == "user":# for user
                degree = self.ui_matrix.sum(0)
                degree += np.ones([self.__usernum, ])*tinynum# to avoid zero division
                nn_degree = (np.transpose(self.ui_matrix.sum(1)*np.ones([self.__usernum, self.__itemnum]))\
                        *self.ui_matrix).sum(0)/degree
            elif target == "item":# for item
                degree = self.ui_matrix.sum(1)
                degree += np.ones([self.__itemnum, ])*tinynum# to avoid zero division
                nn_degree = (self.ui_matrix.sum(0)*np.ones([self.__itemnum, self.__usernum])\
                        *self.ui_matrix).sum(1)/degree
            else:
                print "target arg error !"
                sys.exit()
            if target == "user" or target == "item":
                self.store_data(json.dumps(nn_degree.tolist(), ensure_ascii=False),\
                    "./offline_results/nn_degree_%s.data"%target)
                return nn_degree
        elif method == "offline":# using offline results
            if target == "user" or target == "item":
                nn_degree = self.read_data("./offline_results/nn_degree_%s.data"%target)
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

    def draw_graph(self, analysis_data, data_type, target):
        plt.figure()
        if data_type == "degree_distribution":# degree
            x = analysis_data.keys()
            y = analysis_data.values()
            plt.xlabel("degree(%s)"%target)
            plt.ylabel("frequency")
            plt.title("degree distribution")
        elif data_type == "col_sim":# collaborative similarity
            degree_col_sim = {}
            if target == "user":# for user
                degree = self.ui_matrix.sum(0).tolist()
                for eachdegree in set(degree):
                    count = degree.count(eachdegree)
                    index = -1
                    temp = 0.0
                    for i in np.arange(count):
                        index2 = degree[index+1:].index(eachdegree)
                        temp += analysis_data[index+1+index2]
                        index = index + 1 + index2
                    # pdb.set_trace()
                    degree_col_sim[eachdegree] = temp/count
            elif target == "item":# for item
                degree = self.ui_matrix.sum(1).tolist()
                for eachdegree in set(degree):
                    count = degree.count(eachdegree)
                    index = -1
                    temp = 0.0
                    for i in np.arange(count):
                        index2 = degree[index+1:].index(eachdegree)
                        temp += analysis_data[index+1+index2]
                        index = index + 1 + index2
                    # pdb.set_trace()
                    degree_col_sim[eachdegree] = temp/count
            else:
                print "target arg error !"
                sys.exit()
            x = degree_col_sim.keys()
            y = degree_col_sim.values()
            plt.xlabel("degree(%s)"%target)
            plt.ylabel("collaborative similarity")
            plt.title("collaborative similarity-degree distribution")
        elif data_type == "nn_degree":# nearest neighbors' degree
            pass
        else:
            print "data_type arg error !"
            sys.exit()
        # print analysis_data
        plt.plot(x, y, "g-",linestyle="-")
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
    t0 = time.clock()
    data_analysis = DataAnalysis(filepath="../../data/k_5_2/sample_fengniao.txt")
    similarity = data_analysis.creat_sim_matrix("item", "online")
    # col_sim = data_analysis.col_sim_analysis(similarity, "user", "online")
    # nndegree = data_analysis.nearest_neighbor_degree_analysis(0, 1)
    # dd = data_analysis.degree_analysis(0, 1)
    t1 = time.clock()
    print t1-t0
    # data_analysis.draw_graph(dd, "degree_distribution", "user")
    # data_analysis.draw_graph(col_sim, "col_sim", "user")

