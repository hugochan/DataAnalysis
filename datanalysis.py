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
        self.ui_matrix = np.zeros([self.__usernum, self.__itemnum])
        for each_instance in self.__instanceSet:
            self.ui_matrix[self.__userset.index(each_instance[0])][self.__itemset.index(each_instance[1])] = 1

    def creat_sim_matrix(self, type=0, method=0):
        """Jaccard similarity
        type = 0 means similarity among users, type = 1 means similarity among items
        method = 0 means online calculation, method = 1 means using offline calculation
        """
        tinynum = 0.00000001
        if method == 0:# online calculation
            if type == 0:# for user
                intersection = np.dot(np.transpose(self.ui_matrix), self.ui_matrix)# shape: (usernum, usernum)
                unionsection = np.zeros([self.__usernum, self.__usernum], dtype=np.float)
                # performance to be improved
                # version 1
                # for i in np.arange(self.__usernum):
                #     for j in np.arange(i, self.__usernum):# to do
                #         unionsection[i, j] = np.sum(self.ui_matrix[:,i] + self.ui_matrix[:, j])
                #         unionsection[j, i] = np.sum(self.ui_matrix[:,i] + self.ui_matrix[:, j])
                # version 2
                # for i in np.arange(self.__usernum):
                #     temp = np.transpose(np.array([self.ui_matrix[:, i] for j in np.arange(i, self.__usernum)]))
                #     unionsection[i, i:] = (temp + self.ui_matrix[:, i:]).sum(0)# to be corrected
                # version 3
                for i in np.arange(self.__usernum):
                    temp = np.transpose(self.ui_matrix[:, i]*np.ones([self.__usernum-i, self.__itemnum]))# shape: (itemnum, usernum-i)
                    unionsection[i, i:] = (temp + self.ui_matrix[:, i:]).sum(0) - intersection[i:, i]# sum plus intersection equals to unionsection
                unionsection = unionsection + np.transpose(unionsection)# to make a full matrix using symmetry
                unionsection += np.ones([self.__usernum, self.__usernum])*tinynum# to avoid zero division
                similarity = intersection/unionsection
            elif type == 1:# for item
                intersection = np.dot(self.ui_matrix, np.transpose(self.ui_matrix))# shape: (itemnum, itemnum)
                unionsection = np.zeros([self.__itemnum, self.__itemnum], dtype=np.float)
                for i in np.arange(self.__itemnum):
                    temp = self.ui_matrix[i, :]*np.ones([self.__itemnum-i, self.__usernum])# shape: (itemnum-i, usernum)
                    unionsection[i:, i] = (temp + self.ui_matrix[i:, :]).sum(1) - intersection[i:, i]# sum plus intersection equals to unionsection
                unionsection = unionsection + np.transpose(unionsection)# to make a full matrix using symmetry
                unionsection += np.ones([self.__itemnum, self.__itemnum])*tinynum# to avoid zero division
                similarity = intersection/unionsection
            else:
                print "type arg error !"
                sys.exit()
            if type == 0 or type == 1:
                similarity_json = json.dumps(similarity.tolist(), ensure_ascii=False)
                self.store_data(similarity_json, "./offline_results/similarity_%s.data"%(type==0 and "user" or "item"))
                return similarity
        elif method == 1:# using offline calculation
            if type == 0 or type == 1:
                similarity = self.read_data("./offline_results/similarity_%s.data"%(type==0 and "user" or "item"))
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
                print "type arg error !"
                sys.exit()
        else:
            print "method arg error !"
            sys.exit()

    def degree_analysis(self):
        pass
    def col_sim_analysis(self, type=0):
        if type == 0:# for user
            pass            



        elif type == 1:
            pass
        else:
            print "arg error !"
            sys.exit()
        
    def nearest_neighbor_degree_analysis(self):
        pass

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
    similarity = data_analysis.creat_sim_matrix()
    print similarity.shape
    t1 = time.clock()
    print t1-t0