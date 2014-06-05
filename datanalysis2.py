#!/usr/bin/env python
#encoding=utf-8

import sys, json, random
from math import log
import numpy as np
from scipy import sparse
from scipy import io
import matplotlib.pyplot as plt
import pdb, time
from datanalysis import DataAnalysis

class DataAnalysis2(DataAnalysis):
    """A general framework for analysis of user-item behavior patterns considering time factor,
        inheriting from DataAnalysis class
    """
    def __init__(self, filepath):
        super(DataAnalysis2, self).__init__(filepath)
        self.item_records = []

    def import_data(self):
            try:
                with open(self._filepath, 'r') as f:
                    temp_itemset = []# item set       
                    templine = f.readline()
                    while(templine):
                        self.instancenum += 1
                        temp = templine.split('\t')[:3]
                        user = int(temp[0])
                        item = int(temp[1])
                        time = int(temp[2][:-1])
                        self.item_records.append(item)
                        try:
                            self.instanceSet[user].append([item, time])
                        except:
                            self.instanceSet[user] = [[item, time]]
                        templine = f.readline()
            except Exception, e:
                print "import datas error !"
                print e
                sys.exit()
            f.close()
            self.userset = self.instanceSet.keys()
            temp_itemset = list(set(self.item_records))# remove redundancy
            self.usernum = len(self.userset)
            self.itemnum = len(temp_itemset)
            for item_index in np.arange(self.itemnum):
                self.itemset[temp_itemset[item_index]] = item_index

            print "user num: %s"%self.usernum
            print "item num: %s"%self.itemnum
            print "instance num: %s"%self.instancenum

    def create_ui_matrix(self, method="online"):
    	filepath = "./offline_results/ui_matrix"
        if method == "online":
            self.ui_matrix = sparse.lil_matrix((self.itemnum, self.usernum))
            user_index = 0
            for user, record in self.instanceSet.iteritems():
                for eachrecord in record:
                    self.ui_matrix[self.itemset[eachrecord[0]], user_index] = 1
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

    def create_sim_matrix(self, block_length, method="online"):
        super(DataAnalysis2, self).create_sim_matrix(target="item", block_length=block_length, method=method)

    def calc_interitems_baseline(self, sampling_ration=1, method="online"):
            if method == "online":
                similarity = sparse.lil_matrix((self.itemnum, self.itemnum))# simulate
                ave_sim = 0.0
                if sampling_ration == 1:
                    pass
                else:
                    sampling_num = sampling_ration*self.instancenum*(self.instancenum-1)/2
                    print sampling_num
                    for each in np.arange(sampling_num):
                        item1, item2 = random.sample(self.item_records, 2)
                        item1_index = self.itemset[item1]
                        item2_index = self.itemset[item2]
                        ave_sim += (item1_index <= item2_index) and similarity[item1_index, item2_index] or similarity[item2_index, item1_index]
                    ave_sim /= sampling_num
                print ave_sim
                return ave_sim
            elif method == "offline":
                pass
            else:
                print "method arg error !"
                sys.exit()
        

    def calc_user_interitems_baseline(self, method="online"):
        pass

    def calc_user_interitems_time(self, method="online"):
        pass

if __name__ == '__main__':
    datanalysis2 = DataAnalysis2(filepath="../../data/fengniao/fengniao_filtering_0604.txt")
    datanalysis2.import_data()
    datanalysis2.create_ui_matrix("offline")
    t0 = time.clock()
    datanalysis2.create_sim_matrix(block_length=3000, method="online")
    t1 = time.clock()
    print t1 - t0

    # datanalysis2.calc_interitems_baseline(sampling_ration=0.0002, method="online")
