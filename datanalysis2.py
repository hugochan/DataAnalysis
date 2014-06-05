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
        filepath = "./offline_results2/ui_matrix"
        if method == "online":
            self.ui_matrix = sparse.lil_matrix((self.itemnum, self.usernum))
            user_index = 0
            for user, record in self.instanceSet.iteritems():
                for eachrecord in record:
                    self.ui_matrix[self.itemset[eachrecord[0]], user_index] = 1
                user_index += 1
            try:
                # io.mmwrite(filepath, self.ui_matrix)
                io.savemat(filepath, {"ui_matrix":self.ui_matrix}, oned_as='row')
            except Exception, e:
                print e
                sys.exit()
        elif method == "offline":
            try:
                # self.ui_matrix = io.mmread(filepath)
                self.ui_matrix = io.loadmat(filepath, mat_dtype=False)["ui_matrix"]
            except Exception,e:
                print e
                sys.exit()
        else:
            print "method arg error !"
            sys.exit()

    def create_sim_matrix(self, block_length, method="online"):
        similarity = super(DataAnalysis2, self).create_sim_matrix(target="item", block_length=block_length, method=method)
        return similarity

    def calc_interitems_baseline(self, similarity, sampling_ration=1, method="online"):
        filepath = "./offline_results2/interitems_baseline"
        if method == "online":
            ave_sim = 0.0
            if sampling_ration == 1:
                pass
            elif sampling_ration > 0 and sampling_ration < 1:
                sampling_num = int(sampling_ration*self.instancenum*(self.instancenum - 1)/2)
                print "sampling_num: %s"%sampling_num
                
                i = 0
                while i < sampling_num:
                    item1, item2 = random.sample(self.item_records, 2)
                    item1_index = self.itemset[item1]
                    item2_index = self.itemset[item2]
                    ave_sim += (item1_index <= item2_index) and similarity[item1_index, item2_index] or similarity[item2_index, item1_index]
                    i += 1
                ave_sim /= sampling_num
                print "ave_sim: %s"%ave_sim
            else:
                print "sampling_ration arg error !"
                sys.exit()

            try:
                self.store_data(json.dumps(ave_sim, ensure_ascii=False), filepath+"_%s.json"%sampling_ration)
            except Exception, e:
                print e
                sys.exit()
            return ave_sim
            
        elif method == "offline":
            ave_sim = self.read_data(filepath+"_%s.json"%sampling_ration)
            if ave_sim != "":
                try:
                    ave_sim = json.loads(ave_sim)
                except Exception, e:
                    print e
                    sys.exit()
                return ave_sim
            else:
                print "read nothing !"
                sys.exit()
        else:
            print "method arg error !"
            sys.exit()
        

    def calc_user_interitems_baseline(self, similarity, sampling_ration=1, method="online"):
        filepath = "./offline_results2/user_interitems_baseline"
        if method == "online":
            if sampling_ration == 1:
                user_interitems = sparse.lil_matrix((self.usernum, 1))                
                user_index = 0
                for each_user, each_instance in self.instanceSet.iteritems():
                    ave_sim = 0.0
                    instancenum = len(each_instance)
                    if instancenum > 1:
                        for instance_index1 in np.arange(instancenum):
                            for instance_index2 in np.arange(instance_index1+1, instancenum):
                                item1_index = self.itemset[each_instance[instance_index1][0]]
                                item2_index = self.itemset[each_instance[instance_index2][0]]
                                ave_sim += (item1_index <= item2_index) and similarity[item1_index, item2_index] or similarity[item2_index, item1_index]
                        user_interitems[user_index, 0] = ave_sim/(instancenum*(instancenum-1)/2)
                    else:
                        user_interitems[user_index, 0] = 0
                    if user_index%100 == 0 and user_index != 0:
                        print "100 users done!"
                    user_index += 1

            elif sampling_ration > 0 and sampling_ration < 1:
                # user_index = 0
                # for each_user, each_instance in self.instanceSet.iteritems():
                #     ave_sim = 0.0
                #     instancenum = len(each_instance)
                #     if instancenum > 1:
                #         sampling_num = int(sampling_ration*instancenum*(instancenum - 1)/2)
                #         i = 0
                #         while i < sampling_num:
                #             instance1, instance2 = random.sample(each_instance, 2)
                #             item1_index = self.itemset[instance1[0]]
                #             item2_index = self.itemset[instance2[0]]
                #             ave_sim += (item1_index <= item2_index) and similarity[item1_index, item2_index] or similarity[item2_index, item1_index]
                #             i += 1
                #         if sampling_num > 0:
                #             user_interitems[user_index, 0] = ave_sim/sampling_num
                #         else:
                #             user_interitems[user_index, 0] = 0
                #     else:
                #         user_interitems[user_index, 0] = 0
                #     if user_index%100 == 0 and user_index != 0:
                #         print "100 users done!"
                #     user_index += 1
                sampling_num = int(sampling_ration*self.usernum)
                print "sampling_num: %s"%sampling_num
                user_interitems = sparse.lil_matrix((sampling_num, 1))                
                user_index = 0
                for each_user, each_instance in random.sample(self.instanceSet.items(), sampling_num):
                    ave_sim = 0.0
                    instancenum = len(each_instance)
                    if instancenum > 1:
                        for instance_index1 in np.arange(instancenum):
                            for instance_index2 in np.arange(instance_index1+1, instancenum):
                                item1_index = self.itemset[each_instance[instance_index1][0]]
                                item2_index = self.itemset[each_instance[instance_index2][0]]
                                ave_sim += (item1_index <= item2_index) and similarity[item1_index, item2_index] or similarity[item2_index, item1_index]
                        user_interitems[user_index, 0] = ave_sim/(instancenum*(instancenum-1)/2)
                    else:
                        user_interitems[user_index, 0] = 0
                    if user_index%100 == 0 and user_index != 0:
                        print "100 users done!"
                    user_index += 1

            else:
                print "sampling_ration arg error !"
                sys.exit()

            try:
                io.savemat(filepath+"_%s"%sampling_ration, {"user_interitems":user_interitems}, oned_as='row')
            except Exception, e:
                print e
                sys.exit()
            return user_interitems

        elif method == "offline":
            try:
                user_interitems = io.loadmat(filepath+"_%s"%sampling_ration, mat_dtype=False)["user_interitems"]
            except Exception,e:
                print e
                sys.exit()
            return user_interitems
        else:
            print "method arg error !"
            sys.exit()

    def calc_user_interitems_time(self, similarity, sampling_ration=1, method="online"):
        filepath = "./offline_results2/user_interitems_time"
        if method == "online":
            if sampling_ration == 1:
                calc_set =  self.instanceSet.items()
                max_relative_timespan = sele.get_max_relative_timespan(calc_set)
                user_interitems_time = sparse.lil_matrix((self.usernum, max_relative_timespan))


            elif sampling_ration > 0 and sampling_ration < 1:
                sampling_num = int(sampling_ration*self.usernum)
                print "sampling_num: %s"%sampling_num
                calc_set =  random.sample(self.instanceSet.items(), sampling_num)
                max_relative_timespan = sele.get_max_relative_timespan(calc_set)
                user_interitems_time = sparse.lil_matrix((sampling_num, max_relative_timespan))


            else:
                print "sampling_ration arg error !"
                sys.exit()

            user_index = 0
            for each_user, each_instance in calc_set:
                instancenum = len(each_instance)
                if instancenum > 1:
                    timespan_count = sparse.lil_matrix((1, max_relative_timespan))
                    for instance_index1 in np.arange(instancenum):
                        for instance_index2 in np.arange(instance_index1+1, instancenum):
                            timespan = instance_index2 - instance_index1 - 1
                            item1_index = self.itemset[each_instance[instance_index1][0]]
                            item2_index = self.itemset[each_instance[instance_index2][0]]
                            pdb.set_trace()
                            
                            timespan_count[0, timespan] += 1
                            user_interitems_time[user_index, timespan] += (item1_index <= item2_index) and similarity[item1_index, item2_index] or similarity[item2_index, item1_index]
                    # pdb.set_trace()
                    user_interitems_time[user_index, :] = user_interitems_time[user_index, :]/timespan_count[0, :] 
                else:
                    user_interitems_time[user_index, :] = 0
                if user_index%100 == 0 and user_index != 0:
                    print "100 users done!"
                user_index += 1 

            try:
                io.savemat(filepath+"_%s"%sampling_ration, {"user_interitems_time":user_interitems_time}, oned_as='row')
            except Exception, e:
                print e
                sys.exit()
            return user_interitems_time
            
        elif method == "offline":
            try:
                user_interitems_time = io.loadmat(filepath+"_%s"%sampling_ration, mat_dtype=False)["user_interitems_time"]
            except Exception,e:
                print e
                sys.exit()
            return user_interitems_time

        else:
            print "method arg error !"
            sys.exit()

    def get_max_relative_timespan(self, calc_set):
        max_relative_timespan = 0
        for user, record in calc_set.iteritems():
            temp = len(record) - 1
            if max_relative_timespan < temp:
                max_relative_timespan = temp
        return max_relative_timespan

if __name__ == '__main__':
    datanalysis2 = DataAnalysis2(filepath="../../data/fengniao/fengniao_filtering_0604.txt")
    datanalysis2.import_data()
    datanalysis2.create_ui_matrix("offline")
    t0 = time.clock()
    similarity = datanalysis2.create_sim_matrix(block_length=5000, method="offline")
    t1 = time.clock()
    print "create_sim_matrix costs: %ss"%(t1 - t0)

    # t0 = time.clock()
    # sampling_ration = 0.0002
    # datanalysis2.calc_interitems_baseline(similarity, sampling_ration=sampling_ration, method="online")
    # t1 = time.clock()
    # print "sampling_ration: %s"%sampling_ration
    # print t1 - t0

    # t0 = time.clock()
    # sampling_ration = 0.1
    # datanalysis2.calc_user_interitems_baseline(similarity, sampling_ration=sampling_ration, method="online")
    # t1 = time.clock()
    # print "sampling_ration: %s"%sampling_ration
    # print "calc_user_interitems_baseline costs: %ss"%(t1 - t0)

    t0 = time.clock()
    sampling_ration = 0.1
    datanalysis2.calc_user_interitems_time(similarity, sampling_ration=sampling_ration, method="online")
    t1 = time.clock()
    print "sampling_ration: %s"%sampling_ration
    print "calc_user_interitems_time costs: %ss"%(t1 - t0)
    