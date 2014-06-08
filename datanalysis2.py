#!/usr/bin/env python
#encoding=utf-8

import sys, json, random
from math import log
import numpy as np
from scipy import sparse
from scipy import io
import matplotlib.pyplot as plt
from matplotlib import cm, colors, axes
# from matplotlib import LinearLocator
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
                    temp_instanceSet = {}
                    while(templine):
                        self.instancenum += 1
                        temp = templine.split('\t')[:3]
                        user = int(temp[0])
                        item = int(temp[1])
                        time = int(temp[2][:-1])
                        self.item_records.append(item)
                        try:
                            temp_instanceSet[user].append([item, time])
                        except:
                            temp_instanceSet[user] = [[item, time]]
                        templine = f.readline()
            except Exception, e:
                print "import datas error !"
                print e
                sys.exit()
            f.close()
            temp_userset = temp_instanceSet.keys()
            temp_itemset = list(set(self.item_records))# remove redundancy
            self.usernum = len(temp_userset)
            self.itemnum = len(temp_itemset)
            for user_index in range(self.usernum):
                self.userset[temp_userset[user_index]] = user_index
            for item_index in range(self.itemnum):
                self.itemset[temp_itemset[item_index]] = item_index

            # replace the key of instanceSet with user_index
            for k, v in temp_instanceSet.iteritems():
                self.instanceSet[self.userset[k]] = v

            print "user num: %s"%self.usernum
            print "item num: %s"%self.itemnum
            print "instance num: %s"%self.instancenum

    def create_ui_matrix(self, method="online"):
        filepath = "./offline_results2/ui_matrix"
        if method == "online":
            self.ui_matrix = sparse.lil_matrix((self.itemnum, self.usernum))
            for user, record in self.instanceSet.iteritems():
                for eachrecord in record:
                    self.ui_matrix[self.itemset[eachrecord[0]], user] = 1
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
                for each_user, each_instance in self.instanceSet.iteritems():
                    ave_sim = 0.0
                    instancenum = len(each_instance)
                    if instancenum > 1:
                        for instance_index1 in np.arange(instancenum):
                            for instance_index2 in np.arange(instance_index1+1, instancenum):
                                item1_index = self.itemset[each_instance[instance_index1][0]]
                                item2_index = self.itemset[each_instance[instance_index2][0]]
                                ave_sim += (item1_index <= item2_index) and similarity[item1_index, item2_index] or similarity[item2_index, item1_index]
                        user_interitems[each_user, 0] = ave_sim/(instancenum*(instancenum-1)/2)
                    else:
                        user_interitems[each_user, 0] = 0
                    if each_user%100 == 0 and each_user != 0:
                        print "100 users done!"

            elif sampling_ration > 0 and sampling_ration < 1:
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
                sampling_num = self.usernum

            elif sampling_ration > 0 and sampling_ration < 1:
                sampling_num = int(sampling_ration*self.usernum)
                print "sampling_num: %s"%sampling_num
                calc_set =  random.sample(self.instanceSet.items(), sampling_num)

            else:
                print "sampling_ration arg error !"
                sys.exit()
            
            relative_timespan = self.get_relative_timespan(calc_set)
            calc_set = dict(calc_set)# to make sure calc_set and relative_timespan have the same key order
            max_relative_timespan = max(relative_timespan.values())
            user_interitems_time = sparse.lil_matrix((sampling_num, max_relative_timespan))

            user_index = 0
            for each_user, each_instance in calc_set.iteritems():
                instancenum = len(each_instance)
                if instancenum > 1:
                    timespan_count = sparse.lil_matrix((1, max_relative_timespan))
                    for instance_index1 in np.arange(instancenum):
                        for instance_index2 in np.arange(instance_index1+1, instancenum):
                            timespan = instance_index2 - instance_index1 - 1
                            item1_index = self.itemset[each_instance[instance_index1][0]]
                            item2_index = self.itemset[each_instance[instance_index2][0]]
                            
                            timespan_count[0, timespan] += 1
                            user_interitems_time[user_index, timespan] += (item1_index <= item2_index) and similarity[item1_index, item2_index] or similarity[item2_index, item1_index]
                    user_interitems_time[user_index, :] = user_interitems_time[user_index, :]/timespan_count[0, :] 
                    
                else:
                    user_interitems_time[user_index, :] = 0
                if user_index%10 == 0 and user_index != 0:
                    print user_index
                user_index += 1 

            relative_timespan = relative_timespan.items()# I can`t believe the order of a dict, so transfer to a list
            try:
                io.savemat(filepath+"_%s"%sampling_ration, {"user_interitems_time":user_interitems_time}, oned_as='row')
                # io.mmwrite(filepath+"_%s"%sampling_ration, user_interitems_time)
                self.store_data(json.dumps(relative_timespan, ensure_ascii=False), filepath+"_relative_timespan_%s.json"%sampling_ration)
            except Exception, e:
                print e
                sys.exit()
            return [user_interitems_time, relative_timespan]
            
        elif method == "offline":
            try:
                user_interitems_time = io.loadmat(filepath+"_%s"%sampling_ration, mat_dtype=False)["user_interitems_time"]
                # user_interitems_time = io.mmread(filepath+"_%s"%sampling_ration)
                relative_timespan = self.read_data(filepath+"_relative_timespan_%s.json"%sampling_ration)
                if relative_timespan != "":
                    try:
                        relative_timespan = json.loads(relative_timespan)
                    except Exception, e:
                        print e
                        sys.exit()
                else:
                    print "read nothing !"
                    sys.exit()
            except Exception,e:
                print e
                sys.exit()
            return [user_interitems_time, relative_timespan]

        else:
            print "method arg error !"
            sys.exit()

    def calc_userdegree_time_similarity(self, analysis_data, time_type="relative_time"):#?????????
        user_interitems_time, relative_timespan = analysis_data
        user_interitems_time = user_interitems_time.tocsr()
        usernum, max_timespan = user_interitems_time.shape
        user_index = []
        for each in relative_timespan:
            user_index.append(each[0])
        relative_timespan = dict(relative_timespan)
        degree = np.asarray(self.ui_matrix.sum(0)[0, user_index])[0].tolist()
        degree_time_similarity = {}
        degree_userindex = {}

        for each in np.arange(usernum):
            try:
                degree_time_similarity[degree[each]] = degree_time_similarity[degree[each]] + user_interitems_time[each, :]
                degree_userindex[degree[each]].append(user_index[each])
            except:
                degree_time_similarity[degree[each]] = user_interitems_time[each, :]
                degree_userindex[degree[each]] = [user_index[each]]


        for eachdegree in set(degree):
            timespan_usercount = np.ones([1, max_timespan])*len(degree_userindex[eachdegree])
            temp_max_timespan = 0
            for eachuser in degree_userindex[eachdegree]:
                timespan_usercount[0, relative_timespan[eachuser]:] -= 1
                if temp_max_timespan < relative_timespan[eachuser]:
                    temp_max_timespan = relative_timespan[eachuser]
            degree_time_similarity[eachdegree] = (degree_time_similarity[eachdegree]/sparse.csc_matrix(timespan_usercount)).toarray()[0].tolist()
            
        return degree_time_similarity

    def get_relative_timespan(self, calc_set):
        """get relative timespan for each user in the calc_set"""
        relative_timespan = {}
        for user, record in calc_set:
            relative_timespan[user] = len(record) - 1
        return relative_timespan

    def draw_graph(self, analysis_data, data_type, time_type="relative_time", save="off"):
        filepath = "./image2/"
        if data_type == "similarity_time":
            plt.figure()
            user_interitems_time = sparse.csc_matrix(analysis_data[2][0].sum(0))
            relative_timespan = dict(analysis_data[2][1])
            usernum, max_timespan = analysis_data[2][0].shape
            timespan_usercount = np.ones([1, max_timespan])*usernum
            for eachuser_timespan in relative_timespan.values():
                timespan_usercount[0, eachuser_timespan:] -= 1
            user_interitems_time = (user_interitems_time/sparse.csc_matrix(timespan_usercount)).toarray()[0].tolist()

            time = range(max_timespan+1)[1:]
            temp = analysis_data[0]
            interitems_baseline = [temp for each in time]
            temp = analysis_data[1].mean()
            user_interitems_baseline = [temp for each in time]# user_interitems

            plt.xlabel("relative time")
            plt.ylabel("similarity")
            plt.title("similarity_time curves")
            plt.plot(time, interitems_baseline, "b-", label="interitems_baseline")
            plt.plot(time, user_interitems_baseline, "g-", label="user_interitems_baseline")
            plt.plot(time, user_interitems_time, "r-", label="user_interitems_time")

        elif data_type == "userdegree_time_similarity":
            degree_time_similarity = self.calc_userdegree_time_similarity(analysis_data[2], time_type="relative_time")
            
            max_timespan = analysis_data[2][0].shape[1]
            time = range(max_timespan+1)[1:]
            degree = degree_time_similarity.keys()
            sim = np.array(degree_time_similarity.values())
            print "degree_len: %s"%len(degree)
            print "max_timespan: %s"%max_timespan

            fig = plt.figure(facecolor='w')
            ax1 = fig.add_subplot(1,1,1, position=[0.1,0.15,0.9,0.8])
            cmap = cm.get_cmap('spectral', 100000)
            map = ax1.imshow(sim, interpolation="nearest", cmap=cmap, aspect='auto', vmin=sim.min(), vmax=sim.max())
            cb = plt.colorbar(mappable=map, cax=None, ax=None, shrink=0.5)
            cb.set_label('(heat)')
            
            plt.xlabel("relative time")
            plt.ylabel("degree")
            plt.title("userdegree_time_similarity heatmap")

        
        else:
            print "data_type arg error !"
            sys.exit()
        if save == "on":
            try:
                plt.savefig(filepath+"%s.png"%data_type)
            except Exception, e:
                print e
        elif save == "off":
            pass
        else:
            print "save arg error !"
        plt.show()


if __name__ == '__main__':
    datanalysis2 = DataAnalysis2(filepath="../../data/fengniao/fengniao_filtering_0604.txt")
    datanalysis2.import_data()
    datanalysis2.create_ui_matrix("offline")
    # t0 = time.clock()
    # similarity = datanalysis2.create_sim_matrix(block_length=5000, method="offline")
    # t1 = time.clock()
    # print "create_sim_matrix costs: %ss"%(t1 - t0)

    # t0 = time.clock()
    # sampling_ration = 0.0002
    # interitems = datanalysis2.calc_interitems_baseline(similarity=0, sampling_ration=sampling_ration, method="offline")
    # t1 = time.clock()
    # print "sampling_ration: %s"%sampling_ration
    # print "calc_interitems_baseline costs: %ss"%(t1 - t0)

    # t0 = time.clock()
    # sampling_ration = 0.1
    # user_interitems = datanalysis2.calc_user_interitems_baseline(similarity=0, sampling_ration=sampling_ration, method="offline")
    # t1 = time.clock()
    # print "sampling_ration: %s"%sampling_ration
    # print "calc_user_interitems_baseline costs: %ss"%(t1 - t0)

    t0 = time.clock()
    sampling_ration = 0.001
    uit = datanalysis2.calc_user_interitems_time(similarity=0, sampling_ration=sampling_ration, method="offline")
    t1 = time.clock()
    print "sampling_ration: %s"%sampling_ration
    print "calc_user_interitems_time costs: %ss"%(t1 - t0)
    datanalysis2.draw_graph(analysis_data=[0, 0, uit], data_type="userdegree_time_similarity", time_type="relative_time", save="on")
