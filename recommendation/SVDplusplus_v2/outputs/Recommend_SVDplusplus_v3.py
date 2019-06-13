# -*- coding: utf-8 -*-
"""
Created on Wed May 29 12:11:45 2019

@author: lity3
"""

import random
import math
import pandas as pd
import numpy as np
import time
import tensorflow as tf
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    explained_variance_score,
    roc_auc_score,
    log_loss,
)

"""HYPERPARAMS"""
tf.app.flags.DEFINE_integer('latentFactorNum',200,'number of latent Factors to use')
tf.app.flags.DEFINE_integer('Epochs',1,'number of maximum epochs to train the model')
FLAGS = tf.app.flags.FLAGS

udata_path = 'datasets/ml-100k/u.data'
u1base_path = 'datasets/ml-100k/u1.base'
u1test_path = 'datasets/ml-100k/u1.test'
pred_matrix_path = 'outputs/SVDplusplus_item_mat.npy'

class SVDplusplus():
    def __init__(self, allfile, trainfile, testfile, latentFactorNum=20,alpha_u=0.01,alpha_v=0.01,alpha_w=0.01,beta_u=0.01,beta_v=0.01,learning_rate=0.01):
        data_fields = ['user_id', 'item_id', 'rating', 'timestamp']
        # all data file
        allData = pd.read_csv(udata_path,names=data_fields,sep='\t')
        user_list=sorted(set(allData['user_id'].values))
        item_list=sorted(set(allData['item_id'].values))
        # Do data shuffling with allData
        ua_base=allData.sample(n=len(allData),replace=False)
        self.test_df=pd.read_csv(testfile,names=data_fields,sep='\t')
        '''Evaluation measures'''
        self.RMSE=10000
        self.MAE=10000
        self.precision=0
        self.recall = 0
        self.iterations = 0
        self.timePerEpoch = 0
        # Split the ua_base into 2 parts.
        # ua_base_implicit only have (user_id,item_id)
        # ua_base_explicit have (user_id, item_id, ratings)
        self.ua_base_implicit=ua_base.sample(frac=0.5,replace=False)
        self.ua_base_explicit=ua_base.drop(self.ua_base_implicit.index,axis=0)

        # Transform the rating history dataframe(implicit and explicit part) to two user-item rating matrix
        self.implicit=self.ua_base_implicit.pivot(index='user_id', columns='item_id', values='rating')
        print(self.test_df.shape)
        print(self.ua_base_explicit.shape)
        print(self.ua_base_implicit.shape)

        data_df = pd.DataFrame(index=user_list, columns=item_list)
        rating_matrix=self.ua_base_explicit.pivot(index='user_id', columns='item_id', values='rating')
        data_df.update(rating_matrix)
        self.rating_matrix=data_df
        # training set file
        #self.train_df = pd.read_table(trainfile, names=data_fields)
        # testing set file
        self.real_test_df = pd.read_csv(testfile, names=data_fields, sep='\t')
        #self.test_df=pd.read_table(testfile, names=data_fields)
        # get factor number
        self.latentFactorNum = latentFactorNum
        # get user number
        self.userNum = len(set(allData['user_id'].values))
        # get item number
        self.itemNum = len(set(allData['item_id'].values))
        # learning rate
        self.learningRate = learning_rate
        # the regularization lambda
        self.alpha_u=alpha_u
        self.alpha_v=alpha_v
        self.alpha_w=alpha_w
        self.beta_u=beta_u
        self.beta_v=beta_v
        # initialize the model and parameters
        self.initModel()

    # initialize all parameters
    def initModel(self):
        self.mu = self.ua_base_explicit['rating'].mean()
        self.bu=(self.rating_matrix-self.mu).sum(axis=1)/self.rating_matrix.count(axis=1)
        self.bu=self.bu.values#dataFrame转numpy
        print(self.bu.shape)
        self.bi = (self.rating_matrix - self.mu).sum() / self.rating_matrix.count()
        self.bi = self.bi.values  # dataFrame转numpy
        self.bi[np.isnan(self.bi)]=0 #填充缺失值


        # r = (np.random.random(1)[0]-0.05)*0.01
        # np.mat((np.random.rand(self.userNum, self.latentFactorNum)-0.05)*0.01)
        self.U = np.mat((np.random.rand(self.userNum, self.latentFactorNum)-0.05)*0.01)
        self.V = np.mat((np.random.rand(self.itemNum, self.latentFactorNum)-0.05)*0.01)
        self.W = np.mat((np.random.rand(self.itemNum, self.latentFactorNum)-0.05)*0.01)
        # self.bu = [0.0 for i in range(self.userNum)]
        # self.bi = [0.0 for i in range(self.itemNum)]
        # temp = math.sqrt(self.latentFactorNum)
        # self.U = [[(0.1 * random.random() / temp) for i in range(self.latentFactorNum)] for j in range(self.userNum)]
        # self.V = [[0.1 * random.random() / temp for i in range(self.latentFactorNum)] for j in range(self.itemNum)]

        print("Initialize end.The user number is:%d,item number is:%d" % (self.userNum, self.itemNum))

    def train(self, iterTimes=5):
        print("Beginning to train the model......")
        preRmse = 10000.0
        temp_count = 0
        start_time=time.time()
        # Set up an early_stopping threshold
        early_stopping_flag=0
        for iter in range(iterTimes):
            count=0
            for index in self.ua_base_explicit.index:
                user = int(self.ua_base_explicit.loc[index]['user_id'])-1
                item = int(self.ua_base_explicit.loc[index]['item_id'])-1
                rating = float(self.ua_base_explicit.loc[index]['rating'])
                pscore = self.predictScore(self.mu, self.bu[user], self.bi[item], self.U[user], self.V[item],self.W[item],user+1)
                eui = rating - pscore
                # update parameters bu and bi(user rating bias and item rating bias)
                self.mu= -eui
                self.bu[user] += self.learningRate * (eui - self.beta_u * self.bu[user])
                self.bi[item] += self.learningRate * (eui - self.beta_v * self.bi[item])

                temp_Uuser = self.U[user]
                temp_Vitem = self.V[item]

                if user+1 in self.implicit.index:
                    temp = self.implicit.loc[user+1][self.implicit.loc[user+1].isnull() == False]
                    U_bar = self.W[temp.index-1].sum()/temp.count()
                else:
                    U_bar = np.zeros(self.latentFactorNum)
                self.U[user] += self.learningRate * (eui * self.V[user] - self.alpha_u * self.U[user])
                self.V[item] += self.learningRate * ((temp_Uuser+U_bar) * eui - self.alpha_v * self.V[item])
                if user+1 in self.implicit.index:
                    self.W[item] += self.learningRate * (eui * temp_Vitem / math.sqrt(self.implicit.loc[user+1].count())- self.alpha_w * self.W[item])
                else:
                    self.W[item] += self.learningRate * (eui * temp_Vitem - self.alpha_w * self.W[item])
                # for k in range(self.latentFactorNum):
                #     temp = self.U[user][k]
                #     # update U,V
                #     self.U[user][k] += self.learningRate * (eui * self.V[user][k] - self.alpha_u * self.U[user][k])
                #     self.V[item][k] += self.learningRate * (temp * eui - self.alpha_v * self.V[item][k])
                #
                count += 1
                if count  % 5000 == 0 :
                    print("第%s轮进度：%s/%s" %(iter+1,count,len(self.ua_base_explicit.index)))
                    # calculate the current rmse
            self.learningRate = self.learningRate * 0.9 # 缩减学习率
            curRmse,curMae = self.test()
            self.RMSE = curRmse
            self.MAE = curMae
            self.iterations = iter+1
            self.timePerEpoch = (time.time()-start_time)/(iter+1)
            print("Iteration %d times,RMSE is : %f" % (iter + 1, curRmse),"Avg time per epoch: {0}s".format(self.timePerEpoch))

            if curRmse > preRmse * 0.995:
                early_stopping_flag+=1
                if early_stopping_flag>3:
                    break
                print('preRmse:{0},curRmse:{1},early_stopping_flag:{2},mae:{3}'.format(preRmse,curRmse,early_stopping_flag,curMae))
            else:
                early_stopping_flag=0
                print('preRmse:{0},curRmse:{1},early_stopping_flag:{2},mae:{3}'.format(preRmse,curRmse,early_stopping_flag,curMae))
                preRmse = curRmse

        print("Iteration finished!")

    # test on the test set and calculate the RMSE
    def test(self):
        cnt = self.test_df.shape[0]
        rmse = 0.0
        mae = 0.0

        buT=self.bu.reshape(self.bu.shape[0],1)
        predict_rate_matrix = self.mu + np.tile(buT,(1,self.itemNum))+ np.tile(self.bi,(self.userNum,1)) +  self.U * self.V.T
        print(predict_rate_matrix)
        # 保存predict_rate_matrix矩阵
        pred_matrix_path_new = pred_matrix_path[:-4]+'_'+str(self.latentFactorNum)+'.npy'
        np.save(pred_matrix_path_new,predict_rate_matrix)

        cur = 0
        for i in self.test_df.index:
            cur +=1
            if cur % 1000 == 0:
                print("测试进度:%s/%s" %(cur,len(self.test_df.index)))
            user = int(self.test_df.loc[i]['user_id']) - 1
            item = int(self.test_df.loc[i]['item_id']) - 1
            score = float(self.test_df.loc[i]['rating'])
            pscore = self.predictScore(self.mu,self.bu[user], self.bi[item], self.U[user], self.V[item],self.W[item],user+1)
            # pscore = predict_rate_matrix[user,item]
            rmse += math.pow(score - pscore, 2)
            mae += abs(score - pscore)
            #print(score,pscore,rmse)
        RMSE=math.sqrt(rmse / cnt)
        MAE=mae / cnt
        return RMSE,MAE

    # calculate the inner product of two vectors
    def innerProduct(self, v1, v2):
        result = 0.0
        for i in range(len(v1)):
            result += v1[i] * v2[i]
        return result

    def predictScore(self, mu, bu, bi, U, V, W ,user_id):
        #pscore = mu + bu + bi + self.innerProduct(U, V)
        if user_id in self.implicit.index:
            temp = self.implicit.loc[user_id][self.implicit.loc[user_id].isnull() == False]
            U_bar = self.W[temp.index-1].sum() / temp.count()
        else:
            U_bar = np.zeros(self.latentFactorNum)
        pscore = mu + bu + bi + np.multiply(U,V).sum() +np.multiply(U_bar,V).sum()
        if np.isnan(pscore):
            print("!!!!")
            print(mu,bu,bi,np.multiply(U,V).sum(),np.multiply(U_bar,V).sum(),U_bar)
        if pscore < 1:
            pscore = 1
        if pscore > 5:
            pscore = 5
        return pscore

    def load_existing_matrix_and_recommend(self, user=1, topK=10):
        time1=time.time()
        pred_matrix_path = pred_matrix_path[:-4]+'_'+str(self.latentFactorNum)+'.npy'
        loaded_weight_matrix=np.load(pred_matrix_path)
        reco_list = []
        for i in range(len(loaded_weight_matrix[user-1])):
            reco_list.append((i,loaded_weight_matrix[user-1][i]))
        #print('reco_list[{0}]={1}'.format(i,reco_list[i]))
    #print(type(reco_list))
    #print(reco_list)
        print("Total SVD++ Time: {0}s".format(time.time()-time1))
        return sorted(reco_list,key=lambda jj:jj[1], reverse=True)[:topK]

    def merge_rating_true_pred(
        rating_true,
        rating_pred,
        col_user='user_id',
        col_item='item_id',
        col_rating='rating',
        col_prediction='prediction',
        ):
        """Join truth and prediction data frames on userID and itemID and return the true
        and predicted rated with the correct index.

        Args:
            rating_true (pd.DataFrame): True data
            rating_pred (pd.DataFrame): Predicted data
            col_user (str): column name for user
            col_item (str): column name for item
            col_rating (str): column name for rating
            col_prediction (str): column name for prediction

        Returns:
            np.array: Array with the true ratings
            np.array: Array with the predicted ratings

        """

        # pd.merge will apply suffixes to columns which have the same name across both dataframes
        suffixes = ["_true", "_pred"]
        rating_true_pred = pd.merge(
            rating_true, rating_pred, on=[col_user, col_item], suffixes=suffixes
            )
        if col_rating in rating_pred.columns:
            col_rating = col_rating + suffixes[0]
        if col_prediction in rating_true.columns:
            col_prediction = col_prediction + suffixes[1]
        return rating_true_pred[col_rating], rating_true_pred[col_prediction]

    def merge_ranking_true_pred(
            rating_true,
            rating_pred,
            relevancy_method,
            col_user='user_id',
            col_item='item_id',
            col_rating='rating',
            col_prediction='prediction',
            k=10,
            threshold=10,
    ):
            """Filter truth and prediction data frames on common users

            Args:
                rating_true (pd.DataFrame): True DataFrame
                rating_pred (pd.DataFrame): Predicted DataFrame
                col_user (str): column name for user
                col_item (str): column name for item
                col_rating (str): column name for rating
                col_prediction (str): column name for prediction
                relevancy_method (str): method for determining relevancy ['top_k', 'by_threshold']
                k (int): number of top k items per user (optional)
                threshold (float): threshold of top items per user (optional)

            Returns:
                pd.DataFrame, pd.DataFrame, int:
                DataFrame of recommendation hits
                DataFrmae of hit counts vs actual relevant items per user
                number of unique user ids
            """

        # Make sure the prediction and true data frames have the same set of users
            common_users = set(rating_true[col_user]).intersection(set(rating_pred[col_user]))
            rating_true_common = rating_true[rating_true[col_user].isin(common_users)]
            rating_pred_common = rating_pred[rating_pred[col_user].isin(common_users)]
            n_users = len(common_users)

        # Return hit items in prediction data frame with ranking information. This is used for calculating NDCG and MAP.
        # Use first to generate unique ranking values for each item. This is to align with the implementation in
        # Spark evaluation metrics, where index of each recommended items (the indices are unique to items) is used
        # to calculate penalized precision of the ordered items.
            if relevancy_method == "top_k":
                top_k = k
            elif relevancy_method == "by_threshold":
                top_k = threshold
            else:
                raise NotImplementedError("Invalid relevancy_method")
            df_hit = get_top_k_items(
                dataframe=rating_pred_common,
                col_user=col_user,
                col_rating=col_prediction,
                k=top_k,
            )
            df_hit["rank"] = df_hit.groupby(col_user)[col_prediction].rank(
                method="first", ascending=False
            )
            df_hit = pd.merge(df_hit, rating_true_common, on=[col_user, col_item])[[col_user, col_item, "rank"]]

            # count the number of hits vs actual relevant items per user
            df_hit_count = pd.merge(
                df_hit.groupby(col_user, as_index=False)[col_user].agg({"hit": "count"}),
                rating_true_common.groupby(col_user, as_index=False)[col_user].agg(
                {"actual": "count"}
                ),
                on=col_user,
            )

            return df_hit, df_hit_count, n_users

    def evaluation(self,topK=10):
        usercol='user_id'
        itemcol='item_id'
        predcol='prediction'
        '''Rating predictions'''
        predictions = [self.predictScore(self.mu,self.bu[getattr(row, usercol)], self.bi[getattr(row, itemcol)], self.U[getattr(row, usercol)], self.V[getattr(row, itemcol)],self.W[getattr(row, itemcol)],getattr(row, usercol)+1) for row in self.test_df.itertuples()]
        predictions = pd.DataFrame(predictions)
        predictions = predictions.rename(index=str, columns={'uid': usercol, 'iid': itemcol, 'est': predcol})
        print('rating predictions sample:',predictions.head(1))
        print('true rating sample:',self.test_df.head(1))
        def compute_ranking_predictions(remove_seen=False):
            """Computes predictions of an algorithm from Surprise on all users and items in data. can be used for computing
            ranking metrics like NDCG.

            Args:
                algo (surprise.prediction_algorithms.algo_base.AlgoBase): an algorithm from Surprise
                data (pd.DataFrame): the data from which to get the users and items
                usercol (str): name of the user column
                itemcol (str): name of the item column
                remove_seen (bool): flag to remove (user, item) pairs seen in the training data

            Returns:
                pd.DataFrame: dataframe with usercol, itemcol, predcol
            """
            data = self.test_df
            preds_lst = []
            for user in data[usercol].unique():
                for item in data[itemcol].unique():
                    preds_lst.append([user, item, algo.predictScore(self.mu,self.bu[user], self.bi[item], self.U[user], self.V[item],self.W[item],user+1).est])

            all_predictions = pd.DataFrame(data=preds_lst, columns=[usercol, itemcol, predcol])

            if remove_seen:
                tempdf = pd.concat([data[[usercol, itemcol]],
                            pd.DataFrame(data=np.ones(data.shape[0]), columns=['dummycol'], index=data.index)],
                            axis=1)
                merged = pd.merge(tempdf, all_predictions, on=[usercol, itemcol], how="outer")
                return merged[merged['dummycol'].isnull()].drop('dummycol', axis=1)
            else:
                return all_predictions

        df_hit, df_hit_count, n_users = merge_ranking_true_pred(
            rating_true=self.test_df,
            rating_pred=predictions,
            relevancy_method='top_k',
        )
        print("df_hit.shape:",df_hit.shape,",df_hit_count.shape:",df_hit_count.shape)
        def MAE():
            return self.MAE
        def RMSE():
            return self.RMSE
        def precision():
            if df_hit.shape[0] == 0:
                return 0.0
            return (df_hit_count["hit"] / topK).sum() / n_users
        def recall_at_topK():
            if df_hit.shape[0] == 0:
                return 0.0
            return (df_hit_count["hit"] / df_hit_count["actual"]).sum() / n_users
        self.precision = precision()
        self.recall = recall_at_topK()
        return MAE,RMSE,precision(),recall_at_topK()

    def report_generation(self):
        file_path = 'outputs/report_'+str(self.latentFactorNum)+'latentFactors.txt'
        f = open(file_path,'a')
        content='Adopted {0} Latent factors, iteration {1} times, RMSE is: {2}, MAE:{3}, precision: {4}, recall: {5}, avg time per epoch: {6}s'.format(self.latentFactorNum, self.iterations, self.RMSE, self.MAE, self.precision, self.recall, self.timePerEpoch)
        f.write(content)
        f.close()

def main(_):
    s = SVDplusplus(allfile=udata_path, trainfile=u1base_path, testfile=u1test_path, latentFactorNum=FLAGS.latentFactorNum)

    print(s.test_df.shape[0])
    print('s.mu:',s.mu)
    print('s.bu:',s.bu,'shape:',s.bu.shape)
    print('s.bi:',s.bi,'shape:',s.bi.shape)
    print('s.U:',s.U,'shape:',s.U.shape)
    print('s.V:',s.V,'shape:',s.V.shape)
    print('s.W:',s.W,'shape:',s.W.shape)
    s.train(iterTimes=FLAGS.Epochs)
    s.evaluation()
    s.report_generation()


if __name__ == '__main__':
    tf.app.run()
