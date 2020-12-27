import abc
import numpy as np
import math
from scipy.special import comb

class Bandit(metaclass=abc.ABCMeta):
    @abc.abstractclassmethod
    def requireArms(self):
        pass

    @abc.abstractclassmethod
    def updateWithRewards(self,rewards):
        pass

class Rexp3Bandit(Bandit):
    def __init__(self,num_clients):
        self.num_clients = num_clients
        self.batch_size = 500
        self.gamma = 0.1
        self.round_idx = 0
        self.clients = np.arange(self.num_clients)
        self.weights = np.ones((self.num_clients))



    def requireArms(self,num_picked):
        if self.round_idx >= self.batch_size:
            self.__init_batch()

        possi = self.__possibilities()
        draw = np.random.choice(self.clients,num_picked,p=possi,replace=False)
        return draw

    def updateWithRewards(self,rewards):
        possi = self.__possibilities(reweight=True)
        for client, reward in rewards.items():
            xhat = reward/possi[client]
            self.weights[client] = self.weights[client] * math.exp(self.gamma * xhat / self.num_clients)
        

    def __init_batch(self):
        self.round_idx = 0
        self.weights = np.ones((self.num_clients))
        return

    def __possibilities(self,reweight=False):
        if reweight == True:
            self.weights = self.weights * self.num_clients / sum(self.weights)
        weights_sum = sum(self.weights)
        possi = np.zeros((self.num_clients))
        for i in range(self.num_clients):
            possi[i] = (1-self.gamma)*(self.weights[i]/weights_sum) + (self.gamma/self.num_clients)
        return possi

class UcbqrBandit(Bandit):
    def __init__(self,num_clients):
        self.num_clients = num_clients
        self.clients = np.arange(self.num_clients)
        self.lose = {}
        self.utility = {}
        for client in self.clients:
            self.lose[client] = 0
            self.utility[client] = 0
        self.uninitialized = [i for i in range(self.num_clients)]
        self.lastinit = 0
        self.confidence = 0.8
        
    def requireArms(self,num_picked):
        if len(self.uninitialized) >= num_picked:
            if len(self.uninitialized) == num_picked and self.lastinit == 0:
                self.lastinit = 1
            result = np.random.choice(self.uninitialized,num_picked,replace=False)
            for i in result:
                self.uninitialized.remove(i)
            return result

        if self.lastinit == 0:
            self.lastinit = 1

        if len(self.uninitialized) > 0:
            reserved = np.array(self.uninitialized)
            num_left = num_picked - len(self.uninitialized)
            self.uninitialized.clear()
            temp = self.clients.copy()
            for i in reserved:
                temp = np.delete(temp, np.argwhere(temp == i))
            newpicked = np.random.choice(temp,num_left,replace=False)
            result = np.concatenate([reserved,newpicked])
            return result

        optimism = self.__optimism()
        sortopti = sorted(optimism.items(),key=lambda x:x[1],reverse=True)

        result = np.zeros((num_picked))

        for i in range(num_picked):
            result[i] = sortopti[i][0]
        
        return result


    def updateWithRewards(self,rewards):
        
        if self.lastinit <= 1:
            sortrewards = sorted(rewards.items(),key=lambda x:x[1],reverse=True)
            assignutil = self.num_clients
            decrease = math.floor(self.num_clients / len(sortrewards))

            for i in range(len(sortrewards)):
                client = sortrewards[i][0]
                self.utility[client] = assignutil
                assignutil -= decrease
            
            if self.lastinit == 1:
                self.lastinit = 2
            
            return

        for i in rewards.keys():
            for j in rewards.keys():
                if i == j:
                    continue
                if rewards[i] > rewards[j]:
                    self.lose[j] += 1
                    if self.utility[i] <= self.utility[j]:
                        if self.utility[i] >= self.num_clients or self.utility[j] <= 1:
                            continue
                        self.utility[i] += 1
                        self.utility[j] -= 1
        return
                


    def __optimism(self):
        results = self.utility.copy()
        for client in self.clients:
            util = self.utility[client]
            lose = self.lose[client]
            opti_freedom = self.num_clients - util
            if opti_freedom == 0:
                continue

            correctLeaving = False
            for k in range(1,opti_freedom+1):
                possi = comb(opti_freedom,k) * math.pow((k/opti_freedom),lose)
                if possi > self.confidence:
                    correctLeaving = True
                    results[client] += opti_freedom - k + 1
                    break
            
            if correctLeaving == False:
                print("Error in bandit: derive optimism")

        return results

if __name__ == "__main__":
    bandit = UcbqrBandit(100)
    for r in range(500):
        arms = bandit.requireArms(10)
        print("Round %3d: reward %.1f" % (r,sum(arms)/len(arms)))
        rewards = {}
        for i in arms:
            flowing = (np.random.ranf()/2.5) + 0.8
            rewards[i] = i * (1-0.001*r) * flowing
            #rewards[i] = i * flowing
        bandit.updateWithRewards(rewards)
    