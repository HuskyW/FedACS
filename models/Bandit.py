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
        self.gamma = 0.2
        self.round_idx = 0
        self.clients = np.arange(self.num_clients,dtype='int')
        self.weights = np.ones((self.num_clients),dtype='int')



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
        self.weights = np.ones((self.num_clients),dtype='int')
        return

    def __possibilities(self,reweight=False):
        if reweight == True:
            self.weights = self.weights * (self.num_clients / sum(self.weights))
        weights_sum = sum(self.weights)
        possi = np.zeros((self.num_clients),dtype='int')
        for i in range(self.num_clients):
            possi[i] = (1-self.gamma)*(self.weights[i]/weights_sum) + (self.gamma/self.num_clients)
        return possi


class MoveAvgBandit(Bandit):
    def __init__(self,args):
        self.num_clients = args.num_users
        self.avgs = {}
        for client in range(self.num_clients):
            self.avgs[client] = 0
        self.clients = np.arange(self.num_clients,dtype='int')
        self.uninitialized = [i for i in range(self.num_clients)]
        self.lastinit = 0 # 0: initializing, 1: the last round for initialization, 2: working

        # Parameters
        self.learningRate = 0.05
        self.epsilon = 0.05

    def requireArms(self,num_picked):
        # All required arms is uninitialized
        if len(self.uninitialized) >= num_picked:
            if len(self.uninitialized) == num_picked and self.lastinit == 0:
                self.lastinit = 1
            result = np.random.choice(self.uninitialized,num_picked,replace=False)
            for i in result:
                self.uninitialized.remove(i)
            return result

        if self.lastinit == 0:
            self.lastinit = 1

        # Part of arms is uninitialized
        if len(self.uninitialized) > 0:
            reserved = np.array(self.uninitialized,dtype='int')
            num_left = num_picked - len(self.uninitialized)
            self.uninitialized.clear()
            temp = self.clients.copy()
            for i in reserved:
                temp = np.delete(temp, np.argwhere(temp == i))
            newpicked = np.random.choice(temp,num_left,replace=False)
            result = np.concatenate([reserved,newpicked])
            return result

        # All arms initialized
        sortarms = sorted(self.avgs.items(),key=lambda x:x[1],reverse=True)
        results = []
        idx = 0
        for i in range(num_picked):
            draw = np.random.ranf()
            if draw < self.epsilon:
                while True:
                    client = np.random.choice(self.clients,1)
                    if client not in results:
                        results.append(client)
                        break
            else:
                while sortarms[idx][0] in results:
                    idx += 1
                results.append(sortarms[idx][0])
        return np.array(results,dtype='int')

    def updateWithRewards(self,rewards):
        for client,reward in rewards.items():
            if self.lastinit <= 1:
                self.avgs[client] = reward
            else:
                self.avgs[client] = (1-self.learningRate) * self.avgs[client] + self.learningRate * reward

        if self.lastinit == 1:
            self.lastinit = 2
        return

class SelfSparringBandit(Bandit):
    def __init__(self,args):
        self.num_clients = args.num_users
        self.s = [0] * self.num_clients
        self.f = [0] * self.num_clients
        self.extension = args.extension
        self.historical_rounds = args.historical_rounds
        self.history = []
        if self.historical_rounds > 0:
            self.lr = float(1/self.historical_rounds)
        else:
            self.lr = 1
    
    def requireArms(self,num_picked):
        num_candidate = int(num_picked * self.extension)
        candidates = [i for i in range(self.num_clients)]
        picked = []
        for j in range(num_candidate):
            record = {}
            for candidate in candidates:
                record[candidate] = np.random.beta(self.s[candidate]+1,self.f[candidate]+1)
            sortedRecord = sorted(record.items(),key=lambda x:x[1],reverse=True)
            winner = sortedRecord[0][0]
            picked.append(winner)
            candidates.remove(winner)
        return np.random.choice(picked,num_picked,replace=False)

    def updateWithRewards(self,rewards):
        x = list(rewards.items())
        y = list(rewards.items())

        # x is current round rewards, y includes historical records (no duplicate clients)

        usedclients = set(rewards.keys())

        for hist in self.history:
            for onerecord in hist:
                if onerecord[0] not in usedclients:
                    usedclients.add(onerecord[0])
                    y.append(onerecord)
        
        self.history.append(x.copy())
        if len(self.history) > self.historical_rounds:
            del self.history[0]

        for i,ir in x:
            for j, jr in y:
                if i == j:
                    continue
                if ir > jr:
                    self.s[i] += self.lr
                else:
                    self.f[i] += self.lr
        return

def tryBandit(bandit,iter,verbose=True):
    record = []
    for r in range(iter):
        arms = bandit.requireArms(10)
        realuti = sum(arms)/len(arms)
        record.append(realuti)
        if verbose:
            print("Round %3d: reward %.1f" % (r,realuti))
        rewards = {}
        for i in arms:
            flowing = 20*np.random.ranf()- 10
            rewards[i] = (i+1) * (1-0.001*r) + flowing
            #rewards[i] = (i+1) * flowing
        bandit.updateWithRewards(rewards)
    if verbose:
        print("Average Utility: %1f" % (sum(record)/len(record)) )

    return float(sum(record)/len(record))


class argument():
    def __init__(self):
        self.extension = 1
        self.num_users = 200
        self.historical_rounds = 5

if __name__ == "__main__":
    args = argument()
    record1 = []
    for i in range(10):
        bandit1 = SelfSparringBandit(args)
        print("Try: %3d" % (i+1))
        record1.append(tryBandit(bandit1,500))

    print("ss : %.1f" % (sum(record1)/len(record1)) )
