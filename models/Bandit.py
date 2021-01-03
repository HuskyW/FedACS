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

class UcbqrBandit(Bandit):
    def __init__(self,num_clients):
        self.num_clients = num_clients
        self.round_idx = 0
        self.clients = np.arange(self.num_clients,dtype='int')
        self.lose = {}
        self.utility = {}
        self.lastcall = {}
        for client in self.clients:
            self.lose[client] = 0
            self.utility[client] = 0
            self.lastcall[client] = 0
        self.uninitialized = [i for i in range(self.num_clients)]
        self.lastinit = 0 # 0: initializing, 1: the last round for initialization, 2: working

        # parameters
        self.confidence = 0.8
        self.extendCandidates = 2
        self.forgetLose = 1 # 1: dont forget, 0.98: forget 2% each round
        self.fairnessReservation = 0
        
    def requireArms(self,num_picked):
        self.round_idx += 1
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

        # All initialized (working rounds)

        reservedClientNum = int(np.floor(num_picked * self.fairnessReservation))
        scrambledClientNum = num_picked - reservedClientNum

        optimism = self.__optimism()
        sortopti = sorted(optimism.items(),key=lambda x:x[1],reverse=True)

        candidates = np.zeros((int(scrambledClientNum*self.extendCandidates)),dtype='int')

        for i in range(int(scrambledClientNum*self.extendCandidates)):
            candidates[i] = sortopti[i][0]

        winner = np.random.choice(candidates,scrambledClientNum,replace=False)
        if reservedClientNum == 0:        
            return winner
        
        sortByAging = sorted(self.lastcall.items(),key=lambda x:x[1],reverse=False)

        fairnessClients = np.zeros((reservedClientNum),dtype='int')
        idx = 0
        i = 0
        for i in range(len(sortByAging)):
            if idx >= reservedClientNum:
                break
            agingClient = sortByAging[i][0]
            if agingClient in winner:
                continue
            fairnessClients[idx] = agingClient
            idx += 1
        result = np.concatenate([winner,fairnessClients])
        return result


    def updateWithRewards(self,rewards):
        # Initializing
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

        # Working
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

        for client in self.lose.keys():
            self.lose[client] = int(np.ceil(self.lose[client]*self.forgetLose))
        
        for client in rewards.keys():
            self.lastcall[client] = self.round_idx

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

class MoveAvgBandit(Bandit):
    def __init__(self,num_clients):
        self.num_clients = num_clients
        self.avgs = {}
        for client in range(num_clients):
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

class MultiRUcb(Bandit):
    def __init__(self,num_clients,num_picked):
        self.num_picked = num_picked
        self.num_clients = num_clients
        self.alpha = 0.8
        self.w = np.zeros((num_clients,num_clients))
        self.b = []
        self.iter = 1
        for i in range(num_picked):
            self.b.append(set())
    
    def requireArms(self,num_picked):
        u = self.__deriveU()
        reserved = set()
        return self.__armsRec(num_picked,u,reserved,0)

    def updateWithRewards(self,rewards):
        self.iter += 1
        for i in rewards.keys():
            for j in rewards.keys():
                if i == j:
                    continue
                if rewards[i] > rewards[j]:
                    self.w[i][j] += 1


    def __armsRec(self,num_picked,u,reserved,idx):
        num_left = num_picked - len(reserved)
        if idx >= num_picked:
            print('Error')
        c = self.__deriveC(reserved,u)
        if len(c) == 0:
            print("c=0!")
        self.b[idx] = self.b[idx] & c
        if len(c) <= 1:
            self.b[idx] = c
            reserved = reserved | c
            return self.__armsRec(num_picked,u,reserved.copy(),idx+1)
        if len(c) < num_left:
            reserved = reserved | c
            return self.__armsRec(num_picked,u,reserved.copy(),idx+1)
        if len(c) == num_left:
            reserved = reserved | c
            return reserved
        if np.random.ranf() > 0.5:
            reserved = reserved | self.b[idx]
            new_num_left = num_left = num_picked - len(reserved)
            c = c - self.b[idx]
            draw = np.random.choice(list(c),new_num_left,replace=False)
            return reserved | set(draw)
        c = c - self.b[idx]
        draw = np.random.choice(list(c),num_left,replace=False)
        return reserved | set(draw)

    
    def __deriveC(self,reserved,u):
        c = set()
        for i in range(self.num_clients):
            if i in reserved:
                continue
            select = True
            for j in range(self.num_clients):
                if j in reserved:
                    continue
                if u[i][j] < 0.5:
                    select = False
                    break
            if select:
                c.add(i)

        return c

    def __deriveU(self):
        shape = self.w.shape
        udown = np.zeros(shape)
        udown = self.w + self.w.T
        u1 = self.__specDiv(self.w,udown)
        u2up = np.ones(shape)
        u2up = u2up * (self.alpha * math.log(self.iter))
        u2 = np.sqrt(self.__specDiv(u2up,udown))
        ans = u1 + u2
        for i in range(shape[0]):
            ans[i][i] = 0.5
        return ans

    def __specDiv(self,a,b):
        shape = a.shape
        ans = np.zeros(shape)
        width = shape[0]
        height = shape[1]
        for i in range(width):
            for j in range(height):
                if b[i][j] == 0:
                    ans[i][j] = 1
                else:
                    ans[i][j] = a[i][j] / b[i][j]
        return ans


class SelfSparringBandit(Bandit):
    def __init__(self,num_clients):
        self.num_clients = num_clients
        self.s = [0] * num_clients
        self.f = [0] * num_clients
        self.lr = 1
    
    def requireArms(self,num_picked):
        candidates = [i for i in range(self.num_clients)]
        picked = []
        for j in range(num_picked):
            record = {}
            for candidate in candidates:
                record[candidate] = np.random.beta(self.s[candidate]+1,self.f[candidate]+1)
            sortedRecord = sorted(record.items(),key=lambda x:x[1],reverse=True)
            winner = sortedRecord[0][0]
            picked.append(winner)
            candidates.remove(winner)
        return np.array(picked,dtype='int')

    def updateWithRewards(self,rewards):
        for i in rewards.keys():
            for j in rewards.keys():
                if i == j:
                    continue
                if rewards[i] > rewards[j]:
                    self.s[i] += self.lr
                    self.f[j] += self.lr
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

if __name__ == "__main__":

    record1 = []
    for i in range(10):
        bandit1 = SelfSparringBandit(200)
        bandit2 = UcbqrBandit(200)
        bandit3 = MoveAvgBandit(200)
        print("Try: %3d" % (i+1))
        record1.append(tryBandit(bandit1,500))

    print("ss : %.1f" % (sum(record1)/len(record1)) )
