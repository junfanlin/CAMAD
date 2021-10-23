import numpy as np
import random
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import logging

class Simple_multiclass(nn.Module):
    def __init__(self, input_dim, embed_dim, output_dim, cls=3):
        super(Simple_multiclass, self).__init__()
        self.cls = cls
        self.propensity = nn.Sequential(
                                nn.Linear(input_dim, embed_dim), \
                                nn.ReLU(inplace=True), \
                                nn.Linear(embed_dim, embed_dim), \
                                nn.ReLU(inplace=True),
                                nn.Linear(embed_dim, embed_dim), \
                            )
        self.classifier = nn.Sequential(
                                nn.ReLU(inplace=True), \
                                nn.Linear(embed_dim, output_dim*cls)
                            )
    
    def encode(self, x):
        return self.propensity(x)
    
    def forward(self, x):
        logits = self.classifier(self.encode(x)).view(x.shape[0], -1, self.cls)
        return logits
    
class Patient():
    def __init__(self, data_selfr, data_symp, data_dise, block_data, bg):
        self.selfrbase = data_selfr
        self.sympbase = data_symp
        self.disebase = data_dise
        self.block_data = block_data
        
        if bg == 'all':
            path = 'multiclassifier_prop_withtest_ori'
        else:
            path = 'multiclassifier_prop_ori'
        
        print(path)
        
        self.classifier = Simple_multiclass(66+4, 32, 66).cuda()    
        self.classifier.load_state_dict(torch.load('checkpoints/new_exp_models/classifier/'+path))
#         self.reset_iterative()
    
    def _printf(self, sym_id, state):
        response_dict = {-1: "I don't have %s", 0: "I'm not sure about %s", 1: "I have %s"}
        print(response_dict[state]%(sym_list[sym_id]),sym_id)
    
    def report(self, rep):
        mask_poses = np.argwhere(rep != 0).reshape(-1)
        for mask_pos in mask_poses:
            self._printf(mask_pos, rep[mask_pos])
    
    def answer(self, query, verbose=False):
        if verbose: self.printf(query, self.state[0, query])
        return self.state[0, query]
    
    def sample(self, k = None, np_random=np.random):
        input_sample = np.concatenate([self.state, np.eye(4).astype(np.float32)[self.dise]], -1)
        pool_mask = self.pool*self.mask
        pool_sample = np.concatenate([pool_mask, np.eye(4).astype(np.float32)[self.disebase[self.indexes]]], -1)
        
        all_the_same = np.all(input_sample == pool_sample)
        
        if not all_the_same:
            input_sample_batch = np.concatenate([input_sample, pool_sample], 0)
            with torch.no_grad():
                prop_sample_batch = self.classifier.encode(torch.from_numpy(input_sample_batch).cuda())
#                 distance = torch.sqrt(((prop_sample_batch[0]-prop_sample_batch[1:])**2/prop_sample_batch[1:].var(0)).sum(-1))
                distance = ((prop_sample_batch[0]-prop_sample_batch[1:])**2/prop_sample_batch[1:].var(0)).sum(-1)
                similarity = torch.nn.functional.softmax(-distance, -1).data.cpu().numpy()
        else:
            similarity = np.ones(pool_sample.shape[0])/pool_sample.shape[0]
        
        prop_ori = (np.eye(3)[self.pool.astype(np.int)+1] * similarity[:, None, None]).sum(0)
        prop = prop_ori[:, [0, 2]]
        prop_sum = prop.sum(1)
        unknown_pos = np.argwhere(prop_sum == 0)
        prop[unknown_pos] = 0.5
        prop = prop/prop.sum(1, keepdims=True)

        new_state = np_random.binomial(1, p = prop[:, 1], size=(1, 66))*2-1
        new_state[:, unknown_pos] = 0
        
        return new_state if k is None else new_state[0, k]
    
    def reset_iterative(self, id, np_random=np.random):
        self.id = id
        self.selfr = self.selfrbase[[self.id]]
        self.mask = self.selfr != 0
        if self.mask.sum() != 0:
            known_idx = np.argwhere(self.mask[0]).reshape(-1)
            known_idx = np_random.choice(known_idx, size = len(known_idx), replace=True)
            self.mask = np.zeros_like(self.selfr).astype(np.float32)
            self.mask[:, known_idx] = 1
            self.selfr = self.selfr*self.mask

        self.dise = self.disebase[[self.id]]
        self.state = np.array(self.selfr)
        self.indexes = self.disebase == self.dise[0]
        self.pool = self.sympbase[self.indexes]
        
        unknown = np.argwhere(self.mask[0] == 0)  # this requires direction of symptom effect
        for unknown_pos in unknown:
            self.state[0, unknown_pos] = self.sample(unknown_pos, np_random=np_random)
            self.mask[0, unknown_pos] = 1
        
    
    def reset(self, id, np_random=np.random, allow_rep=False, exact=False):
        self.id = id
        
        resamp = True
        while resamp:
            resamp = False
            
            self.selfr = self.selfrbase[[self.id]]
            self.mask = self.selfr != 0
            if self.mask.sum() != 0 and not exact:
                known_idx = np.argwhere(self.mask[0]).reshape(-1)
                known_idx = np_random.choice(known_idx, size = len(known_idx), replace=True)
                self.mask = np.zeros_like(self.selfr).astype(np.float32)
                self.mask[:, known_idx] = 1
                self.selfr = self.selfr*self.mask

            self.dise = self.disebase[[self.id]]
        
            self.state = np.array(self.sympbase[[self.id]])
            if not exact:
                mask = np_random.randint(2, size=self.state.shape).astype(np.float32)
                self.state = self.state * mask
                self.state[self.selfr!=0] = self.selfr[self.selfr!=0]

            self.indexes = self.disebase == self.dise[0]
            self.pool = self.sympbase[self.indexes]

            self.mask = self.state != 0

            new_state = self.sample(np_random=np_random)
            self.state = new_state*(1-self.mask)+self.state*self.mask
            
            self.in_block_data = np.any(np.all(self.state == self.block_data, -1))
            
            if allow_rep and self.in_block_data:
                resamp = True

        
class MuzhiEnv():
    metadata = {'render.modes': ['human'], 'unknown':0, 'positive':1, 'negative':-1}
    
    def __init__(self, data_bg, mode='train', tol=11):
        with open('split_data/goal_dict_all_block.p', 'rb') as fhand:
            import pickle
            self.goal_dicts = pickle.load(fhand)
            fhand.close()
            
        with open('split_data/slot_set.txt', 'r') as fhand:
            self.sym_list = fhand.readlines()
            self.sym_list = [sym.strip() for sym in self.sym_list]
            fhand.close()
            
        with open('split_data/diseases.txt', 'r') as fhand:
            self.dise_list = fhand.readlines()
            self.dise_list = [dise.strip() for dise in self.dise_list]
            fhand.close()
        
        self.tol = tol
        self.mode = mode
        self.user_id = -1
        
        self.goals = self.goal_dicts[data_bg]
        
        data_symp = np.zeros((len(self.goals), len(self.sym_list)), dtype=np.float32)
        data_selfr = np.zeros((len(self.goals), len(self.sym_list)), dtype=np.float32)
        data_dise = np.zeros(len(self.goals), dtype=np.int)

        for i in range(len(self.goals)):
            c_goal = self.goals[i]
            for slot in c_goal['implicit_inform_slots']:
                data_symp[i][self.sym_list.index(slot)] = 1 if c_goal['implicit_inform_slots'][slot] else -1
            for slot in c_goal['explicit_inform_slots']:
                data_symp[i][self.sym_list.index(slot)] = 1 if c_goal['explicit_inform_slots'][slot] else -1
                data_selfr[i][self.sym_list.index(slot)] = 1 if c_goal['explicit_inform_slots'][slot] else -1
            data_dise[i] = self.dise_list.index(c_goal['disease_tag'])
        
        
        block_data_name = 'test_data_bgall'
        data_symp_block = np.zeros((len(self.goal_dicts[block_data_name]), len(self.sym_list)), dtype=np.float32)
        
    
        for i in range(len(self.goal_dicts[block_data_name])):
            c_goal = self.goal_dicts[block_data_name][i]
            for slot in c_goal['implicit_inform_slots']:
                data_symp_block[i][self.sym_list.index(slot)] = 1 if c_goal['implicit_inform_slots'][slot] else -1
            for slot in c_goal['explicit_inform_slots']:
                data_symp_block[i][self.sym_list.index(slot)] = 1 if c_goal['explicit_inform_slots'][slot] else -1
        
        self.patient = Patient(data_symp=data_symp, data_selfr=data_selfr, data_dise=data_dise, block_data = data_symp_block, bg=data_bg)
        self.seed()
        self.reset()
    
    def seed(self, seed=None):
        from gym.utils import seeding
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def rewarding(self, info):
        if 'success' in info:
            return 10 if info['success'] else -100
        if 'repeat' in info:
            if info['repeat']:
                return -88
            else:
                return 0
    
    def step(self, action, verbose=False):
        assert self.done is False, "It's game over now."
        info = {}
        reward = None
        
        if action < len(self.dise_list):
            self.done = True
            info['success'] = (action == self.dise)
            info['hit'] = (action == self.dise)
        else:
            action = action - len(self.dise_list)
            info['repeat'] = (self.have_visit[action] != 0)
            if not info['repeat']:
                self.obs[action] = self.state[action]
                self.have_visit[action] = 1.
                info['hit'] = (self.state[action] != self.metadata['unknown'])
            
            if verbose: print("%d %s"%(self.state[action], self.sym_list[action]))
        self.runs += 1
        
        
        if self.runs >= self.tol and not self.done:
            info['leave'] = True
            info['success'] = False
            self.done = True
        
        reward = self.rewarding(info)
        assert reward is not None, "You shoud design a reward for %s"%(str(info))
        
        info['runs'] = self.runs
        return np.array(np.hstack((self.obs, self.have_visit))), reward, self.done, info
    
    def reset(self, verbose=False, allow_rep=False):
        self.runs = 0
        self.done = False
        if self.mode == 'train':
            self.user_id = self.np_random.randint(len(self.goals))
        elif self.mode == 'test':
            self.user_id = (self.user_id + 1)%len(self.goals)
        
        if self.mode == 'train':
            self.patient.reset(id=self.user_id, np_random=self.np_random, allow_rep=allow_rep)    

            self.state = self.patient.state[0]
            self.obs = self.patient.selfr[0]
            self.dise = self.patient.dise[0]
        
        elif self.mode == 'test':
            selfr = self.patient.selfrbase[self.user_id]
            self.obs = np.array(selfr)
            state = self.patient.sympbase[self.user_id]
            self.state = np.array(state)
            self.dise = self.patient.disebase[self.user_id]
            
        
        self.have_visit = (self.obs != 0).astype(np.float32)

        if verbose: print(np.array(self.sym_list)[self.have_visit==1], self.dise_list[self.dise])
        return np.array(np.hstack((self.obs, self.have_visit)))
    
    def render(self, mode='human', close=False):
        return np.array(np.hstack((self.obs, self.have_visit)))       



class Simple_classifier(nn.Module):
    def __init__(self, input_dim, embed_dim, output_dim):
        super(Simple_classifier, self).__init__()
        self.propensity = nn.Sequential(
                                nn.Linear(input_dim, embed_dim*8), \
                                nn.LeakyReLU(0.1, inplace=True), \
                                nn.Linear(embed_dim*8, embed_dim), \
#                                 nn.Softmax(-1), \
                          )
        self.classfier = nn.Sequential(
                                nn.ReLU(inplace=True), \
                                nn.Linear(embed_dim, output_dim) \
                          )
    def encode(self, x):
        return self.propensity(x)
    
    def forward(self, x):
        logits = self.classfier(self.propensity(x))
        return logits


class DQN(nn.Module):
    def __init__(self, input_shape, hidden_size, num_actions, dueling=False):
        super(DQN, self).__init__()

        self.input_shape = input_shape
        self.num_actions = num_actions
        self.hidden_size = hidden_size
        self.dueling = dueling

        if self.dueling:
            self.model_obs = nn.Sequential(nn.Linear(input_shape, self.hidden_size), \
                                       nn.ReLU(inplace=True), \
                                       nn.Linear(self.hidden_size, self.hidden_size), \
                                       nn.ReLU(inplace=True))

            self.model_adv = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size), \
                                           nn.ReLU(inplace=True), \
                                           nn.Linear(self.hidden_size, num_actions))

            self.model_val = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size), \
                                           nn.ReLU(inplace=True), \
                                           nn.Linear(self.hidden_size, 1))
        else:
            self.model_obs = nn.Sequential(nn.Linear(input_shape, self.hidden_size), \
                                   nn.ReLU(inplace=True), \
                                   nn.Linear(self.hidden_size, self.hidden_size), \
                                   nn.ReLU(inplace=True), \
                                   nn.Linear(self.hidden_size, self.num_actions))

    def forward(self, x):
        if self.dueling:
            feat = self.model_obs(x)
            adv = self.model_adv(feat)
            val = self.model_val(feat)

            out = val+adv-adv.mean(1, keepdim=True)
        else:
            out = self.model_obs(x)

        return out-x[:, self.num_actions:self.num_actions*2]*1000000

    def soft_update(self, state_dict, ratio = 0.9):
        model_dict = self.state_dict()
        for key in model_dict:
            model_dict[key].copy_(model_dict[key]*ratio+state_dict[key]*(1-ratio))
        self.load_state_dict(model_dict)

class ExperienceReplayMemory:
    def __init__(self, capacity, memory_type):
        self.capacity = capacity
        self.memory_type = memory_type
        self.buffer = []

    def push(self, s, a, r, d, s_):
        transition = np.hstack((s, [a, r, d], s_))
        self.buffer.append(transition)
        if len(self.buffer) > self.capacity:
            if self.memory_type == "list":
                del self.buffer[0]
            elif self.memory_type == "random":
                del self.buffer[np.random.randint(self.capacity)]

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
    
    def full(self):
        return len(self.buffer) == self.capacity

class ExperienceMemory:
    def __init__(self, capacity, memory_type):
        self.capacity = capacity
        self.memory_type = memory_type
        self.input_buffer = []
        self.label_buffer = []
        

    def push(self, input, label):
        self.input_buffer.append(input)
        self.label_buffer.append(label)
        
        if len(self.input_buffer) > self.capacity:
            if self.memory_type == "list":
                del self.input_buffer[0]
                del self.label_buffer[0]
            elif self.memory_type == "random":
                index = np.random.randint(self.capacity)
                del self.input_buffer[index]
                del self.label_buffer[index]

    def sample(self, batch_size):
        indexes = np.arange(len(self.input_buffer))
        indexes = np.random.choice(indexes, batch_size, replace=True)
        return np.array([self.input_buffer[index] for index in indexes]), np.array([self.label_buffer[index] for index in indexes])

    def __len__(self):
        return len(self.input_buffer)
    
    def full(self):
        return len(self.input_buffer) == self.capacity
    
    def erase(self):
        del self.input_buffer
        del self.label_buffer
        self.input_buffer = []
        self.label_buffer = []

class Agent(nn.Module):
    def __init__(self, state_len, action_len, dise_len, emb_len, params):
        super(Agent, self).__init__()
        self.state_len = state_len
        self.action_len = action_len
        self.dise_len = dise_len
        self.emb_len = emb_len
        self.conf_threshold = params['conf_threshold']
        self.device = params['device']
        self.random_start = params['random_start']
        self.num_bootstrap = params['num_bootstrap']
        self.dise_index = torch.arange(dise_len)[None, :, None].to(self.device)
        
        self.reclassifier = Simple_multiclass(66, 32, 66).cuda()
        self.optimizer_reclassifier = torch.optim.Adam(self.reclassifier.parameters(), lr = 1e-3)
        
        self.classifier3 = []
        self.optimizer_classifier3 = []
        for i in range(self.num_bootstrap):
            self.classifier3.append(Simple_classifier(66, 8, 4).cuda())
            self.optimizer_classifier3.append(torch.optim.Adam(self.classifier3[i].parameters(), lr = 1e-3))
            
        self.classifier3_module = nn.ModuleList(self.classifier3)#+self.optimizer_classifier3)
        
        self.model = DQN(state_len, 128, self.action_len, params['dueling'])
        self.target_model = DQN(state_len, 128, self.action_len, params['dueling'])
        self.target_model.load_state_dict(self.model.state_dict())
        
        self.params = params
        self.batch_size = params['batch_size']
        self.memory_size = params['memory_size']
        self.memory = ExperienceReplayMemory(capacity=self.memory_size, memory_type=params['memory_type'])
        self.memory_checker = ExperienceMemory(capacity=self.batch_size*40, memory_type=params['memory_type'])
        self.gamma = params['gamma']
        self.epsilon = params['epsilon']
        self.target_update_freq = params['target_update_freq']
        self.elim_update_freq = params['elim_update_freq']
        self.soft_ratio = params['soft_ratio']
        
        self.learn_step_counter = 0
        
        self.criterion = nn.MSELoss()
        self.criterion_class = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = params['lr'])
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, params['update_steps'])
    
    
    def cal_prop_uncertainty(self, input_data, mask, sample_num = 1):
        samples = input_data
        props = []
        for i in range(self.num_bootstrap):
            props.append(torch.nn.functional.softmax(self.classifier3[i](samples), -1).view(-1, 1, 4))
        props = torch.cat(props, 1)
#         print(props.shape)
        return props.mean(1), props.std(1)
    
    def predict(self, s_t):
        with torch.no_grad():
            pre_dises = []
            for i in range(self.num_bootstrap):
                pre_dises.append(torch.nn.functional.softmax(self.checker_model[i](s_t[:, :self.action_len*2]), -1)[:, :, None])
            pre_dises = torch.cat(pre_dises, -1)
            confs = pre_dises.mean(-1)
            guess_dise = confs.max(-1, keepdim=True)[1]
        
        return guess_dise, confs
    
    def new_state(self, s_t, verbose=False):
        with torch.no_grad():
            prop_mean, prop_std = self.cal_prop_uncertainty(s_t[:, :66], s_t[:, 66:132], sample_num=50)
            prop_mean_maxval, prop_mean_maxind = prop_mean.max(1, keepdim=True)
            
            if verbose: print("Possible disease: %s, conf: %f"%(patience.dise_list[int(prop_mean_maxind.item())],\
                                                                prop_mean_maxval.item()))
            s_t = torch.cat((s_t, prop_mean), 1)
        return s_t, prop_mean, prop_std
    
    def final_action(self, s_t, act_ind, prop_mean, prop_std, verbose=False):
        prop_mean_future = prop_mean + prop_std*3
        prop_mean_maxval, prop_mean_maxind = prop_mean.max(1, keepdim=True)
        number = (prop_mean_future >= prop_mean_maxval).float().sum().item()
        
        if verbose:
            print("guess_confs is %s, future_confs is %s, number is %f"%(prop_mean[0].cpu().data.numpy(), prop_mean_future[0].cpu().data.numpy(), number))
    
        if number > 1:
            if verbose:
                print("Have better choice.")
            return act_ind + self.dise_len
        else:
            if verbose:
                print("No better choice.")
            return prop_mean_maxind[0].item()
            
    def choose_action(self, s_t, prop_mean, prop_std, test=False, verbose=False):
        with torch.no_grad():
            a_val = self.model(s_t)
            if (np.random.uniform() < self.epsilon or self.learn_step_counter < 1000) and not test:
                a = np.random.choice(np.argwhere(s_t[0, self.action_len:self.action_len*2].cpu().data.numpy() == 0).reshape(-1))
                if verbose: print("Choose symptom randomly: ", patience.sym_list[a])
            else:
                a = a_val.max(1)[1].view(-1).item()
                if verbose:
                    print("Choose symptom: ", patience.sym_list[a])
#                         print("q values: ", a_val[0].cpu().data.numpy(), " id: ", a)
            assert s_t[0, self.action_len+a].item() == 0, "q values: %s id: %d"%(a_val[0].cpu().data.numpy(), a)
            if verbose: print("Q values examples %s"%(a_val[0].cpu().data.numpy()[:5]))
            
        a_ = self.final_action(s_t, a, prop_mean, prop_std, verbose = verbose)
        
        return a_, a
    
    def forward(self, s_t, verbose=False):
        return self.model(s_t)
    
    def compute_loss(self, s_t, act, rew, done, s_tplus1):
        q_vals = self.model(s_t)
        q_val = q_vals.gather(1, act)  # because r is for act
        with torch.no_grad():
            q_vals_next = self.model(s_tplus1)
            q_vals_targ = rew + (1-done)*self.gamma*(q_vals_next.detach().max(1)[0].view(-1, 1))    # how about done?
        
        loss = self.criterion(q_val, q_vals_targ.detach())
        return loss
    
    def learn(self, batch_size):
        if (self.learn_step_counter % self.target_update_freq == 0):
#             print("Updating ... ")
            self.target_model.soft_update(self.model.state_dict(), self.soft_ratio)
            
        self.learn_step_counter += 1
        
        batch_trans = np.array(self.memory.sample(batch_size=batch_size))
        st, en = 0, self.state_len
        batch_s = torch.FloatTensor(batch_trans[:, st:en]).to(self.params['device'])
        st, en = en, en+1
        batch_a = torch.LongTensor(batch_trans[:, st:en].astype(np.int)).to(self.params['device'])
        st, en = en, en+1
        batch_r = torch.FloatTensor(batch_trans[:, st:en]).to(self.params['device'])
        st, en = en, en+1
        batch_d = torch.FloatTensor(batch_trans[:, st:en]).to(self.params['device'])
        st, en = en, en+self.state_len
        batch_s_ = torch.FloatTensor(batch_trans[:, st:]).to(self.params['device'])
        
        loss = self.compute_loss(batch_s, batch_a, batch_r, batch_d, batch_s_)
        
        self.optimizer.zero_grad()
        loss.backward()
        for p in self.model.parameters(): p.grad.data.clamp_(min=-5., max=5.)
        self.optimizer.step()
    
    def update_reclassifier(self, verbose=False):
        if (self.learn_step_counter % self.target_update_freq != 0):
            return
        if verbose:
            print("#### UPDATE CHECKER ####")
        input_data_input_mask, input_label = self.memory_checker.sample(self.batch_size)

        input_data = input_data_input_mask[:, :66]
        input_mask = input_data_input_mask[:, 66:]
    
        input_data_masked = torch.from_numpy(np.random.randint(2, size=input_data.shape).astype(np.float32)*input_data).to(self.params['device'])
        input_data = torch.from_numpy(np.array(input_data).astype(np.float32)).to(self.params['device'])
        input_mask = torch.from_numpy(np.array(input_mask).astype(np.float32)).to(self.params['device'])
        input_label = torch.from_numpy(np.array(input_label).astype(np.int)).to(self.params['device'])
        
        self.optimizer_reclassifier.zero_grad()
        output_logits = self.reclassifier(input_data_masked)[input_mask==1]
        mask_label = input_data[input_mask==1]
        ce_loss = self.criterion_class(output_logits, (mask_label+1).long())
        ce_loss.backward()
        self.optimizer_reclassifier.step()

        
    def update_checker(self, verbose=False):
        if (self.learn_step_counter % self.target_update_freq != 0):
            return
        
        for i in range(self.num_bootstrap):
            input_data_input_mask, input_label = self.memory_checker.sample(self.batch_size)
            input_data = input_data_input_mask[:, :66]
            input_mask = input_data_input_mask[:, 66:]

            input_data = torch.from_numpy(np.array(input_data).astype(np.float32)).to(self.params['device'])
            input_mask = torch.from_numpy(np.array(input_mask).astype(np.float32)).to(self.params['device'])
            input_label = torch.from_numpy(np.array(input_label).astype(np.int)).to(self.params['device'])


            self.optimizer_classifier3[i].zero_grad()
            ce_loss = self.criterion_class(self.classifier3[i](input_data), input_label)
            ce_loss.backward()
            self.optimizer_classifier3[i].step()

def eval_all_patience(doctor, patience, writer, i_episode_out, verbose=False, threshold=0.):
    patience.user_id = -1
    ep_r = 0
    ep_s = 0
    ep_t = 0
    ep_ts = 0
    ep_o = 0
    ep_repeat = 0
    ep_conf = 0
    ep_num = 0
    ep_num_valid = 0
    ep_s_force = 0
    for i_episode in range(len(patience.goals)):
        obs = patience.reset()
        if verbose:
            print("########## id %d with dise %s ##########"%(patience.user_id, patience.dise_list[patience.dise]))
            print("[Usr] %s"%(np.array(patience.sym_list)[obs[66:132] != 0]))
        while True:
            obs_tensor = torch.from_numpy(obs.reshape(1, -1)).cuda()
            obs_tensor, prop_mean, prop_std = doctor.new_state(obs_tensor)
            a_act, _ = doctor.choose_action(obs_tensor, prop_mean, prop_std, test=True, verbose=False)
            
            if a_act < doctor.dise_len:
                if verbose: print("[Doc(%s, %.2f)] You got %s"%(patience.dise_list[a_act], prop_mean[0, a_act], patience.dise_list[a_act]))
            else:
                if verbose: print("[Doc(%s, %.2f)] Do you have %s"%(patience.dise_list[prop_mean.max(1)[1].item()], prop_mean.max(1)[0].item(), patience.sym_list[a_act-4]))
                
            obs_, r, done, info = patience.step(a_act)
            
            if verbose and not done: print("[Usr] %s"%({-1: "No", 0: "Not-sure", 1:"Yes"}[obs_[a_act-4]]))
            
            ep_r += r
            if 'repeat' in info and info['repeat']:
                ep_repeat += 1
            
            if done:
                ep_t += info['runs']
                if info['success']:
                    ep_s += 1
                if 'leave' not in info or not info['leave']:
                    ep_num_valid += 1
                    ep_ts += info['runs']
                ep_s_force += (prop_mean.max(1)[1].item() == patience.dise).item()
                ep_o += obs_[66:66*2].sum()
                ep_conf += prop_mean.max(1)[0].item()
                ep_num += 1
                break
            
            obs = obs_
    
    print("%s: numb %d | avg conf is %.2f | avg success is %.3f | avg success valid is %.3f | avg success force is %.3f| avg run is %.2f | avg repeat is %.2f | avg obs is %.2f"%\
          (patience.mode, ep_num, ep_conf/ep_num, ep_s/ep_num, (ep_s/ep_num_valid if ep_num_valid > 0 else 0), ep_s_force/ep_num, ep_t/ep_num, ep_repeat, ep_o/ep_num))
    
    logging.info("acc %.4f turn %.4f acc_valid %.4f turn_valid %.4f, acc_invalid %.4f valid %d conf_mean %s conf_std %s"%\
                (ep_s_force/ep_num, ep_t/ep_num, (ep_s/ep_num_valid if ep_num_valid > 0 else 0),\
                 (ep_ts/ep_num_valid if ep_num_valid > 0 else 0), ep_s/ep_num, ep_num_valid, prop_mean[0].data.cpu().numpy(), prop_std[0].data.cpu().numpy()))
    
    return ep_s_force/ep_num


