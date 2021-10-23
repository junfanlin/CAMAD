from nips2019_direct import *


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=0, type=int)
args = parser.parse_args()

params = {}
params['device'] = 'cuda'
params['num_episode'] = 200000
# params['update_frequence'] = 1
params['lr'] = 1e-3
params['memory_size'] = 50000
params['memory_type'] = "list" #"random" or "list"
params['gamma'] = 0.95
params['epsilon'] = 0.1
params['soft_ratio'] = 0.99
params['target_update_freq'] = 10
params['elim_update_freq'] = 10
params['update_steps'] = [100000, 200000]
params['batch_size'] = 32
params['test_freq'] = 1000
params['elim_branch'] = 'ISE'
params['max_turn'] = 22
params['verbose'] = False
params['conf_threshold'] = 0.97
params['random_start'] = 40*128
params['num_bootstrap'] = 1
params['sample_num'] = 50
params['dueling'] = True
params['seed'] = args.seed
params['block'] = True

params['logger_name'] = 'my_all_test_data_bgall_block3-22_direct_'+str(params['seed'])+'_bknoboot'#'my_all_block_log'
import logging
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(filename=params['logger_name'], level=logging.DEBUG, format=LOG_FORMAT, filemode='w')

random.seed(params['seed'])
np.random.seed(params['seed'])
torch.manual_seed(params['seed'])

patience = MuzhiEnv(data_bg='all', tol=params['max_turn'])
patience_test = MuzhiEnv(mode='test', data_bg='test_data_bgall', tol=params['max_turn']+2)
patience.seed(params['seed'])
patience_test.seed(params['seed'])

doctor = Agent(state_len=66*2+4, action_len=66, dise_len=4, emb_len=24, params=params).to(params['device'])

best_acc = 0
action_ratio = np.zeros(66+4)
for i_episode in range(params['num_episode']):
    ep_r = 0
    ep_h = 0
    verbose = params['verbose']
    if (i_episode % 200) == 0:
#     if i_episode == 800:
        print('#################', i_episode, '#####################')
        verbose = True
    
#     logging.info("running %d"%(i_episode))
    done = False
    obs = patience.reset(verbose=verbose, allow_rep=params['block'])
    obs_tensor = torch.from_numpy(obs.reshape(1, -1)).to(params['device'])
    obs_tensor, prop_mean, prop_std = doctor.new_state(obs_tensor)
    conf, guess_dise = prop_mean.max(1)
    obs = obs_tensor.cpu().data.numpy()
        
    if guess_dise.item() != patience.dise and verbose:
        print('##### Wrong predict %s/%s'%(patience.dise_list[guess_dise.item()], patience.dise_list[patience.dise]))
        
    while not done:
        if verbose: print("Possible dise %s with conf %.2f"%(patience.dise_list[guess_dise], conf.item()))
        a_act, a_act_ori = doctor.choose_action(obs_tensor, prop_mean, prop_std, verbose=False)
        obs_, r, done, info = patience.step(a_act, verbose=verbose)
        obs_tensor = torch.from_numpy(obs_.reshape(1, -1)).to(params['device'])
        
        obs_tensor, prop_mean, prop_std = doctor.new_state(obs_tensor)
        conf_, guess_dise = prop_mean.max(1)
        obs_ = obs_tensor.cpu().data.numpy()
            
        action_ratio[a_act] += 1
            
        if done:
            if info["success"]:
                if verbose:
                    print("CORRECT!")
            elif 'leave' in info and info['leave']:
                if verbose:
                    print("TOO MUCH!")
            else:
                if verbose:
                    print("WRONG!")
        
        r = -0.1
        doctor.memory.push(obs[0], a_act_ori, r, done, obs_[0])
        if a_act < doctor.dise_len:
            if verbose: print("Inform Dise %s with conf %.2f"%(patience.dise_list[a_act], conf.item()))
        
        if 'hit' in info:
            ep_h += 1 if info['hit'] else 0
        
        ep_r += r
        
        if len(doctor.memory) > params['random_start']:
            doctor.learn(params['batch_size'])
        
        if done:

            if True:
                if verbose:
                    print(obs[0, :66], patience.dise, len(doctor.memory_checker))
                doctor.memory_checker.push(obs[0, :132], patience.dise)
                if patience.dise != guess_dise:
                    doctor.memory_checker.push(obs[0, :132], patience.dise)
                if len(doctor.memory_checker) > params['batch_size']:
#                     if verbose:
                    doctor.update_checker(verbose=verbose)
            
            doctor.scheduler.step()
            break
        
        obs = obs_
        conf = conf_
    
    if ((i_episode) % params['test_freq'] == 0):
        print("At iter %d, epsilon %.3f, learning rate %.1e "%(i_episode, doctor.epsilon, doctor.optimizer.param_groups[0]['lr']))
#         eval_all_patience(doctor, patience_train_test, writter, i_episode)
        test_acc = eval_all_patience(doctor, patience_test, None, i_episode)
        if test_acc >= best_acc:
            best_acc = test_acc
        print((doctor.classifier3[0](torch.from_numpy(patience_test.patient.sympbase).cuda()).max(1)[1] == torch.from_numpy(patience_test.patient.disebase).cuda()).float().mean())
