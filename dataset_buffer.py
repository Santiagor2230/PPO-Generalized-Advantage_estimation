from torch.utils.data.dataset import IterableDataset
import torch
import random
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class RLDataset(IterableDataset):
  
  def __init__(self, env, policy, value_net, samples_per_epoch, gamma, 
               lamb, repeats):
    self.env = env
    self.policy = policy
    self.value_net = value_net
    self.samples_per_epoch = samples_per_epoch
    self.gamma = gamma
    self.lamb = lamb
    self.repeats = repeats
    self.obs = self.env.reset()

  @torch.no_grad()
  def __iter__(self):
    transitions = []
    for step in range(self.samples_per_epoch):
      loc, scale = self.policy(self.obs)
      action = torch.normal(loc, scale)
      next_obs, reward, done, info = self.env.step(action)
      transitions.append((self.obs, loc, scale, action, reward, done, next_obs))
      self.obs = next_obs

    transitions = map(torch.stack, zip(*transitions)) #for each it turns into a tensor such as scale, reward , etc...
    obs_b, loc_b, scale_b, action_b, reward_b, done_b, next_obs_b = transitions
    '''reward_b and done_b have a shape of (samples_per_epoch, num_envs)
    we want it to be (Samples_per_epoch, num_envs, 1)'''
    reward_b = reward_b.unsqueeze(dim=-1) #unsqueeze creates new dimension
    done_b = done_b.unsqueeze(dim=-1)

    values_b = self.value_net(obs_b) #v(s|a)
    next_values_b = self.value_net(next_obs_b) #v(s'|a')

    '''temporal difference error -> td = r + y*v(s') - v(s)
    td_error_b=[reward1,reward2,reward3] + [1*gamma,1*gamma,1*gamma] * 
    [next_value1,next_value2, next_value3] - [value1, value2, value3]
    '''
    td_error_b = reward_b + (1- done_b) * self.gamma * next_values_b - values_b

    running_gae = torch.zeros((self.env.num_envs, 1), dtype=torch.float32, device=device)
    gae_b = torch.zeros_like(td_error_b)

    for row in  range(self.samples_per_epoch - 1, -1, -1):
      '''Genarelized advantage estimation'''
      running_gae = td_error_b[row] + (1-done_b[row]) * self.gamma *self.lamb * running_gae
      gae_b[row] = running_gae #advantage estimate a matrix rows

    target_b = gae_b + values_b # expected sum of reward A + V(s|a) rows of return + rows of values

    num_samples = self.samples_per_epoch * self.env.num_envs
    reshape_fn = lambda x: x.view(num_samples, -1)
    batch = [obs_b, loc_b, scale_b, action_b, reward_b, gae_b, target_b]

    obs_b, loc_b, scale_b, action_b, reward_b, gae_b, target_b = map(reshape_fn, batch)

    for repeat in range(self.repeats):
      idx = list(range(num_samples))
      random.shuffle(idx)

      for i in idx:
        yield obs_b[i], loc_b[i], scale_b[i], action_b[i], reward_b[i], gae_b[i], target_b[i]