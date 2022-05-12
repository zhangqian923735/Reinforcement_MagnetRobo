"""
多进程实现
"""

import sys 
sys.path.append("..") 
import json
import pygame
import numpy as np
import random
import torch
import torch.multiprocessing as mp
import Agent.PPO as PPO
import MagnetEnv
import cv2
from torch.utils.tensorboard import SummaryWriter


def img_resize(img, shape):
	# cv2的 x, y 坐标与 np 的相反
	img = cv2.resize(img, (shape[1], shape[0]))
	img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	img = img.reshape(1, shape[0], shape[1])
	return 2. * img / 255 - 1


def select_action(model, s, device="cuda:0"):
	s = torch.tensor(s).unsqueeze(dim=0).type(torch.float32).to(device)
	model.eval()
	predict_prob, _ = model(s)
	# 依概率选择动作
	action = torch.multinomial(predict_prob, 1)[0]
	return int(action), float(predict_prob.squeeze()[int(action)])



# 一个与环境交互并训练网络的进程
# 用法：复制权值，玩一局，训练，覆盖权值。
def train_runner(opt, model, globle_epo, device="cuda:0"):	
	agent = PPO.PPO_agent()
	# 创建一个本地模型的张量副本
	agent.load_model(model)
	begin_epo = int(globle_epo)
	agent.lr = opt.lr               # Actor学习率
	agent.c =  opt.lr_c             # Critic学习率倍数
	agent.gamma = opt.gamma         # 奖励衰减系数
	agent.lambda_ = opt.lambda_     # GAE系数
	agent.beta = opt.beta           # 熵损失系数
	agent.epsilon = opt.epsilon            # PPO裁剪系数
	agent.batch_size = opt.batch_size      # 每次训练个数
	agent.train_times = opt.train_times    # 每组训练次数
	epoch = opt.num_epochs                 # 训练次数
	writer = SummaryWriter(log_dir=f"runs/{opt.name}")

	s_list = []              # 状态容器
	a_list = []              # 动作容器
	r_list = []              # 奖励容器
	prob_list = []           # 动作概率容器
	done_list = []           # 环境结束标识容器

	losses = None

	input_shape = opt.pic_size
	env = MagnetEnv.MagnetEnv()
	with open(opt.map) as json_file:
		env.wall_poslist = json.load(json_file)
	
	state = env.reset()
	state = img_resize(state, input_shape)
	# state = torch.from_numpy(state)
	total = 0

	for epo in range(opt.num_epochs):
		globle_epo += 1
		# 每轮开始时拷贝权值
		agent.model.load_state_dict(model.state_dict())
		total = 0
		for i in range(opt.max_actions + 1):
			a, prob = agent.select_action(state)
			s_next, r, d, debug = env.step(a)
			s_list.append(state)
			a_list.append(a)
			r_list.append(r)
			prob_list.append(prob)
			done_list.append(d)
			
			state = s_next
			total += r
			state = img_resize(state, input_shape)
			# state = torch.from_numpy(state)
			if int(globle_epo) % 1 == 0:
				env.render()
			
			if d:
				env.close()
				state = env.reset()
				curr_step = 0
				state = img_resize(state, input_shape)
				break

		if len(s_list) >= agent.batch_size:
			losses = agent.learn(s_list, a_list, r_list, prob_list, done_list)
			print(f"# of episode :{int(globle_epo)}, score : {total}, steps :{debug[2]}")
			model.load_state_dict(agent.model.state_dict())
			s_list = []
			a_list = []
			r_list = []
			prob_list = []
			done_list = []


		if losses is not None:
			writer.add_scalar('Reward/reward', total, int(globle_epo))
			writer.add_scalar('Reward/Spend Steps', debug[2], int(globle_epo))
			writer.add_scalar('Loss/actor', losses[0], int(globle_epo))
			writer.add_scalar('Loss/critic', losses[1], int(globle_epo))
			writer.add_scalar('Loss/entropy', losses[2], int(globle_epo))
			writer.add_scalar('Loss/all', losses[3], int(globle_epo))


		if ((int(globle_epo) % opt.save_interval == 0) and (int(globle_epo) != begin_epo)):		
			writer.add_image("Last_result Image", state.reshape(input_shape[0], input_shape[1]), 
			global_step=int(globle_epo), dataformats="HW",)




class sub_train_env:
	"""docstring for sub_train_env"""
	def __init__(self, opt):
		self.opt = opt
		self.env = MagnetEnv.MagnetEnv()
		self.env.max_steps = opt.max_actions

	# 重启环境
	def reset(self):
		
		self.opt.end_distance
		angle = random.randint(0, 628) / 100
		pos_ball = tuple([int(i/2) for i in self.env.engine.size])
		end_pos = tuple([
			int(pos_ball[0] + np.sin(angle) * self.opt.end_distance),
			int(pos_ball[1] + np.cos(angle) * self.opt.end_distance),
		]) 
		self.env.close()
		state = self.env.reset(pos_ball=pos_ball, pos_target=end_pos)
		state = img_resize(state, self.opt.pic_size)
		return state


	# 环境执行一步
	def step(self, a, render=False):
		# 环境任务如果已经完成则重置
		s_next, r, d, debug = self.env.step(a)
		if render:
			self.env.render()
		if d:
			self.env.close()
			s_next = self.reset()
			return s_next, r, d, debug
		
		s_next = img_resize(s_next, self.opt.pic_size)
		return s_next, r, d, debug


# 存储每个独立游戏的数据
class Sub_datas:
	def __init__(self, opt):	
		self.opt = opt
		self.datas = {
			"s":[],
			"a":[],
			"p":[],
			"r":[],
			"v":[],
			"d":[],
		}

	# 存储单步的数据
	def pack_step_data(self, s, a, p, r, v, d):
		self.datas["s"].append(s)
		self.datas["a"].append(a)
		self.datas["p"].append(p)
		self.datas["r"].append(r)
		self.datas["v"].append(v)
		self.datas["d"].append(d)

	def needed_data_for_train(self):
		s = np.array(self.datas["s"])
		a = self.datas["a"].append(a)
		p = self.datas["p"].append(p)
		r = self.datas["r"].append(r)
		g, adv = self.calc_GAE()
		return s,a,r,p,g,adv

	# 转换为GAE
	def calc_GAE(self):
		s = self.datas["s"]
		r = self.datas["r"]
		d = self.datas["d"]
		v = self.datas["v"]
		next_value = 0.
		
		G_list = []
		advantages = []
		GAE = 0
		for state, reward, done, value in list(zip(s, r, d, v))[::-1]:	
			GAE = GAE * self.opt.lambda_ * self.opt.gamma
			GAE += reward + self.opt.gamma * next_value * (1. - done) - value  # <-- 这个是gae优势值
			next_value = value
			advantages.insert(0, GAE)       # <-- 这个是gae优势值
			G_list.insert(0, GAE + value)   # <-- 这里储存的是折算后总收益，没有减去基线的
		return G_list, advantages


	# 清空数据
	def clear(self):
		del self.datas
		self.datas = {
			"s":[],
			"a":[],
			"p":[],
			"r":[],
			"v":[],
			"d":[],
		}
		
class Mult_Envs_Stepper:
	def __init__(self, num_envs, opt):
		# 创建一堆通讯管道,每个子环境一个
		self.agent_conns, self.env_conns = zip(*[mp.Pipe() for _ in range(num_envs)])
		# 创建子环境的进程
		self.opt = opt
		# 子环境的交互数据
		self.env_datas = [Sub_datas(self.opt) for _ in range(num_envs)]
		for index in range(num_envs):
			process = mp.Process(target=self.run, args=(index, self.opt))
			process.start()
			# self.env_conns[index].close()

	# 子环境进程，每一个维护一个env
	def run(self, index, opt):
		# self.agent_conns[index].close()	
		Own_env = sub_train_env(opt)
		while True:
			request, action = self.env_conns[index].recv()
			if request == "step":
				self.env_conns[index].send(Own_env.step(action, self.opt.render_all))
			elif request == "reset":
				self.env_conns[index].send(Own_env.reset())
			else:
				raise NotImplementedError

	# 将所有子环境的数据汇总
	def collect_all_datas(self):
		s = []
		a = []
		r = []
		p = []
		v = []
		g = []
		adv = []
		for data_class in self.env_datas:
			s += data_class.datas["s"]
			a += data_class.datas["a"]
			r += data_class.datas["r"]
			p += data_class.datas["p"]
			v += data_class.datas["v"]
			tg, tadv = data_class.calc_GAE()
			g += tg
			adv += tadv		
		s = np.array(s)
		return s,a,r,p,v,g,adv




if __name__ == "__main__":
    pass