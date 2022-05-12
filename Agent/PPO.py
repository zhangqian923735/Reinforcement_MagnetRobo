"""
智能代理
"""
import numpy as np
import collections
import rando0218
import torch
import torch.nn as nn
import torch.nn.functional as F




class PPO_agent():
	def __init__(self, opt):
		self.device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		print("cuda:0" if torch.cuda.is_available() else "cpu")
		
		self.lr = opt.lr               # Actor学习率
		self.c =  opt.lr_c             # Critic学习率倍数
		self.gamma = opt.gamma         # 奖励衰减系数
		self.lambda_ = opt.lambda_     # GAE系数
		self.beta = opt.beta           # 熵损失系数
		self.epsilon = opt.epsilon            # PPO裁剪系数
		self.batch_size = opt.batch_size      # 每次训练个数
		self.train_times = opt.train_times    # 每组训练次数

		self.model = None      # 主网络
		self.optm  = None      # 构建优化器
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       

	def load_model(self, net):
		self.model = net.to(self.device)
		self.optm = torch.optim.Adam(self.model.parameters(), lr=self.lr)


	def select_action_return_v(self, s):  # 概率决策，返回选择的动作以及此动作被选中的概率
		s = torch.tensor(s).unsqueeze(dim=0).type(torch.float32).to(self.device)
		self.model.eval()
		predict_prob, value = self.model(s)
		# 依概率选择动作
		action = torch.multinomial(predict_prob, 1)[0]
		return int(action), float(predict_prob.squeeze()[int(action)]), float(value)


	def best_action(self, s):  # 贪心决策
		s = torch.tensor(s).unsqueeze(dim=0).type(torch.float32).to(self.device)
		self.model.eval()
		predict, _ = self.model(s)[0]
		return int(torch.argmax(predict, dim=0))


	def learn(self, s, a, r, old_probs, G, advantages):
		if self.optm.defaults["lr"] != self.lr:
			self.optm = torch.optim.Adam(self.model.parameters(), lr=self.lr)  # 构建优化器
		self.model.train()  # 【好习惯】将模型设置为训练模式	          
		
		# 数据整理
		s  = torch.from_numpy(s).type(torch.float32)
		a  = torch.tensor(a).type(torch.int64).reshape(-1, 1).to(self.device)
		r  = torch.tensor(r).type(torch.float32).reshape(-1, 1).to(self.device)
		G  = torch.tensor(G).type(torch.float32).reshape(-1, 1).to(self.device)
		advantages= torch.tensor(advantages).reshape(-1, 1).type(torch.float32).to(self.device)
		old_probs = torch.tensor(old_probs).type(torch.float32).reshape(-1, 1).to(self.device)
		

		avg_losses = [0,0,0,0]  # 将此次计算得到的平均损失输出用于记录
		times = round(self.train_times * len(s) / self.batch_size)
		for i in range(times):
			indice = torch.randperm(len(s))[:self.batch_size]  # 随机采样一部分
			s_batch = s[indice]
			a_batch = a[indice]
			G_batch = G[indice]
			advantages_batch = advantages[indice]
			old_probs_batch = old_probs[indice]
			
			s_batch = s_batch.to(self.device)
			new_probs, values = self.model(s_batch)
			# π(At|St, θ) / π_old(At|St, θ)
			ratio = torch.gather(new_probs, 1, a_batch) / old_probs_batch
			surr1 = ratio * advantages_batch
			# 通过裁剪 π(At|St, θ) / π_old(At|St, θ) 到1的附近，限制过大的梯度更新，来自PPO论文
			surr2 = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * advantages_batch
			# 更新较小的那一个梯度
			actor_loss = - torch.mean(torch.min(surr1, surr2))
			# 基线价值函数的损失函数，希望与GAE收益较为接近
			critic_loss = self.c * F.smooth_l1_loss(G_batch, values)
			# 熵损失的计算公式 
			# 熵损失比较大，则鼓励智能体保持足够的探索
			# 熵损失比较小，则鼓励智能体输出结果更加确定
			entropy = torch.mean(torch.sum(-new_probs * torch.log(torch.clamp(new_probs, min=1e-5)), axis=1))
			entropy_loss = -self.beta * entropy
			
			Final_loss = actor_loss + critic_loss + entropy_loss
			
			self.optm.zero_grad()
			Final_loss.backward()
			# 梯度裁剪，减轻过拟合以及梯度爆炸
			torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
			self.optm.step()
			avg_losses[0] += float(actor_loss) / times
			avg_losses[1] += float(critic_loss) / times
			avg_losses[2] += float(entropy) / times
			avg_losses[3] += float(Final_loss) / times
		return avg_losses



	def syn_models(self, from_model, to_model, tau):
		# 按比例复制网络的权值： 子网络 <-- tau * 子网络 + （1 - tau）* 主网络
		dic = from_model.state_dict()
		for i in to_model.state_dict():
			# i 是字典的索引名。 使用 += 来赋值可以替换源地址的内容，运行效率高
			to_model.state_dict()[i] += (tau - 1) * to_model.state_dict()[i] + (1. - tau) * dic[i]


	def load_weights(self, PATH):
		self.model.load_state_dict(torch.load(PATH))
			






if __name__ == '__main__':
	import gym
	
