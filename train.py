import Agent.PPO as PPO
import Agent.MP as MP
import Models.models as models 
import MagnetEnv as env
import yaml
import argparse
import torch
from torch.utils.tensorboard import SummaryWriter



def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of model described in the paper: Proximal Policy Optimization Algorithms""")
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--lr_c', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor for rewards')
    parser.add_argument('--lambda_', type=float, default=0.95, help='parameter for GAE')
    parser.add_argument('--beta', type=float, default=0.01, help='entropy coefficient')
    parser.add_argument('--epsilon', type=float, default=0.3, help='parameter for Clipped Surrogate Objective')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=int(1e6))
    parser.add_argument('--pic_size', type=tuple, default=(96, 96))
    parser.add_argument('--train_times', type=int, default=10)          # 平均每个数据采样学习多少次
    parser.add_argument("--num_processes", type=int, default=8)
    parser.add_argument("--save_interval", type=int, default=50, help="Number of steps between savings")
    parser.add_argument("--plot_interval", type=int, default=10, help="Number of steps between plot result")
    parser.add_argument("--max_actions", type=int, default=500, help="Maximum repetition steps in test phase")
    parser.add_argument("--name", type=str, default="MINI_CNN_FC")
    parser.add_argument("--map", type=str, default="real.json")
    # parser.add_argument("--end_points", type=tuple, default=((270,100),))
    parser.add_argument("--end_points", type=tuple, default=((24, 531),(310, 485),(362, 90),(567, 331)))
    parser.add_argument("--radius", type=float, default=6.5)
    args = parser.parse_args()
    return args

# 读取现有模型已训练的世代数
def load_epo(PATH):
	try:
		with open(PATH, "r") as f:
			epo = yaml.load(f, Loader=yaml.SafeLoader)
		print("epo:", epo)
	except:
		epo = 0
	return epo



# 将当前世代数写入硬盘
def write_epo(PATH, epo):
	with open(PATH, 'w', encoding='utf-8') as f:
		yaml.dump(epo, f)

def eval_onece(env, opt, agent,):
	s = env.reset()
	reward_total = 0
	spend_steps  = 0
	for i in range(opt.max_actions):
		spend_steps += 1
		a, _, _ = agent.select_action_return_v(s)
		s1, r, d, debug = env.step(a, render=True)
		s = s1
		reward_total += r
		if d:
            
			break
	return reward_total, spend_steps, s


def main():
    opt = get_args()
    writer = SummaryWriter(log_dir=f"runs/{opt.name}")

    envs = MP.Mult_Envs_Stepper(num_envs=opt.num_processes, opt=opt)
    model = models.Mario(input_channels=1, act_num=5)
    agent = PPO.PPO_agent(opt)
    agent.load_model(model)
    try:
        agent.load_weights(f"Weights/{opt.name}.pt")
        print("load Model")
    except:
        print("New Model")
    begin_epo = load_epo(f".\\Weights\\{opt.name}.yaml")
    evel_env = MP.sub_train_env(opt)

    # 向所有环境发布指令：重置！
    [agent_conn.send(("reset", None)) for agent_conn in envs.agent_conns]
    # 从所有环境回收初始状态
    curr_states = [agent_conn.recv() for agent_conn in envs.agent_conns]
    # 保存的损失函数
    losses = []
    for epo in range(begin_epo, begin_epo+opt.num_epochs):
        for i in range(opt.max_actions):
            # 获得一个列表，每个元素是其对应环境的动作，概率，价值。			
            apv = [agent.select_action_return_v(s) for s in curr_states]		
            # 在所有子环境中执行一次动作
            for agent_conn, (a, _, _) in zip(envs.agent_conns, apv):
                agent_conn.send(("step", a))
            # 从所有子环境中接收数据
            s1_r_d_bug = [agent_conn.recv() for agent_conn in envs.agent_conns]
            # 保存单步的数据
            for k, i in enumerate(envs.env_datas):
                s=curr_states[k]
                a=apv[k][0]
                p=apv[k][1]
                r=s1_r_d_bug[k][1]
                v=apv[k][2]
                d=s1_r_d_bug[k][2]
                i.pack_step_data(s, a, p, r, v, d)	
            curr_states = [data[0] for data in s1_r_d_bug]		
        # 提取所有的数据
        states, acts, rewards, probs, values, Gs, advantages = envs.collect_all_datas()
        # 训练神经网络
        losses = agent.learn(states, acts, rewards, probs, Gs, advantages)
        # 清空所有的数据
        [datas.clear() for datas in envs.env_datas]


        if losses is not None:
            writer.add_scalar('Loss/actor', losses[0], epo)
            writer.add_scalar('Loss/critic', losses[1], epo)
            writer.add_scalar('Loss/entropy', losses[2], epo)
            writer.add_scalar('Loss/all', losses[3], epo)


        if ((epo % opt.plot_interval == 0)):
            total, steps, img = eval_onece(evel_env, opt, agent)
            writer.add_scalar('Reward/reward', total, epo)
            writer.add_scalar('Reward/Spend Steps', steps, epo)
            writer.add_image("Last_result Image", s.reshape(opt.pic_size[0], opt.pic_size[1]), 
                global_step=epo, dataformats="HW",)
            
            torch.save(agent.model.state_dict(), f".\\Weights\\{opt.name}.pt")

            print("weights saved")
            write_epo(f".\\Weights\\{opt.name}.yaml", epo)
            print(f"# of episode :{epo}, score : {total}, steps :{steps}")

        if ((epo % opt.save_interval == 0)):
            torch.save(agent.model.state_dict(), f".\\Weights\\time_capsule\\{opt.name}_{epo}.pt")


if __name__ == '__main__':
	main()