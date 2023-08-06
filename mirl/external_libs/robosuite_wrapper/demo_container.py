# import robosuite_container
import matplotlib.pyplot as plt
from PIL import Image

from mirl.external_libs.robosuite_wrapper import robosuite_container

def save_fig(obs, step):
    fig = plt.figure()
    img = Image.fromarray(obs.transpose(1,2,0)[:,:,:3])
    plt.imshow(img)
    plt.axis('off')
    plt.savefig(f"/home/qzhou/temp/{step}.jpg", bbox_inches='tight', dpi=fig.dpi, pad_inches=0.0)


env = robosuite_container.make(
    env_name="Door", # ["Door", "TwoArmPegInHole", "NutAssemblyRound", "TwoArmLift"]
    robots="Panda", # ["Panda", "Kinova3", "Jaco", "Sawyer", "IIWA", "UR5e"]
    mode="eval", # ["train", "eval-easy", "eval-hard", "eval-extreme"]
    episode_length=500,
    image_height=168,
    image_width=168,
    randomize_camera=True,
    randomize_color=True,
    randomize_light=True,
    randomize_dynamics=False,
    seed=2
)

obs = env.reset()
num_step = 0
total_reward = 0
for i in range(10):
    save_fig(obs, num_step)
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    num_step += 1
    total_reward += reward
    if done:
        print(f"The episode terminates after {num_step} steps, total reward is {total_reward}.")
        break

print(env.action_space)
obs = env.reset()
num_step = 0
total_reward = 0
for i in range(10):
    save_fig(obs, num_step+100)
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    print(obs.shape, reward)
    num_step += 1
    total_reward += reward
    if done:
        print(f"The episode terminates after {num_step} steps, total reward is {total_reward}.")
        break