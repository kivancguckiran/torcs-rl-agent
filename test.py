from gym_torcs import TorcsEnv


max_episode = 100

env = TorcsEnv(path='/usr/local/share/games/torcs/config/raceman/quickrace.xml')

ep = 0
done = True

while ep < max_episode:
    if done:
        ep += 1
        ob = env.reset(relaunch=True, sampletrack=False, render=True)

    state, reward, done, _ = env.step(env.action_space.sample())

env.close()
