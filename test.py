from gym_torcs import TorcsEnv


max_episode = 5

env = TorcsEnv(path='/usr/local/share/games/torcs/config/raceman/quickrace.xml')

ep = 0
done = True

while ep < max_episode:
    if done:
        ep += 1
        ob = env.reset(relaunch=True, sampletrack=False, render=True)

    state, reward, done, _ = env.step([0, 0, 0])

env.close()
