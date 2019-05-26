from gym import spaces
from env import snakeoil3_gym as snakeoil3
import numpy as np
import copy
import os
import time
import xml.etree.ElementTree as ET

from env.utils import sample_track
from env.utils import set_render_mode
from env.utils import sigmoid


class TorcsEnv:
    """
    Gym torcs environment.

    Start a Torcs process which wait for client(s) to
    connect through a socket connection. Environment sets
    the connection and communicates with the game whenever
    step, reset or constructor is called.

    Note: In order to change the track randomly
    at reset, you need to feed the path of the
    game's quickrace.xml file. If you installed the
    game as a root user it can be found at:
    "/usr/local/share/games/torcs/config/raceman"
    Also you may need to change the permissions
    for the file in order to modify it through
    the environment's reset method.


    Arguments:
        port: port the game will wait for connection
        path: path of the "quickrace.xml" file

    Methods:
        step: send the given action container(list, tuple
            or any other container). Action needs to have
            3 elements.
        reset: send a message to reset the game.

    """
    terminal_judge_start = 250  # Speed limit is applied after this step
    termination_limit_progress = 5/200  # [km/h], episode terminates if car is running slower than this limit
    backward_counter = 0

    initial_reset = True

    def __init__(self, port=3101, path=None, reward_type='original', track='none', client_mode=False):
        self.port = port
        self.client_mode = client_mode
        self.initial_run = True
        self.reward_type = reward_type
        self.reset_counter = 0
        self.reset_torcs()
        
        if path:
            self.tree = ET.parse(path)
            self.root = self.tree.getroot()
            self.path = path

        self.track = track

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,))
        high = np.concatenate([
            np.array([1.0]),
            np.ones(19),
            np.array([1.0]),
            np.array([1.0]),
            np.array([1.0]),
            np.array([1.0]),
            np.ones(4),
            np.array([1.0]),
        ])
        low = np.concatenate([
            np.array([-1.0]),
            np.ones(19)*-1/200,
            np.array([-1.0]),
            np.array([-1.0]),
            np.array([-1.0]),
            np.array([-1.0]),
            np.zeros(4),
            np.array([0.0]),
        ])
        self.observation_space = spaces.Box(low=low, high=high)

    def step(self, u):
        assert self.initial_reset == False, "Call the reset() function before step() function!"
        client = self.client

        this_action = self.agent_to_torcs(u)

        # Apply Action
        action_torcs = client.R.d

        # Steering
        action_torcs["steer"] = this_action["steer"]
        action_torcs["brake"] = this_action["brake"]
        action_torcs["accel"] = abs(this_action["accel"])

        if this_action["accel"] < 0:
            action_torcs['gear'] = -1
        else:
            action_torcs['gear'] = 1
            if client.S.d['speedX'] > 50:
                action_torcs['gear'] = 2
            if client.S.d['speedX'] > 80:
                action_torcs['gear'] = 3
            if client.S.d['speedX'] > 110:
                action_torcs['gear'] = 4
            if client.S.d['speedX'] > 140:
                action_torcs['gear'] = 5
            if client.S.d['speedX'] > 170:
                action_torcs['gear'] = 6

        # Save the privious full-obs from torcs for the reward calculation
        obs_pre = copy.deepcopy(client.S.d)

        # One-Step Dynamics Update #################################
        # Apply the Agent's action into torcs
        client.respond_to_server()
        # Get the response of TORCS
        client.get_servers_input()

        # Get the current full-observation from torcs
        obs = client.S.d

        self.last_obs = copy.deepcopy(obs)
        self.last_speed = np.sqrt(obs['speedX']**2 + obs['speedY']**2)


        # Make an obsevation from a raw observation vector from TORCS
        self.observation = self.make_observaton(obs)

        episode_terminate = False
        info = {}
        # Reward setting Here #######################################
        # direction-dependent positive reward
        track = np.array(obs['track'])
        sp = np.array(obs['speedX'])/200
        progress = sp * np.cos(obs['angle'])

        if self.reward_type == 'original':
            reward = progress - np.abs(sp * np.sin(obs["angle"])) - sp * np.abs(obs['trackPos']) / 5
        elif self.reward_type == 'no_trackpos':
            reward = progress - np.abs(sp * np.sin(obs["angle"]))
        elif self.reward_type == 'custom_trackpos':
            reward = progress - np.abs(sp * np.sin(obs["angle"])) - sp * (obs['trackPos'] ** 2) / 5
        elif self.reward_type == 'trackpos':
            reward = progress - np.abs(sp * np.sin(obs["angle"])) - sp * np.abs(obs['trackPos'])
        elif self.reward_type == 'no_penalty':
            reward = progress
        elif self.reward_type == 'speed':
            reward = sp - np.abs(sp * np.sin(obs["angle"]))
        elif self.reward_type == 'endtoend': # https://team.inria.fr/rits/files/2018/02/ICRA18_EndToEndDriving_CameraReady.pdf
            reward = sp * (np.cos(obs['angle']) - np.abs(obs['trackPos']))
        elif self.reward_type == 'extra': # https://github.com/bhanuvikasr/Deep-RL-TORCS/blob/master/report.pdf
            Vx = obs['speedX'] / 200
            Vy = obs['speedY'] / 200
            trackpos = np.abs(obs['trackPos'])
            sintheta = np.abs(np.sin(obs['angle']))
            costheta = np.cos(obs['angle'])
            reward = Vx * costheta - Vx * sintheta - Vx * trackpos * sintheta - Vy * costheta
        elif self.reward_type == 'extra_github':
            speedX = obs['speedX'] / 200
            speedY = obs['speedY'] / 200
            reward = speedX * np.cos(1.0 * obs['angle']) \
                        - np.abs(1.0 * speedX * np.sin(obs['angle'])) \
                        - 2 * speedX * np.abs(obs['trackPos'] * np.sin(obs['angle'])) \
                        - speedY * np.cos(obs['angle'])


        elif self.reward_type == 'extra_github_lidar':
            speedX = obs['speedX'] / 200
            speedY = obs['speedY'] / 200

            lidar_front = obs['track'][10]##lidar value on front of the car ( lidar 10th)
            lidar_penalty_on = 1 if np.abs(lidar_front)<=50 else 0
            lidar_rate = np.abs(lidar_front)/50
            lidar_penalty_ratio = np.power(lidar_rate,lidar_penalty_on)

            reward = speedX * np.cos(1.0 * obs['angle']) \
                        - np.abs(1.0 * speedX * np.sin(obs['angle'])) \
                        - 2 * speedX * np.abs(obs['trackPos'] * np.sin(obs['angle'])) \
                        - speedY * np.cos(obs['angle'])

            reward = lidar_penalty_ratio * reward
        elif self.reward_type == 'last_resort':
            Vx = obs['speedX'] / 200
            Vy = obs['speedY'] / 200
            trackpos = np.abs(obs['trackPos'])
            sintheta = np.abs(np.sin(obs['angle']))
            costheta = np.cos(obs['angle'])
            reward = Vx * costheta - Vx * sintheta - Vy * costheta
        elif self.reward_type == 'sigmoid':
            Vx = obs['speedX'] / 200
            Vy = obs['speedY'] / 200
            clipped_cos = sigmoid(np.cos(obs['angle']) * 3)
            inverse_clipped_cos = (1 - clipped_cos) / 2
            reward = Vx * clipped_cos - Vx * inverse_clipped_cos - Vy * clipped_cos
        elif self.reward_type == 'sigmoid_v2':
            Vx = obs['speedX'] / 200
            Vy = obs['speedY'] / 200
            clipped_cos = sigmoid(np.cos(obs['angle']) * 2)
            inverse_clipped_cos = (1 - clipped_cos) / 2
            reward = Vx * clipped_cos - Vx * inverse_clipped_cos - Vy * clipped_cos
        elif self.reward_type == 'sigmoid_v3':
            Vx = obs['speedX'] / 200
            Vy = obs['speedY'] / 200
            clipped_cos = sigmoid(np.cos(obs['angle']) * 0.5)
            inverse_clipped_cos = (1 - clipped_cos) / 2
            reward = Vx * clipped_cos - Vx * inverse_clipped_cos - Vy * clipped_cos
        elif self.reward_type == 'paper':
            Vx = obs['speedX'] / 200
            costheta = np.cos(obs['angle'])
            d = np.abs(obs['trackPos'])
            reward = Vx * (costheta - d)
        elif self.reward_type == 'race_pos':
            reward = progress - np.abs(sp * np.sin(obs["angle"]))  # no trackpos
            if obs['racePos'] > obs_pre['racePos']:
                reward += 1
            elif obs['racePos'] < obs_pre['racePos']:
                reward -= 1

        # collision detection
        if obs['damage'] - obs_pre['damage'] > 0:
            info["collision"] = True
            reward = -1

        if self.terminal_judge_start < self.time_step: # Episode terminates if the progress of agent is small
            if abs(progress) < self.termination_limit_progress:
                    reward -= 10
                    # print("--- No progress restart : reward: {},x:{},angle:{},trackPos:{}".format(reward,sp,obs['angle'],obs['trackPos']))
                    # print(self.time_step)
                    episode_terminate = True
                    info["no progress"] = True
                    # client.R.d['meta'] = True

        if np.cos(obs['angle']) < 0:  # Episode is terminated if the agent runs backward
            if self.time_step > 20:
                reward -= 1
                # print("--- backward restart : reward: {},x:{},angle:{},trackPos:{}".format( reward, sp, obs['angle'], obs['trackPos']))
                # print(self.time_step)
                # episode_terminate = True
                info["moving back"] = True
                # client.R.d['meta'] = True

            self.backward_counter += 1
            if self.backward_counter >= 250:
                episode_terminate = True
        else:
            self.backward_counter = 0

        info["place"] = int(obs["racePos"])
        if episode_terminate is True: # Send a reset signal
            # reward += (8 - obs["racePos"]) * 20 # If terminated and first place
            self.initial_run = False
            # client.respond_to_server()

        self.time_step += 1

        return self.get_obs(), reward, episode_terminate, info

    def reset(self, relaunch=False, sampletrack=False, render=False):
        """ Reset the environment
            Arguments:
                - relaunch: Relaunch the game. Necessary to call with
                    from time to time because of the memory leak
                sampletrack: Sample a random track and load the game
                    with it at the relaunch. Relaunch needs to be 
                    true in order to modify the track!
                render: Change the mode. If true, game will be launch
                    in "render" mode else with "results only" mode.
                    Relaunch needs to be true in order to modify the track!
        """
        self.time_step = 0
        self.backward_counter = 0

        if relaunch:
            if sampletrack:
                try:
                    self.track_name, _ = sample_track(self.root, self.reset_counter, self.track)
                    self.reset_counter += 1
                except AttributeError:
                    pass
            try:
                set_render_mode(self.root, render=render)
            except AttributeError:
                    pass
            self.tree.write(self.path)
            time.sleep(0.5)

        if not self.client_mode:
            if self.initial_reset is not True:
                self.client.R.d['meta'] = True
                self.client.respond_to_server()

                ## TENTATIVE. Restarting TORCS every episode suffers the memory leak bug!
                if relaunch is True:
                    self.reset_torcs()
                    # print("### TORCS is RELAUNCHED ###")

        # Modify here if you use multiple tracks in the environment
        self.client = snakeoil3.Client(p=self.port, vision=False, client_mode=self.client_mode)  # Open new UDP in vtorcs
        self.client.MAX_STEPS = np.inf

        client = self.client
        client.get_servers_input()  # Get the initial input from torcs
        self.client.MAX_STEPS = np.inf

        obs = client.S.d  # Get the current full-observation from torcs
        self.observation = self.make_observaton(obs)

        self.initial_reset = False
        return self.get_obs()

    def kill(self):
        if not self.client_mode:
            os.system('pkill torcs')

    def close(self):
        self.client.R.d['meta'] = True
        self.client.respond_to_server()

    def get_obs(self):
        return self.observation

    def reset_torcs(self, port=3101):
        if not self.client_mode:
            os.system('pkill torcs')
            time.sleep(0.5)
            os.system('torcs -nofuel -nodamage -nolaptime -p 3101 &')
            time.sleep(0.5)
            os.system('sh autostart.sh')
            time.sleep(0.5)

    def agent_to_torcs(self, u):
        torcs_action = {'steer': u[0]}
        torcs_action.update({'accel': u[1]})
        torcs_action.update({'brake': (u[2] + 1)/2})
        return torcs_action

    def make_observaton(self, raw_obs):
        return np.concatenate(
            [np.array(raw_obs["angle"], dtype=np.float32).reshape(1)/np.pi*2,
            np.array(raw_obs["track"], dtype=np.float32)/100,
            np.array(raw_obs["trackPos"], dtype=np.float32).reshape(1)/2,
            np.array(raw_obs["speedX"], dtype=np.float32).reshape(1)/200,
            np.array(raw_obs["speedZ"], dtype=np.float32).reshape(1)/200,
            np.array(raw_obs["speedY"], dtype=np.float32).reshape(1)/200,
            np.array(raw_obs["wheelSpinVel"], dtype=np.float32)/200,
            np.array(raw_obs["rpm"], dtype=np.float32).reshape(1)/5000]
        )

    def __del__(self):
        self.kill()

