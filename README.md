# PESLA - BLG604E Deep Reinforcement Learning Term Project

This project is for the BLG604E Deep Reinforcement Learning term project at ITU, taught by Nazım Kemal Üre.

## TORCS

Competition took place within TORCS environment and the winner agent was our Pesla agent. There is an upcoming conference paper regarding architecture.

## Usage

Winner agent:
```
python test_torcs.py --on-render --test --algo sac-lstm --load-from releases/TORCS_SACLSTM_512256128_L1_EP5000_N1_G99.pt
```

Training: (please see command line arguments)

```
python run_torcs.py
```

## Alternatives

There are multiple agents under releases folder. But their architecture is not all the same. I would update here for each of them in time.

## Information

This project and repository is the joint work of [Kıvanç Güçkıran](https://github.com/kivancguckiran), [Can Erhan](https://github.com/ccerhan) and [Onur Karadeli](https://github.com/okaradeli).

The codebase is a clone of and heavily borrowed from [https://github.com/medipixel/rl_algorithms/](https://github.com/medipixel/rl_algorithms/)
