# PPO-Generalized-Advantage_estimation
Implementation of PPO-Generalized-Advantage_estimation

# Requirements
gym == 0.23.0

brax==0.0.12

jax == 0.4.14

jaxlib=0.3.14

pytorch-lightining == 1.6.0

torch == 2.0.1

# Collab installations
!apt-get install -y xvfb

!pip install gym==0.23.1 \
    pytorch-lightning==1.6 \
    pyvirtualdisplay

!pip install -U brax==0.0.12 jax==0.3.14 jaxlib==0.3.14+cuda11.cudnn82 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html


# Description
PPO-Generalized-Advantage_estimation is a continuation of PPO this architecture allows us to use the the generalized advantage estimation which will take into account states in the future and calculate the advantage function, this is done similar to n_step where we get the immediate reward of future states plus the discounted factor of the last state value function minus the current state value function. This implementation lets us have more accurate estimations of the rewards that can be obtain over a certain time step.

# Environment
robotic_halfcheetah

# Architecture
PPO with Generalized-Advantage_estimation

# optimizer
Policy: AdamW
Value: AdamW

# loss function
Policy: clipped surrogate objective function
Value: smooth L1 loss function
