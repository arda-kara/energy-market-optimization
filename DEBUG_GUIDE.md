# Debugging Guide for Energy Market Optimization

This guide provides instructions on how to debug tensor shape mismatches and other issues in the Energy Market Optimization project.

## Common Issues

1. **Tensor Shape Mismatches**: The most common issue is tensor shape mismatches, especially in the SAC agent. This can happen when:
   - The state or action dimensions don't match the expected dimensions
   - The rewards or next_states tensors have the wrong shape
   - The Critic's forward method returns a tuple, but the code expects a single tensor

2. **NaN Values**: NaN values can occur during training, especially if the learning rate is too high or the clipping parameters are not set correctly.

3. **CUDA Out of Memory**: This can happen if the batch size or model size is too large for your GPU.

## Debug Scripts

The project includes several debug scripts to help diagnose and fix issues:

### 1. debug_tensor_shapes.py

This script provides utilities to track tensor shapes throughout the code. It can be used to diagnose tensor shape mismatches.

```bash
# Debug the SAC agent with default settings
python debug_tensor_shapes.py

# Debug the PPO agent
python debug_tensor_shapes.py --agent ppo

# Debug with a specific number of training steps
python debug_tensor_shapes.py --steps 100

# Debug with a specific hidden dimension
python debug_tensor_shapes.py --hidden-dim 128

# Debug with a specific episode length
python debug_tensor_shapes.py --episode-length 48
```

### 2. test_sac_agent.py

This script tests the basic functionality of the SAC agent, including initialization, action selection, and network updates.

```bash
# Run the test script
python test_sac_agent.py
```

### 3. test_sac_training.py

This script tests the training process of the SAC agent, including buffer sampling, network updates, and evaluation.

```bash
# Run the test script
python test_sac_training.py
```

## Debugging Tips

1. **Check Tensor Shapes**: Always check the shapes of tensors before and after operations. Use the debug scripts to track tensor shapes throughout the code.

2. **Use Debug Mode**: Enable debug mode in the agents to get more detailed logging:

```python
agent = SACAgent(
    env=env,
    debug_mode=True
)
```

3. **Check Network Architectures**: Make sure the network architectures match the expected input and output dimensions:

```python
# Log network architectures
for name, param in agent.actor.named_parameters():
    print(f"{name}: {param.shape}")
```

4. **Check Buffer Sampling**: Make sure the buffer is sampling correctly and returning tensors with the right shapes:

```python
batch = buffer.sample(64)
states, actions, rewards, next_states, dones = batch
print(f"states: {np.shape(states)}")
print(f"actions: {np.shape(actions)}")
print(f"rewards: {np.shape(rewards)}")
print(f"next_states: {np.shape(next_states)}")
print(f"dones: {np.shape(dones)}")
```

5. **Check Environment**: Make sure the environment is returning states and actions with the right shapes:

```python
state, _ = env.reset()
print(f"state: {np.shape(state)}")
action = env.action_space.sample()
print(f"action: {np.shape(action)}")
```

## Fixing Common Issues

### Tensor Shape Mismatches

1. **State Dimension Mismatch**: Make sure the state dimension in the agent matches the state dimension in the environment:

```python
state_dim = env.observation_space.shape[0]
agent = SACAgent(env=env)  # This will automatically set state_dim
```

2. **Action Dimension Mismatch**: Make sure the action dimension in the agent matches the action dimension in the environment:

```python
action_dim = env.action_space.shape[0]
agent = SACAgent(env=env)  # This will automatically set action_dim
```

3. **Rewards Shape Mismatch**: Make sure the rewards tensor has the right shape:

```python
# In ReplayBuffer.sample
if len(rewards.shape) > 1 and rewards.shape[1] > 1:
    rewards = rewards[:, 0]
```

4. **Next States Shape Mismatch**: Make sure the next_states tensor has the right shape:

```python
# In ReplayBuffer.sample
if len(next_states.shape) == 1:
    next_states = np.reshape(next_states, states.shape)
```

### NaN Values

1. **Reduce Learning Rate**: If you're getting NaN values during training, try reducing the learning rate:

```python
agent = SACAgent(
    env=env,
    lr=1e-4  # Default is 3e-4
)
```

2. **Adjust Clipping Parameters**: If you're using PPO, try adjusting the clipping parameters:

```python
agent = PPOAgent(
    env=env,
    clip_ratio=0.1  # Default is 0.2
)
```

### CUDA Out of Memory

1. **Reduce Batch Size**: If you're getting CUDA out of memory errors, try reducing the batch size:

```python
metrics = agent.train(
    env=env,
    batch_size=32  # Default is 64 or 256
)
```

2. **Reduce Model Size**: If reducing the batch size doesn't help, try reducing the model size:

```python
agent = SACAgent(
    env=env,
    hidden_dim=64  # Default is 256
)
```

## Conclusion

Debugging tensor shape mismatches and other issues in deep reinforcement learning can be challenging. The debug scripts and tips provided in this guide should help you diagnose and fix common issues in the Energy Market Optimization project.

If you encounter any issues that are not covered in this guide, please open an issue on the project's GitHub repository. 