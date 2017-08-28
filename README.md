# DeepRLBootcampRLTricks
These are tricks that I wrote down while attending summer Deep RL Bootcamp at UC Berkeley.   
These were proposed by [John Schulman](http://joschu.net/) on Day 1.

## Tips to debug new algorithm   
- Simplify the problem in a game with a low dimentional state space.      
  - John suggested to use the [Pendulum problem](https://gym.openai.com/envs/Pendulum-v0) because the problem has a 2-D state space (angle of pendulum and velocity).    
  - Easy to visualize what the value function looks like and what state the algorithm should be in and how they evolve over time.  
  - Easy to visually spot why something isn't working (aka, is the value function smooth enough and so on).

- To test if your algorithm is reasonable, construct a problem you know it should work on.   
  - Ex: For hierarchical reinforcement learning you'd construct a problem with an OBVIOUS huerarchy it should learn. 
  - Can easily see if it's doing the right thing.   
  - WARNING: Don't overfit method to your toy problem (realize it's a toy problem).   

- Useful
