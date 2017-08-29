Add [Chinese version](https://github.com/kuhung/DeepRLHacks/blob/master/README.md##深度强化学习hacks) by (kuhung)[https://github.com/kuhung] 



# DeepRLHacks  
From a talk given by [John Schulman](http://joschu.net/) titled "The Nuts and Bolts of Deep RL Research" (Aug 2017)   
These are tricks written down while attending summer [Deep RL Bootcamp at UC Berkeley](https://www.deepbootcamp.io/).   

## Tips to debug new algorithm   
1. Simplify the problem by using a low dimensional state space environment.      
    - John suggested to use the [Pendulum problem](https://gym.openai.com/envs/Pendulum-v0) because the problem has a 2-D state space (angle of pendulum and velocity).    
    - Easy to visualize what the value function looks like and what state the algorithm should be in and how they evolve over time.  
    - Easy to visually spot why something isn't working (aka, is the value function smooth enough and so on).

2. To test if your algorithm is reasonable, construct a problem you know it should work on.   
    - Ex: For hierarchical reinforcement learning you'd construct a problem with an OBVIOUS hierarchy it should learn. 
    - Can easily see if it's doing the right thing.   
    - WARNING: Don't over fit method to your toy problem (realize it's a toy problem).   

3. Familiarize yourself with certain environments you know well.
    - Over time, you'll learn how long the training should take.   
    - Know how rewards evolve, etc... 
    - Allows you to set a benchmark to see how well you're doing against your past trials.    
    - John uses the hopper robot where he knows how fast learning should take, and he can easily spot odd behaviors.    

## Tips to debug a new task   
1. Simplify the task
    - Start simple until you see signs of life.   
    - Approach 1: Simplify the feature space: 
      - For example, if you're learning from images (huge dimensional space), then maybe hand engineer features first. Example: If you think your function is trying to approximate a location of something, use the x,y location as features as step 1. 
      - Once it starts working, make the problem harder until you solve the full problem.   
   - Approach 2: simplify the reward function.
      - Formulate so it can give you FAST feedback to know whether you're doing the right thing or not.   
      - Ex: Have reward for robot when it hits the target (+1). Hard to learn because maybe too much happens in between starting and reward. Reformulate as distance to target instead which will increase learning and allow you to iterate faster.    

## Tips to frame a problem in RL   
Maybe it's unclear what the features are and what the reward should be, or if it's feasible at all.    

1. First step: Visualize a random policy acting on this problem.   
    - See where it takes you.    
    - If random policy on occasion does the right thing, then high chance RL will do the right thing.   
      - Policy gradient will find this behavior and make it more likely.  
    - If random policy never does the right thing, RL will likely also not.   

2. Make sure observations usable:
    - See if YOU could control the system by using the same observations you give the agent.   
      - Example: Look at preprocessed images yourself to make sure you don't remove necessary details or hinder the algorithm in a certain way.

3. Make sure everything is reasonably scaled.   
    - Rule of thumb: 
      - Observations: Make everything mean 0, standard deviation 1.
      - Reward: If you control it, then scale it to a reasonable value.
        - Do it across ALL your data so far.   
    - Look at all observations and rewards and make sure there aren't crazy outliers.    

4. Have good baseline whenever you see a new problem.   
    - It's unclear which algorithm will work, so have a set of baselines (from other methods)
      - Cross entropy method   
      - Policy gradient methods 
      - Some kind of Q-learning method (checkout [OpenAI Baselines](https://github.com/openai/baselines) as a starter or [RLLab](https://github.com/rll/rllab) 

## Reproducing papers    
Sometimes (often), it's hard to reproduce results from papers. Some tricks to do that:   

1. Use more samples than needed.    
2. Policy right... but not exactly
     - Try to make it work a little bit.   
     - Then tweak hyper parameters to get up to the public performance.   
     - If want to get it to work at ALL, use bigger batch sizes. 
       - If batch size is too small, noisy will overpower signal.  
       - Example: TRPO, John was using too tiny of a batch size and had to use 100k time steps. 
       - For DQN, best hyperparams: 10k time steps, 1mm frames in replay buffer.


## Guidelines on-going training process   
Sanity check that your training is going well.    

1. Look at sensitivity of EVERY hyper parameter
    - If algo is too sensitive, then NOT robust and should NOT be happy with it.   
    - Sometimes it happens that a method works one way because of funny dynamics but NOT in general.

2. Look for indicators that the optimization process is healthy.  
    - Varies 
    - Look at whether value function is accurate.
      - Is it predicting well?    
      - Is it predicting returns well?
      - How big are the updates?   
    - Standard diagnostics from deep networks   

3. Have a system for continuously benchmarking code.    
    - Needs DISCIPLINE.   
    - Look at performance across ALL previous problems you tried.   
      - Sometimes it'll start working on one problem but mess up performance in others.   
      - Easy to over fit  on a single problem.
    - Have a battery of benchmarks you run occasionally.   

4. Think your algorithm is working but you're actually seeing random noise.   
    - Example: Graph of 7 tasks with 3 algorithms and looks like 1 algorithm might be doing best on all problems, but turns out they're all the same algorithm with DIFFERENT random seeds.   

5. Try different random seeds!!
    - Run multiple times and average.   
    - Run multiple tasks on multiple seeds. 
      - If not, you're likely to over fit.   

6. Additional algorithm modifications might be unnecessary.      
    - Most tricks are ACTUALLY normalizing something in some way or improving your optimization.  
    - A lot of tricks also have the same effect... So you can remove some of them and SIMPLIFY your algorithm (VERY KEY).   

7. Simplify your algorithm   
    - Will generalize better

8. Automate your experiments   
    - Don't spend your whole day watching your code spit out numbers.   
    - Launch experiments on cloud services and analyze results.   
    - Frameworks to track experiments and results:
      - Mostly uses iPython notebooks.
      - DBs seem unnecessary to store results.   


## General training strategies
1. Whiten and standardize data (for ALL seen data since the beginning).   
    - Observations:
      - Do it by computing a running mean and standard deviation. Then z-transform everything.   
      - Over ALL data seen (not just the recent data).
        - At least it'll scale down over time how fast it's changing.
        - Might trip up the optimizer if you keep changing the objective. 
        - Rescaling (by using recent data) means your optimizer probably didn't know about that and performance will collapse.
  
    - Rewards:
      - Scale and DON'T shift. 
        - Affects agent's will to live.
        - Will change the problem (aka, how long you want it to survive).

    - Standardize targets:
      - Same way as rewards.
  
    - PCA Whitening?
      - Could help.
      - Starting to see if it actually helps with neural nets.
      - Huge scales (-1000, 1000) or (-0.001, 0.001) certainly makes learning slow.   

2. Parameters that inform discount factors.
    - Determines how far you're giving credit assignment.   
    - Ex: if factor is 0.99, then you're ignoring what happened 100 steps ago... Means you're shortsighted. 
      - Better to look at how that corresponds to real time 
        - Intuition, in RL we're usually discretizing time.  
        - aka: are those 100 steps 3 seconds of actual time? 
        - what happens during that time?
    - If TD methods for policy gradient of Value fx estimation, gamma can be close to 1 (like 0.999)
      - Algo becomes very stable.   

3. Look to see that problem can actually be solved in the discretized level.  
    - Example: In game if you're doing frame skip.
      - As a human, can you control it or is it impossible?
      - Look at what random exploration looks like 
        - Discretization determines how far your browning motion goes. 
        - If do many actions in a row, then tend to explore further.   
        - Choose your time discretization in a way that works.

4. Look at episode returns closely.   
    - Not just mean, look at min and max.
      - The max return is something your policy can hone in pretty well.
      - Is your policy ever doing the right thing??
    - Look at episode length (sometimes more informative than episode reward).
      - if on game you might be losing every time so you might never win, but... episode length can tell you if you're losing SLOWER.
      - Might see a episode length improvement in the beginning but maybe not reward.


## Policy gradient diagnostics   
1. Look at entropy really carefully   
    - Entropy in ACTION space
      - Care more about entropy in state space, but don't have good methods for calculating that.
    - If going down too fast, then policy becoming deterministic and will not explore.   
    - If NOT going down, then policy won't be good because it is really random.   
    - Can fix by:
      - KL penalty
        - Keep entropy from decreasing too quickly.    
      - Add entropy bonus.
    - How to measure entropy.   
      - For most policies can compute entropy analytically. 
        - If continuous usually using a Gaussian so can compute differential entropy.  
    
2. Look at KL divergence
    - Look at size of updates in terms of KL divergence.   
    - example:
      - If KL is .01 then very small.
      - If 10 then too much.
  
3. Baseline explained variance. 
    - See if value function is actually a good predictor or a reward.   
      - if negative it might be over fitting or noisy.
        - Likely need to tune hyper parameters

4. Initialize policy   
    - Very important (more so than in supervised learning).   
    - Zero or tiny final layer to maximize entropy
      - Maximize random exploration in the beginning   

## Q-Learning Strategies 
1. Be careful about replay buffer memory usage.  
    - You might need a huge buffer, so adapt code accordingly.   

2. Play with learning rate schedule.   

3. If converges slowly or has slow warm-up period in the beginning
    - Be patient... DQN converges VERY slowly.   


## Bonus from [Andrej Karpathy](http://cs.stanford.edu/people/karpathy/):   
1. A good feature can be to take the difference between two frames.   
   - This delta vector can highlight slight state changes otherwise difficult to distinguish.   
---
# 深度强化学习hacks 
[John Schulman](http://joschu.net/)“深度增强学习研究基础”的演讲。(Aug 2017)
以下是参加夏季[Deep RL Bootcamp at UC Berkeley](https://www.deepbootcamp.io/)时记录下的tricks。  

## 调试新算法的小技巧
1. 使用低维状态空间简化问题。
  - John建议使用类似[Pendulum problem](https://gym.openai.com/envs/Pendulum-v0)的问题。像这样的问题仅有角度和速度两个维度。
 - 首先，这样做方便将目标函数、算法的最终状态以及算法的迭代情况可视化出来。
 - 其次，当出现问题时，更容易将出问题的点直观的表达。（比如目标函数是否够平滑等问题）。
    
2. 构造一个应该起作用的问题来测试你的算法
 - 比如：对于一个分层强化学习算法，你应该构造一个算法可以直观学习到分层的问题。
 - 这样能够轻易地发现那里出了问题。
 - 注意：不要在这样的小问题上过分的尝试。
    
3. 放在一个你十分熟悉的环境
 - 随着时间的推移，你将能预估训练所需的时间。
 - 明白你的奖赏是如何变化的，以及类似问题..
 - 能够设定一个基线，以便让你知道相对过去改进了多少。
 - John使用他的hpper robot，因为他知道算法应该学多块，以及哪些行为是异常的。

## 调试新任务的小技巧
1. 简化问题
 - 从简单的开始，直到回到问题。
 - 途径1： 简化特征空间
     - 举例来说，如果你是想从图片（高维空间）中学习，那么你可能先需要处理特征。举个例子：如果你的算法是想标定某个事物的位置，一开始，使用单一的x，y坐标可能会更好。
     - 一旦起步，逐步还原问题直到解决问题。    
 - 途径2：简化奖赏函数
     - 简化奖赏函数，这样可以有一个更快的反馈，帮助你知道是不是走偏了。
     - 比如：击中时给robot记一分。这种情况很难学习，因为在开始于奖赏之前有太多的可能。将击中得分改为距离，这样将提升学习速率、更快迭代。

## 将一个问题转化为强化学习的小技巧
可能现实是并不清楚特征是什么，也不清楚奖赏该是什么。或者，问题简直就是一团糟。

1. 第一步：将这个问题使用随机策略可视化出来。
 - 看看那些部分吸引了你
 - 如果这个随机策略在某些情况下做了正确的事，那么很大概率，强化学习也可以做到。
     - 策略的更新将会发现这里面的行为，并促使其更像。

- 如果随机策略永远都做不到，那么强化学习也不可能。
    
2. 确保可观测
 - 确保你能够控制系统，且是通过给agent的、一样的观测环境。
     - 举个例子： 亲自查看处理过图片，以确保你没有移出掉关键信息或者是在某种程度上阻碍算法。
   
3. 确保所有的事物都在合理的尺度
    - 经验法则：
       - 观测环境： 确保均值为0，方差为1.
       - 奖赏： 如果你能控制它，就把他缩放到一个合理的维度。
       - 在所有的数据上都做同样的处理。
 - 检查所有的观测环境以及奖赏，以确保没有特别离奇的异常值。
    
4. 建立一个好的基线，无论在何时看到一个新问题。
 - 一开始并不清楚哪些算法会起作用，所以设定一系列的基线（从其他方法）
     - 交叉熵
     - 策略更新
     - 一些类型的 Q-learning算法: [OpenAI Baselines](https://github.com/openai/baselines)或者[RLLab](https://github.com/rll/rllab)

## 复现论文
某些时候（经常的事），复现论文结果特别困难。有一些技巧如下：
1. 使用比预计更多的样本。
2. 策略正确，但又不完全正确。
 - 尝试让模型运行更久一点。
 - 调整超参数以达到公开的效果。
 - 如果想让他在所有数据上都奏效，使用更大的batch sizes。
     - 如果batch size太小，噪声将会压过真正有用的信号。
     - 比如： TRPO，John使用了一个特别小的batch size，然后不得不训练10万次迭代。
     - 对于DQN，最好的参数：一万次迭代，10亿左右的缓存存储。

## 对于正在进行的训练的一些处理
合理地检查你的训练是否正常

1. 检查每个超参数的敏感性
 - 如果一个算法太过敏感，那么它可能不太鲁棒，并且不容乐观。
 - 有些时候，某个策略生效，可能仅仅是因为巧合而已，它并不足以推广。

2. 关注优化过程是否正常的一些指标
 - 变动情况
 - 关注目标函数是否正确
     - 是否预测正确？
     - 它的返回是否正确？
     - 每次的更新有多大？
 - 一些标准的深度神经网络的诊断方法

3. 有一套能够连续记录代码的系统
 - 有成型的规则
 - 注意过去所作的所有尝试
     - 有些时候，我们关注一个问题，最后却把表现安在了其他问题上。
     - 对于一个问题，很容易过拟合。
 - 有一大推之前时不时测试的基准。

4. 有时候会觉得系统运行正常，但那不过是一团随机噪声。
 - 例如： 有3套算法处理7个任务，可能会有一个算法看起来能很好地处理所有问题，但实际上，它们不过是一套算法，不同的仅仅是随机种子而已。

5. 使用不同的随机种子！
 - 运行多次取平均。
 - 在多个种子的基础上运行多次。
     - 如果不这样做，那你很有可能是过拟合了。

6. 额外的算法修改可能并不重要。
 - 大部分的技巧都是在正则化处理某些对象，或者是在改进你的优化算法。
 - 许多技巧有相同的效果......所以，你可以通过移除它们，来简化你的算法（很关键）。

7. 简化你的算法
 - 将会取得更好的泛化效果。
    
8. 自动化你的实验
 - 不要整天盯着你的代码吞吐数据。
 - 将实验部署在云端并分析结果。
 - 追踪实验与结果的框架：
     - 大部分使用 iPython notebooks。
     - 数据库对于存储结果来说不是很重要。
    
## 总的训练策略
1. 数据的白化与标准化（一开始就这样做，对所有数据）
- Obervation：
     - 计算均值与标准差。然后做标准化转变。
     - 对于全体数据（不仅仅是当前的数据）。
         - 至少它减轻了随时间的变化波动情况。
          - 如果不断地改变对象的话，可能会使优化器迷糊。
         - 缩放（仅最近的数据）意味着你的优化器不知道这个情况，训练将会失败。
            
 - Rewards：
     - 缩放但不要转变对象。
         - 影响agent的存活意愿。
         - 将会改变问题（你想让它存活多久）。
            
- Standardize targets
     - 与rewards相同。
        
 - PCA 白化？
     - 可以起作用。
     - 开始时的时候要看看它是否真的对神经网络起作用。
     - 大规模的缩放（-1000，1000）或者（-0.001，0.001）必然会是学习减缓。

2. 影响折扣因子的参数
 - 判断分配多少信用
 - 举个例子：如果因子是0.99，那么你将忽略100步以前的事情...意味着你是短视的。
     - 最好是看看对应多少的现实时间。
         - 直觉上的，我们通常在强化学习中离散时间。
         - 换句话说，100步是指3秒前吗？
         - 在这个过程中发生了什么？
 - 如果是用于fx估计的策略更新TD算法，gamma可以选择靠近1（比如0.999）
     - 算法将非常稳健。
        
    
3. 观察问题是否真的能被离散化处理
 - 例如：在一个游戏你做跳帧运动。
     - 作为一个人类，你能控制还是不能控制？
     - 查看随机情况是怎么样的
         - 离散化程度决定了你的随机布朗运动能有多远。
         - 如果连续性地运动，往往模型会走很远。
         - 选择一个起作用的时间离散化分。

4. 密切关注返回的episode
 - 不仅仅是均值，还包括极大极小值。
     - 最大返回是你的策略能做到的最好程度。
     - 看看你的策略是否工作正常？
- 查看episode长度（有时比episode reward更为重要）。
     - 在一场游戏中你场场都输，从未赢过，但是episode长度可以告诉你是不是输得越来越慢。
     - 一开始可能看见episode 长度有改进，但reward无反应。

## 策略优化诊断
1. 仔仔细细地观察entropy
 - action层面的entropy
     - 留意状态空间的entropy变化，虽然没有好的计量办法。
 - 如果快速地下降，策略很快便会固定然后失效。
 - 如果没有下降，那么这个策略可能不是很好，因为它是随机的。
 - 通过以下方式补救：
     - KL penalty
           - 使entropy远离快速下降。
     - 增加 entropy bonus。
 - 如何测量entropy
     - 对于大部分策略，可以采用分析式方法。
         - 对于连续变量，通常使用高斯分布，这样可以通过微分计算entropy。

2. 观察KL差异
 - 观察KL差异的更新尺度。
 - 例如：
      - 如果KL是0.01，那它是相当小的。
       - 如果是10，那它又非常的大。

3. 通过基线解释variance
 - 查看value function是不是好的预测或者回报。
     - 如果是负数，可能是过拟合了或者是噪声。
         - 可能需要超参数调节。
             
4. 初始化策略
 - 非常重要（比监督学习更重要）。
 - 最后一层取0或者是一个很小的值来最大化entropy。
     - 初始最大化随机exploration。

## Q-Learning 策略
1. 留意replay buffer存储的使用情况。
- 你可能需要一个非常大的biffer，依代码而定。

2. 手边常备learing rate表。

3. 如果收敛很慢或者一开始有一个很慢的启动
- 保持耐心...DQNs收敛非常慢。
 

## 来自[Andrej Karpathy](http://cs.stanford.edu/people/karpathy/)的福利：
1. 一个好的特征能够被俩个框剪捕捉到不同特征。
- delta向量可以稍稍更改一下，不然难以区分。
## Translated by [kuhung](https://github.com/kuhung) 2017/08/29
- 网上有一个ppt资源[The Nuts and Bolts of Deep RL Research
](http://joschu.net/docs/nuts-and-bolts.pdf)。

