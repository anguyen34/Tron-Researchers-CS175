---
layout:	default
title:	Final Report
---

## Video:

## Project Summary:
For our project, we conducted research of various parameters across three different machine learning algorithms and Monte Carlo search, all of which were used as functional approximators for reinforcement Q-learning. We studied how changing the values of the selected parameters would impact the success of each policy in simulated games against other policies with the same algorithm, but using parameters we defined as the baseline parameters. The platform used for conducting our research was the Tron environment from ColosseumRL, a multiagent reinforcement learning environment comprising several different games. The game we chose, Tron as implemented in ColosseumRL is the Tron light cycles subgame, which is a variation of the snake game concept, but here the goal is to force opponents into walls or trails, while simultaneously avoiding them. However, unlike the original Tron light cycles subgame which was a player versus three AI, the Tron environment we used was four AI playing against each other. 

Our primary goal was to study the effect of changing various parameters of our reinforcement Q-learning functional approximators on our learning rates and overall success in the game for each policy. The three machine learning algorithms used as functional approximators for the Q-learning Greedy Epsilon policy are Random Forest, Neural Network, Monte Carlo Search, and Ensemble which contains a MultiLayer Perceptron Network, Adaboost Regressor, Support Vector Machine, and K-Nearest Neighbor Regressor. To satisfy our primary goal we researched the performance of each functional approximator while changing parameters such as number of hidden layers and nodes per layer for the Neural Network or maximum number of leaves and maximum depth for the Random Forest. Evaluating the performance of a model for each parameter configuration gives insight into the most impactful parameters on the model’s gameplay which is exactly what we are researching.

Our secondary goal was to identify which parameters of high impact were the most beneficial to our AI and to have our four functional approximators with their optimal parameters face off each other. Beyond just observing the impacts of changing certain parameters in a particular functional approximator, we also wanted to observe how the functional approximators compared with each other. We expected our neural network approximator to perform the best, since while researching for functional approximators to use for our project, we learned that neural networks are the most commonly used approximators for reinforcement learning, compared to an approximator like Random Forest. By comparing our neural network approximator with our other ones, we wanted to see if this conclusion also applied to AI that were training on and playing Tron.

What motivated our idea behind the project was the idea that games are a traditional field that AI are used in, such as Tron originally using AI as opponents. Developing better AI could result in more challenging and interesting AI opponents for Tron and other games. Machine learning is a core part of AI, since it can help produce solutions to problems where the solutions are hard to describe. An instance of this is Tron, where the solution to the game is winning against four other players. A solution must produce a series of actions for the player in which they are the winner, but also account for the actions of other players, and the numerous potential board states. Determining machine learning parameters that could result in a faster learning AI or a more successful AI are beneficial to actually producing better AI within a quicker time frame. Through researching the various machine learning parameters across several algorithms, we hoped to find some of the parameters that could lead to faster learning or more successful AI.


## Approaches:
Our approach for our research is to change values for certain parameters we are interested in for each of our four agents in order to examine the impact these parameters have on the performance of our agents in Tron. Our four agents are Greedy Epsilon Reinforcement Q Learning policies that use the following function approximators: Neural Network, Ensemble, Random Forest, and Monte Carlo Search. For each agent we chose a set of parameters that we hypothesized to have a substantial impact on the agent’s performance as well as values for each parameter that were interesting in researching. We then measured the performance of the agents with these different parameter values playing against agents of the same type but with baseline parameter values which allows us to study the impact of these different parameters. In general, our approximators are attempting to predict the cumulative reward a move could result in, which is then being used as a Q-value.

To implement our Neural Network agent, we utilized the Deep Q Network algorithm from the Rays Rlib library. This agent begins with initializing the Tron environment and Ray’s Rlib by defining the general config as well as the policy/Neural Network structure. We use 4 Neural Networks (1 for each player) where the 0th player is the agent with changing parameters, while agents 1-3 have baseline parameters and all agents utilize a Proximal Policy Optimization algorithm which is a type of actor critic optimization that attempts to minimize the loss according to this function: 

![](images/loss.png?raw=true) 

We then run the test function which will run the game of Tron with all 4 players continuously for the specified number of epochs which we chose to be 100 for this agent (meaning 100 games) because this agent takes a large amount of time to run. At the beginning of a game we train the agent using the Deep Q Network which is trained on the position of each player’s head and the game board observation that is preprocessed as follows: the board is rotated such that the current player appears as if they are player 0, there is a border on the board, and every other player appears as the same value rather than their player number. While the ith game is running, Ray’s Rlib computes an action to take for each player depending on the current state, the previous action, and previous reward. This action is then passed to the step function where a greedy epsilon policy is used in order to choose and use a final action on the state for that player as well as calculate the cumulative reward for that player for the purposes of data collection. This function returns the new state, reward, and a boolean representing if the game has terminated. This happens for the duration of the game then the whole algorithm repeats for the next game and so forth for all i games in the number of epochs.
 
For the data collection of the Neural Network agent, we chose to change the number of hidden layers, number of nodes per layer, activation function, and the value of epsilon for our greedy epsilon policy. For hidden layers we chose to test values of 1, 2, 5, 10, 20 with the default/baseline value used for the dummy agents being 1; we didn’t test values higher than 20 for performance/time reasons. Before testing we hypothesized that lower values for this parameter would result in underfitting while higher values might result in overfitting as well as taking more time to train. For the  number of nodes per layer we tested values of 1, 2, 5, 10, 25, 50, 64 with the baseline value being 64. Before testing we believed that, similar to the number of hidden layers, less nodes would underfit the data while taking shorter time to train while larger values might overfit the data and take longer to train. For the activation function, we tested ReLU, Swish, and TanH with ReLU being the baseline value. We believed that ReLU would be the best value for this as we found that it is typically more commonly used than the others, otherwise there aren’t many advantages/disadvantages to these values as this parameter is more of something that you test to see which is best rather than having preset best values. For epsilon, we chose values of 0.01, 0.05, 0.1, 0.25, 0.5 with the baseline being 0.01. Increasing this value increases the volatility of the data as the randomness is increased, however this may be beneficial by letting the regressor learn new patterns it would not have discovered on its own.

The Ensemble agent was implemented using SKLearn’s Voting Regressor, for which we decided to include an AdaBoost regressor, a K-Nearest Neighbors regressor, a Support Vector Machine regressor, and a MultiLayer Perceptron regressor. Each of the algorithms used as part of the Voting Regressor Ensemble are also from SKLearn. Like with the Neural Network, the Ensemble agent with modified parameters was player 0 and the other three Ensemble agents occupied players 1, 2, and 3. Also like with the Neural Network, each epoch corresponded to one game of Tron which ran until terminal, afterwards the cumulative reward of player 0 and the difference in reward between player 0 and the averaged reward of the other three players was collected for that epoch. In each step of an epoch, each of the living players would have a move chosen for them by their Voting Regressor, the actions of the players would be used to generate the next game state, and then for each player their move, reward, and the board state would be recorded as data. At the end of a game each player would receive a new Voting Regressor to replace their old one, with the new one trained using all of their accumulated data thus far. The data that is used to train each player’s Voting Regressor uses the move taken, the preprocessed board state like in the Neural Network, the current position, deaths, winners, and the positions of other players, where these are all features to predict the cumulative reward. 
 
The parameters chosen to be modified for the Ensemble agent are some of parameters from each of its algorithms, with each parameter being tested in isolation from one another. From the AdaBoost regressor the number of estimators and the loss function were tested. For the Support Vector Machine regressor we chose to alter the kernel function. From the MultiLayer Perceptron regressor we tested changes to the activation function and the number of nodes per hidden layer. Finally, we also tested changes to the value of epsilon. Each parameter value was tested for 100 epochs, like the Neural Network, since the Ensemble took longer to run relative to the Random Forest and Monte Carlo search due to the Multilayer Perceptron. For AdaBoost’s number of estimators we tested the values 10, 25, 50, 100, 150, and 200, with the baseline being 50. We hypothesized that adjusting the number of estimators could alter the success of the model or change whether the model overfits or underfits the collected data. The loss functions tested for AdaBoost were linear, square, and exponential loss functions, with linear being the baseline. The kernel functions we tested for the Support Vector Machine were the linear, polynomial, RBF, and exponential, with RBF being the baseline value. For the Multilayer Perceptron we tested the identity, logistic, ReLU, and TanH activation functions, where ReLU was the baseline value. In changing the loss, kernel, and activation functions we hypothesized that the respective baseline values of linear, RBF, and ReLU would perform the best. The number of layers tested for the Multilayer Perceptron was 1, 2, 5, 10, 25, 50, 100, and 200, with the baseline being 1 layer and each layer having 100 nodes. We hypothesized that increasing the number of layers would be advantageous in that it could produce a better model, but could also be disadvantageous by overfitting the data. We further hypothesized that decreasing the number of layers could reduce overfitting if it was occurring, but at the potential of then underfitting the data. Like with the other algorithms, the values of epsilon were tested with 0.01, 0.05, 0.10, 0.25, and 0.50, with 0.01 as the baseline. We hypothesized that an advantage of increasing the epsilon value could be that there is a higher likelihood of the agent learning a new beneficial move, but a disadvantage could be that the agent is more likely to take an action that could lead to its death. However, we also did not expect any individual parameter to have a drastic effect on the success of the overall algorithm, as each parameter only affects one algorithm amongst many in the Voting Regressor Ensemble. Instead, we looked to see if any one parameter may alter the learning rate or have any effects at all on the overall Voting Regressor.

Our Random Forest agent was implemented by using SKLearn’s library for a Random Forest Regressor. The implementation of this agent is the exact same as the Ensemble Agent with the only differences being that this agent utilizes a Random Forest Regressor rather than a Voting Regressor Ensemble and uses different parameters for training/data collection.
The parameters we decided to change for our Random Forest were the number of estimators used, the max depth, the max leaf nodes and the epsilon value used for greedy exploration. These parameter changes were done in isolation. Each parameter value was tested for 200 epochs, since the Random Forest ran relatively quickly, so we could afford to test for more epochs. We tested the number of estimators of 10, 25, 50, 100, 150, and 200 and the baseline value was 100. We hypothesized that 200 trees could possibly lead to more accurate predictions. The values we tested for max depth were 2, 5, 10, 25, 50, 100, 200 and None which was the baseline value. Before testing, we believe larger depths would result in more precise predictions, but too high of a depth would result in overfitting. The values we tested for max leaf nodes were 3, 10, 25, 50, 100, 200, and None which was the baseline value. We assumed that having no max number of leaf nodes would result in better predictions. We tested values of 0.01, 0.05, 0.1, 0.25, and 0.5 for our epsilon, with 0.01 as the baseline. We hypothesized that an epsilon value of 0.05 would result in a good balance of random actions for exploration and optimal actions for learning. 

Our Monte Carlo Search Tree agent didn’t use Ray Rlib and SKLearn like our previous agents, instead its search algorithm was made from scratch, albeit modified due to the nature of Tron. However it did follow a similar pattern of testing parameters where one agent would have one isolated parameter change, and the three other agents would have default parameters. It should be noted that while these three agents have the same parameters they are acting independently and not together against the one agent with the parameter change. Because of how consistent Monte Carlo Search is, we tested with 20 epochs. Each epoch is a full Tron game and the board state would take a step forward until the game terminated. With each step forward, agents would either choose a random action based on epsilon or the action that would result in the highest score. These scores were calculated by searching through all possible states upto a certain depth and returning a score of -50 if it died, 50 if it won, or its length on the board state multiplied by the reward it received. Due to Tron having four players and being a game where all players can move simultaneously, instead of being a turn based game, the traditional Monte Carlo search that we knew would not work. Instead, each agent greedily chooses a move based on the highest returned score from a score function we defined. The algorithm still performs a recursive search up to a specified depth, but always chooses the max score instead of alternating between choosing the min and max as it would in a turn based game.
  
The parameters we decided to change for our Monte Carlo Search Tree are the search depth and epsilon value for greedy exploration. The values we chose for search depth were 1, 2, 3, 4, and 5 with 2 being the default. A disadvantage of a bigger depth is its long run time however its advantage is getting a better grasp of good moves to do. The epsilon values chosen as well as our hypothesis of it are the same as our previous functional approximators.
  
In order to have the four agents play against each other, the code used was based off of the neural network approximation code. Like the original neural network code, the neural network agent is the player indexed at position 0. The choosing of moves for the neural network, training, and running the game is also the same as it was in original neural network code. In contrast to the original neural network code the other three players do not utilize neural networks to choose their moves. Instead, the other three agents exist as objects inside the neural network, and their moves are chosen using the method for choosing moves in the other three algorithms. In the code, the Monte Carlo agent is mapped to position 1, the Random Forest agent to position 2, and the Ensemble agent to position 3. To allow for the three algorithms to choose their moves, modified versions of their classes were reimplemented with methods for running the game removed, leaving only code related to choosing moves based on their respective criterion. In the case of the Ensemble and Random Forest agents, their training is also retained and the data required for training is passed to them from the Neural Network. Furthermore, the training is still done at the end of each game. For choosing moves for the three other agents, the game state is passed to them like it was in their original implementations. Finally, all four agents retain the ability for their parameters to be changed. The reasoning behind this approach of using the Neural Network code as a base is that it is the most complicated of the four agents in terms of code. In addition, it was easy to change the code for choosing moves in the Neural Network agent to use the move choosing methods of the other three agents. Another factor was that the Neural Network was not implemented using the same base code as the other three, so the process of integrating the other three agents was generally the same for all three, whereas integrating the Neural Network into a different algorithm could have been a more difficult and unique task. Furthermore, we collected data on the cumulative reward of each agent over the 100 epochs, so we could see how each agent would perform relative to each other as they learned. We also collected data on the winner of each game, so we could see if any of the agents would start to experience a distinct increase in their win rate.
  
Data collection for testing parameters followed two distinct approaches, one is how we chose to test parameters, and the other is the two types of data we collected. For testing parameters, we chose distinct parameter values and only tested one parameter at a time, meaning when testing a parameter for an agent the other parameters would remain at their baseline values. A key advantage of this approach is it allows for us to see the specific effects that one parameter may have on the model as its value is adjusted. However, we lose out on the potential for finding the optimal combination of parameters for the four agents, as the optimal parameters for an agent could have been composed of parameter values that were not the baseline values. Our primary reasoning for not testing different combinations of parameter values was that certain models already took a significant amount of time to run through all of the individual parameter values, leaving us without sufficient time to test combinations of parameter values. A second disadvantage to our approach is we are only able to get an approximation of the optimal value for a parameter since we’re only testing a set of values and not a complete range. Again, the reason for this decision is the extreme runtime necessary to run through a range of values and that it isn’t needed to show that a parameter has an effect on the functional approximator.
 
The two types of data values we collected at the end of each epoch were the normalized cumulative reward of the modified agent and the difference in rewards of the modified agent and the averaged rewards of the three opponent agents. In normalizing the cumulative reward, we would remove the reward for winning a game if the observed agent had won, with the goal being that we wanted to have more consistent data from non-winning games. We chose to record data on the normalized cumulative reward since the ColosseumRL Tron game rewarded an agent with a reward of 1 for just surviving, so a higher cumulative reward indicated the agent was surviving longer by learning. In addition, when possible we trained the agents to maximize the cumulative received reward from ColossuemRL, so it made sense to record the data for the values the agents were trying to predict. The second type of data collected was the difference in reward between the observed agent’s final cumulative reward and the averaged final cumulative reward of the other agents, which we refer to as the delta reward. An issue for our recorded delta reward was that games where the observed agent won were normalized by removing the bonus reward for winning, but this normalization was not applied to games where other agents won. As a result, the negative delta rewards are greater in magnitude than expected, but our analysis accounts for this error. Our goal in measuring delta reward was to see how the modified agent performed relative to the unmodified agents. A positive delta reward indicated that the modified agent was outperforming the unmodified agents. We hoped to see a consistently positive or negative delta reward, as it would clearly indicate whether a parameter had an effect on the model. Another potential metric we could have measured was the win rates of modified agents over the epochs, but it was a metric that was disconnected from the reward values and could already be encompassed by the delta reward.

Some implementation details that are mostly the same across all four agents is the general layout of each agent. Each agent environment is composed of a class with a test and step function to control the games being run over a defined amount of epochs. Each agent environment accounts for the agent with the modified parameters and the three agents that are unmodified. When an agent environment object is instantiated the parameters for the agent being modified are also passed in and stored. Other variables such as the Regressors for each agent, the cumulative rewards for each agent, alive players, and game state are stored as class attributes.

Pseudocode example of an agent class' test method to run through the epochs:

    def test(num_epochs):

        for i in range(num_epochs):

            reset game state

            train agents

            while game not done:

                step()

            collect cumulative reward data from class attributes of cumulative reward

            collect calculated delta reward from class attributes of cumulative reward

            plot graph of cumulative reward over epochs

            plot graph of delta reward over epochs
        
Pseudocode example of an agent class' step method for a single move in game:

    def step():

         actions = []

         for p in players:

             actions.append(choose_move(p))

         state, players, rewards, terminal, winners = environment.next_state(state, players, actions)

         for p in players:

             cumulative_rewards[p] += rewards[p]

         Save data on move taken, resulting board state, etc if Ensemble or Random Forest
        
## Evaluation:
For evaluating our models we primarily focused our evaluation and data collection on the quantitative aspect, as our goal was to answer which parameters we found to be the most impactful, where we measured the impact using changes in cumulative and delta reward over the course of the testing epochs. Our quantitative data and analysis consisted of collecting data on the cumulative rewards the agents received and plotting graphs for the collected data. Our qualitative evaluation consisted of observing the agents as they played and learned.

What we evaluated for each of our four functional approximators varied between each approximator, as well as how long we evaluated them for. For the Neural Network agent we ran the agents for 100 epochs/games for each parameter value tested, so all the designated parameter values could be completed within a reasonable time, as 100 epochs took about 45 minutes to run through. The Neural Network parameters we tested were the number of hidden layers, number of hidden nodes, the activation function, and the value of epsilon. For the Ensemble voting regressor the agents were run for 100 epochs/games for each parameter value tested, so we could test all of the parameter values within a reasonable time, as each run took about 30 minutes. Since the Ensemble agent was composed of many other algorithms, we tested more parameters than usual, specifically we tested AdaBoost’s loss function, AdaBoost’s number of estimators, the Support Vector Machine’s kernel function, the Multilayer Perceptron’s activation function, the Multilayer Perceptron’s number of hidden layers, and the value of epsilon. The Random Forest regressor was run for 200 epochs/games, since it took significantly less time to run when compared to the Neural Network or Ensemble. The parameters tested for the Random Forest were the max tree depth, max leaf nodes, number of estimators, and epsilon. Our Monte Carlo search was run for only 20 epochs/games, since our Monte Carlo approximator did do any learning we only needed a general average of its performance. Furthermore, it would take the longest to run at higher depth levels, specifically it took 30 minutes per epoch at a depth level of 5. Given the relative simplicity of Monte Carlo search, we only could modify its depth level value and the value of epsilon.

<img src="images/table.png?raw=true" width="450" />
 
The quantitative data collection setup was generally the same across all four of the approximators we tested. For each parameter value we tested we would record two sets of data, the cumulative reward over all of the epochs/games and the delta reward over all epochs/games, which is difference in cumulative reward between the agent with altered parameters and the averaged cumulative reward of the three unaltered agents. Using the collected data values we would then produce a graph for each data set, placing the epochs on the x-axis and the collected values on the y-axis. This way we could see how these two values changed as the altered agent learned over the course of the epochs, allowing us to observe how the parameter value being tested affected the modified agent's performance relative to unmodified agents. In addition to the two graphs produced for each parameter value, we also produced a heat map for each parameter of all approximators except Monte Carlo search. Each parameter’s heat map contained all of the values tested for that parameter on the x-axis and the cumulative rewards at 10 set intervals on the y-axis. The heat maps would allow us to see any strong effects a parameter value could have on an agent as it learned, while also being more easily comparable to the other values tested.
 
When we had all four of our functional approximators play against each other, we also had a unique setup for quantitative data collection methods. The four agents played against each other for 100 epochs/games, where the machine learning agents would train after each game. The quantitative evaluation consisted of recording the cumulative reward of each agent at the end of an epoch/game. Afterwards, the cumulative rewards of each agent would be plotted on a graph, with the x-axis being the epoch/game number and the y-axis being the agent’s cumulative reward. Using these graphs we could see how each agent performed relative to each other as the machine learning agents learned. In addition to collecting each agent’s cumulative reward, we also collected which agent won each game. This data was later plotted on a graph with the game number on the x-axis and the ID of the agent who won on the y-axis. This graph of winners allowed us to see a generalized view of which agents were winning more often as the machine learning agents learned.
 
Our qualitative analysis setup mostly consisted of us observing the GUI as the epochs progressed. Our general goal for each parameter was to try to observe the early epochs, some epochs in the middle of the range of epochs, and some epochs at the end. To facilitate this, we would print out the epoch number as the game ran using a certain parameter value. We did not observe all of the epochs for all of the parameter values, due to the long run times of some algorithms. In general, the majority of the results came from our quantitative analysis, since we measured the learning and success of our agents in terms of their cumulative reward.

For the Neural Network agent we first evaluated the number of hidden layers. We found that as more layers were added after the default value of 1, the volatility of the cumulative reward would increase while consistency decreased. This trend in cumulative reward can be seen in the heatmap below: 

<img src="images/neural/layers/neural_heat_hidden_layers.png?raw=true" width="400" /> 

We ended up recommending 1 hidden layers as it was the most consistent; the graph of its cumulative reward and delta reward can be seen below: 

<img src="images/neural/layers/neural_cumulative_Hidden Layers_1.png?raw=true" width="450" /> <img src="images/neural/layers/neural_delta_Hidden Layers_1.png?raw=true" width="450" /> 

The cumulative graph shows an upward trend in cumulative reward with high peaks while the delta reward shows consistent performance over 0 meaning it is performing well against the other agents. An example of another value with high impact would be 20; while it has high influence, it is negative influence which can be seen in the graphs below: 

<img src="images/neural/layers/neural_cumulative_Hidden Layers_20.png?raw=true" width="450" /> <img src="images/neural/layers/neural_delta_Hidden Layers_20.png?raw=true" width="450" /> 

While the cumulative data does trend upwards, it is much more volatile and inconsistent which makes it a worse choice. The delta reward shows that this value results in most data being below 0 which indicates poor performance against the other agents. For the number of nodes per layer, we found that high node counts resulted in higher reward averages and better learning rates compared to the other baseline agents overall. The overall trend in cumulative rewards can be seen in the heatmap below: 

<img src="images/neural/nodes/neural_heat_num_nodes.png?raw=true" width="400" /> 

Based on these values we chose 64 as the best, most positively impactful parameter as the cumulative reward improves at this value with a mostly positive delta reward. The graphs for the cumulative and delta reward can be seen below: 

<img src="images/neural/nodes/neural_cumulative_Nodes Per Layer_64.png?raw=true" width="450" /> <img src="images/neural/nodes/neural_delta_Nodes Per Layer_64.png?raw=true" width="450" /> 

As you can see the cumulative reward has a high average while trending upward, and the delta reward has mostly positive values compared to the other values. An example of another parameter with high impact would be 2; this value was very influential on the performance in a negative way. The performance can be seen below: 

<img src="images/neural/nodes/neural_cumulative_Nodes Per Layer_2.png?raw=true" width="450" /> <img src="images/neural/nodes/neural_delta_Nodes Per Layer_2.png?raw=true" width="450" /> 

The cumulative reward for this value had lower averages while the delta reward had a negative average value with a downward trend. For the activation function, we found that the performance was similar for each value which would mean that this parameter was not very impactful overall. The performance for each value is summarized below: 

<img src="images/neural/activation/neural_heat_activation.png?raw=true" width="400" /> 

Based on the data, ReLU seems to have the most positive impact on the performance; its data can be seen below: 

<img src="images/neural/activation/neural_cumulative_Activation Function_relu.png?raw=true" width="450" /> <img src="images/neural/activation/neural_delta_Activation Function_relu.png?raw=true" width="450" /> 

The cumulative reward has a high average, upward trend showing improvement, lower lows, and higher highs than the other data. The delta reward has slightly more values over 0 than the others but is mostly the same. An example of a value that negatively impacted performance is Swish; its data can be seen below: 

<img src="images/neural/activation/neural_cumulative_Activation Function_swish.png?raw=true" width="450" /> <img src="images/neural/activation/neural_delta_Activation Function_swish.png?raw=true" width="450" /> 

The cumulative reward has the lowest average of the three, a much smaller upward trend, and very low dips in performance. The delta reward also shows massive performance dips and higher volatility than the other functions which let us conclude that it had the highest negative impact. For the epsilon values, we found that higher values displayed more volatility and erratic behavior while lower values were more consistent and showed more learning. The summary of the performance of each value can be seen in the heatmap below:

<img src="images/neural/epsilon/neural_heat_epsilon.png?raw=true" width="400" /> 

Overall we found 0.1 to have the best performance and consistency based on the data below: 

<img src="images/neural/epsilon/neural_cumulative_Epsilon_0.1.png?raw=true" width="450" /> <img src="images/neural/epsilon/neural_delta_Epsilon_0.1.png?raw=true" width="450" /> 

The cumulative reward data shows a strong average and upward trend which means there is strong improvement. The delta reward also shows a higher average and slight upward trend which let us conclude that this value of 0.1 is in fact the most positively influential of all the epsilon values. The most negatively impactful value was found to be 0.5; the data for this can be seen below: 

<img src="images/neural/epsilon/neural_cumulative_Epsilon_0.5.png?raw=true" width="450" /> <img src="images/neural/epsilon/neural_delta_Epsilon_0.5.png?raw=true" width="450" /> 

The cumulative reward data is completely erratic and inconsistent while the delta reward data is the same but also with a downward trend. Overall this shows that 0.5 is much too high of an epsilon value and that the epsilon parameter itself is a very impactful parameter. To summarize, the most impactful parameters were number of hidden layers, number of nodes per layer, and epsilon while the least impactful was the activation function.

In the Voting Regressor Ensemble agent, we first tested the number of estimators for the Adaboost Regressor and found, surprisingly, that the performance was generally the same for all values implying that this parameter had low impact. The summary of the cumulative reward for each value can be seen below: 

<img src="images/ensemble/estimators/ensemble_heat_ada_estimators.png?raw=true" width="400" /> 

We found 200 to be the most positively impactful value because of its consistency; the data for this can be seen below: 

<img src="images/ensemble/estimators/ensemble_cumulative_ada_estimators_200.png?raw=true" width="450" /> <img src="images/ensemble/estimators/ensemble_delta_ada_estimators_200.png?raw=true" width="450" /> 

The cumulative reward for this value has an upward trend with very consistent values, high peaks when it does deviate from the average, and small dips compared to the other values. The delta reward is similar to the others but with less volatility. The most negatively influential value was found to be 25; the data for this value can be seen below:

<img src="images/ensemble/estimators/ensemble_cumulative_ada_estimators_25.png?raw=true" width="450" /> <img src="images/ensemble/estimators/ensemble_delta_ada_estimators_25.png?raw=true" width="450" /> 

The cumulative reward for this value was lower than the others on average and displayed much more volatile results. The delta reward was largely negative with a strong downward trend which shows decreased learning. For the Adaboost Regressor loss function, we found that the data was very similar for each value which means that this was not a very impactful parameter. The summary of cumulative rewards can be seen below: 

<img src="images/ensemble/loss/ensemble_heat_ada_lossfunction.png?raw=true" width="400" /> 

The best value was found to be exponential; the data for this can be seen below: 

<img src="images/ensemble/loss/ensemble_cumulative_ada_loss_exponential.png?raw=true" width="450" /> <img src="images/ensemble/loss/ensemble_delta_ada_loss_exponential.png?raw=true" width="450" /> 

The cumulative reward data for this value was similar to the other data except that it has more and higher peaks while not dipping in performance by a large amount. The delta reward data shows this value to have the higher average value and a more constant trend rather than downward. The most negatively influential value was found to be the square loss function as it has the lower values, although not by much. This data can be seen below: 

<img src="images/ensemble/loss/ensemble_cumulative_ada_loss_square.png?raw=true" width="450" /> <img src="images/ensemble/loss/ensemble_delta_ada_loss_square.png?raw=true" width="450" /> 

Overall this parameter didn’t have much impact as the data was very similar for each value. For the Support Vector Machine Kernel, we found that changing this parameter impacted the volatility of the performance. The summary of the cumulative reward data for the chosen values can be seen below: 

<img src="images/ensemble/kernal/ensemble_heat_svr_kernal.png?raw=true" width="400" /> 

We found that the Poly kernel had the best performance; its data can be seen below: 

<img src="images/ensemble/kernal/ensemble_cumulative_svr_kernel_poly.png?raw=true" width="450" /> <img src="images/ensemble/kernal/ensemble_delta_svr_kernel_poly.png?raw=true" width="450" /> 

The cumulative rewards for this value had the same average as the others but with fewer dips and more/higher peaks. The delta reward data shows the weakest downward trend of all the values with some of the most positive values which made us conclude in this value being the most positively impactful value. The most negatively impactful parameter was found to be RBF; its data can be seen below: 

<img src="images/ensemble/kernal/ensemble_cumulative_svr_kernel_rbf.png?raw=true" width="450" /> <img src="images/ensemble/kernal/ensemble_delta_svr_kernel_rbf.png?raw=true" width="450" />  

This value’s cumulative rewards has the most dips in performance, and its delta rewards has a strong downward trend with mostly negative values which lets us conclude that this value is the worst for performance. For the MultiLayer Perceptron activation function, we found that changing this parameter significantly changes the volatility and overall behavior of the agent performance. The summary of the performance can be seen below: 

<img src="images/ensemble/activation/ensemble_heat_mlp_activation.png?raw=true" width="400" /> 

We found the ReLU activation function to be the most positively impactful value; the data for this can be seen below: 

<img src="images/ensemble/activation/ensemble_cumulative_mlp_activation_relu.png?raw=true" width="450" /> <img src="images/ensemble/activation/ensemble_delta_mlp_activation_relu.png?raw=true" width="450" /> 

The cumulative reward for this value is not as consistent as some of the other values but it has a higher average reward value and much higher peaks. The delta reward data shows that ReLU has the most positive reward values compared to the other values. The most negatively influential value was found to be TanH; its data can be seen below: 

<img src="images/ensemble/activation/ensemble_cumulative_mlp_activation_tanh.png?raw=true" width="450" /> <img src="images/ensemble/activation/ensemble_delta_mlp_activation_tanh.png?raw=true" width="450" /> 

The cumulative rewards for this value are lower on average with smaller peaks, and the delta reward data shows TanH to have much more negative values than the other parameter values.
For the MultiLayer Perceptron number of layers, we found that changing this parameter greatly affected the performance of the agent. The summary of the performance can be seen in the heatmap below: 

<img src="images/ensemble/layers/ensemble_heat_mlp_numlayers.png?raw=true" width="400" /> 

We found that the best value to use is 50; the data for this can be seen below: 

<img src="images/ensemble/layers/ensemble_cumulative_mlp_numlayers_50.png?raw=true" width="450" /> <img src="images/ensemble/layers/ensemble_delta_mlp_numlayers_50.png?raw=true" width="450" /> 

The cumulative rewards for 50 layers is very consistent with good average values and an upward trend. The delta reward is very positive with a strong upward trend as well which singles this value out as the best overall. The worst parameter was found to be 2 hidden layers; its data can be seen below: 

<img src="images/ensemble/layers/ensemble_cumulative_mlp_numlayers_2.png?raw=true" width="450" /> <img src="images/ensemble/layers/ensemble_delta_mlp_numlayers_2.png?raw=true" width="450" /> 

The cumulative reward was very consistent but at a very low value with only a few peaks. The delta reward has mostly negative values and a strong downward trend. For the epsilon values, we found this value to increase volatility and randomness. The summary of the performance for the values chosen can be seen below: 

<img src="images/ensemble/epsilon/ensemble_heat_epsilon.png?raw=true" width="400" /> 

We found 0.1 to be the best value for this parameter overall; the data for this can be seen below: 

<img src="images/ensemble/epsilon/ensemble_cumulative_Epsilon_0.1.png?raw=true" width="450" /> <img src="images/ensemble/epsilon/ensemble_delta_Epsilon_0.1.png?raw=true" width="450" /> 

The cumulative reward is much more consistent than the others with a stronger upward trend. The delta reward shows many positive values and a slight upward trend which is better than the other parameter values. The worst choice would be 0.5; its data can be seen below: 

<img src="images/ensemble/epsilon/ensemble_cumulative_Epsilon_0.5.png?raw=true" width="450" /> <img src="images/ensemble/epsilon/ensemble_delta_Epsilon_0.5.png?raw=true" width="450" /> 

The cumulative reward shows the lowest average performance and a very low plateau compared to the others. The delta reward showed lower values than the other parameter values and a slight downward trend. To summarize, the highest impact parameters are the Support Vector Machine kernel, the MultiLayer Perceptron activation function and hidden layers, and epsilon; while, the lowest impact parameters were found to be the Adaboost number of estimators and Adaboost loss function.

In our Random Forest agent one of the parameters we tested was the number of estimators/decision trees used in our Random Forest. We found that the number of estimators doesn’t show a noticeable impact on the performance of the agent. At low and high numbers of estimators, the delta rewards don't trend upwards nor do they trend downwards. Graphs for 10 and 200 estimators are shown below:

<img src="images/forest/estimators/forest_delta_estimators_10.png?raw=true" width="450" /> <img src="images/forest/estimators/forest_delta_estimators_200.png?raw=true" width="450" />

These trends are also consistent with the cumulative rewards collected. However, 100 estimators did show a slight greater average cumulative reward, so we decided it was the more favorable value. Cumulative and delta rewards for 100 estimators are shown below:

<img src="images/forest/estimators/forest_cumulative_estimators_100.png?raw=true" width="450" /> <img src="images/forest/estimators/forest_delta_estimators_100.png?raw=true" width="450" />

When looking at the heat map for the different number of estimators over many epochs, there is no noticeable trend upwards nor downwards for any of the values. Some values appear brighter and have a higher overall cumulative reward average like 100, and some have a lower overall average like 150.

<img src="images/forest/estimators/forest_heat_estimators.png?raw=true" width="400" />

For Random Forest’s max depth, we found this parameter has an impact on the agent’s learning. When the max depth is low or set to none you are able to see a noticeable increase in the cumulative reward. Cumulative reward for max depth of 5 and None are shown below:

<img src="images/forest/depth/forest_cumulative_Max Depth_5.png?raw=true" width="450" /> <img src="images/forest/depth/forest_cumulative_Max Depth_None.png?raw=true" width="450" />

However, as the number of max depth increases there is a noticeable decline in performance. When the max depth is set to 200, the delta reward averages between 0 and -2.5, and the cumulative reward shows a constant trend. This means that depth 200 performs worse than a depth of None, and it shows no noticeable growth. A similar behavior was observed with a max depth of 100. We concluded that this is probably due to overfitting. When deciding the most favorable max depth, we were stuck between None and 5. We ended up choosing None because it was the least volatile in the last 50 epochs. Cumulative and delta rewards for max depth of 200 are shown below:

<img src="images/forest/depth/forest_cumulative_Max Depth_200.png?raw=true" width="450" /> <img src="images/forest/depth/forest_delta_Max Depth_200.png?raw=true" width="450" />

A heat map showing the full set of values tested for max depth can be seen below:

<img src="images/forest/depth/forest_heat_maxdepth.png?raw=true" width="400" />

When looking at the heat map, a max depth value of 10 might appear to be a stronger choice than None. However, once you look at it’s cumulative reward graph, its high volatility deterred our group from considering it to a good value. Cumulative reward for max depth of 10 shown below:

<img src="images/forest/depth/forest_cumulative_Max Depth_10.png?raw=true" width="450" />

The third parameter we tested in Random Forest was the max leaf nodes which we found None to be the best cumulative and delta rewards. The cumulative and delta reward for None are show below:

<img src="images/forest/leaf/forest_cumulative_Max Leaf Nodes_None.png?raw=true" width="450" />

As the number of epochs increases, there is an increase in the cumulative reward. While the increase in rewards appears slight it is better than values we tested that show even smaller increases to no increases at all. Cumulative and delta reward for max leaf nodes of 10.

<img src="images/forest/leaf/forest_cumulative_Max Leaf Nodes_10.png?raw=true" width="450" /> <img src="images/forest/leaf/forest_delta_Max Leaf Nodes_10.png?raw=true" width="450" />

The cumulative reward is very volatile and shows no apparent trends upward. The average cumulative reward in the last couple epochs are noticeably smaller for 10 max leaf nodes than for max leaf nodes of None. In the delta reward graph above, the average delta reward is around -2.5 which shows its weak performance in comparison to None. A heat map showing the performance of all other values and be seen below:

<img src="images/forest/leaf/forest_heat_leafnodes.png?raw=true" width="400" />

The heat map shows a max leaf node of None is quite consistent with a couple of high peaks. Other values are more volatile and unpredictable, with some values appearing to have a decreasing trend like 25.
The last value we tested for was epsilon. We found an epsilon value of 0.1 to be the most favorable. However values greater than 0.1 made the agent perform worse and more random. Cumulative and delta reward for 0.1 epsilon is shown below:

<img src="images/forest/epsilon/forest_cumulative_Epsilon_0.1.png?raw=true" width="450" /> <img src="images/forest/epsilon/forest_delta_Epsilon_0.1.png?raw=true" width="450" />

Both the cumulative and delta reward show an upwards trend. Out of all the other epsilon values, it shows the strongest trend upwards. Because of it’s upwards trend in delta rewards, this suggests that it can possibly out perform epsilon value of 0.01. Our group decided the epsilon value of 0.1 was the most favorable. The rewards for epsilon value of 0.05 is shown below:

<img src="images/forest/epsilon/forest_cumulative_Epsilon_0.05.png?raw=true" width="450" /> <img src="images/forest/epsilon/forest_delta_Epsilon_0.05.png?raw=true" width="450" />

Epsilon 0.05 is a value that showed poor performance. In the cumulative reward, there is no apparent trend upward, and reward is very volatile in comparison to epsilon value of 0.1. When looking at the delta rewards, there is no signs that a value of 0.05 is better than a value of 0.01. A heat map showing the full set of epsilon value and cumulative reward is shown below:

<img src="images/forest/epsilon/forest_heat_epsilon.png?raw=true" width="400" />

While the epsilon value of 0.1 starts with low cumulative reward, its rewards steadily increase and ends with the highest cumulative rewards in the last 2 epochs recorded. For our Random Forest agent, the most impactful parameters were the max depth, max leaf nodes, and epsilon while the least impactful value was the number of estimators.
The first parameter we tested for our Monte Carlo Search Tree agent was search depth. We found that no particular depth proved to be better than the others. Because of how Monte Carlo Search tree operates, cumulative reward was not expected to increase over more epochs. However, we did want to observe the average cumulative reward. A depth of one showed to be the slightly more favorable value. Cumulative and delta reward for depth 1 is shown below:

<img src="images/mcsearch/depth/mcsearch_cumulative_depth_1.png?raw=true" width="450" /> <img src="images/mcsearch/depth/mcsearch_delta_depth_1.png?raw=true" width="450" />

In the cumulative reward graph, depth 1 has 2 tall peaks which are greater than the peaks in other graphs. One of the depth that performed more poorly was a depth of 4. Cumulative and delta reward for depth of 4 shown below:

<img src="images/mcsearch/depth/mcsearch_cumulative_depth_4.png?raw=true" width="450" /> <img src="images/mcsearch/depth/mcsearch_delta_depth_4.png?raw=true" width="450" />

The cumulative reward experiences many lows below 25 which appears to be the average. The delta rewards experiences a similar graph showing that it’s performs worse than depth of 2.
The last parameter we tested was epsilon. We found that an epsilon value of 0.1 performs the best. The cumulative and delta reward for epsilon value of 0.1 is shown below:

<img src="images/mcsearch/epsilon/mcsearch_cumulative_epsilon_0.1.png?raw=true" width="450" /><img src="images/mcsearch/epsilon/mcsearch_delta_epsilon_0.1.png?raw=true" width="450" />

The cumulative reward reach has an average around 20 and strong peaks reach up to 40. Based on the delta reward graph, it appears to perform better than an epsilon value of 0.01. An epsilon value that doesn’t perform so well is 0.05. Cumulative and delta reward for 0.5 is shown below:

<img src="images/mcsearch/epsilon/mcsearch_cumulative_epsilon_0.05.png?raw=true" width="450" /><img src="images/mcsearch/epsilon/mcsearch_delta_epsilon_0.05.png?raw=true" width="450" />

The delta reward shows many lows and seems to underperform compared to epsilon of 2. Based on the cumulative reward, it performs decently but not as well as the epsilon value of 0.1. As the epsilon value gets greater, the average cumulative value drops dramatically. Epsilon is an impactful parameter in our Monte Carlo Search Tree agent, but the depth parameter has little impact on the performance of the agent. While no particular depth proved to result in a higher cumulative reward, depth 1 was favored mainly because it required less runtime.



## References:

- [ColosseumRL](https://colosseumrl.igb.uci.edu/)
- ColosseumRL setup/example from course website
- [SciKit-Learn](https://scikit-learn.org/stable/)
- [Ray](https://www.ray.io/)
- [Gym](https://gym.openai.com/)
