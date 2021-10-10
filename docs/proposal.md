---
layout:	default
title:	Proposal
---

## Project Summary
Our project will be implementing an AI to go through a platformer/maze course while utilizing the items in its inventory to succeed. The AI will be given the location it is in of the platformer/puzzle along with the items in its inventory, and it will take an action to get to the next step of the course (use potion, jump, sprint jump, go left, go right, go back, go forward). The agent will mainly be using the items and its available actions (besides combat) to solve puzzles in the course.
The inventory items and use case example scenarios can be seen below:  
  - Potion of Leaping to clear a high jump
  - Potion of fire resistance to cross lava
  - Potion of swiftness for speed
  - Potion of invisibility to sneak past mobs
  - Potion of water breathing to go underwater

## AI/ML Algorithms
Reinforcement learning with neural function approximator

## Evaluation Plan
Our primary metric for evaluating the effectiveness of our agent will be how far the agent gets in the course. The course we build will be broken into steps (1, 2, 3, 4) depending on how far it gets, for example it starts at step 0 and if it completes the first jump for the platformer it would get to step 2, etc. The secondary metric would be the time it takes for the agent to get to the next step/phase of the course. The baseline would be that the agent is stuck at the very beginning (step 0) or keeps dying in lava.

## Appointment with the Instructor
2:15pm October 19th
