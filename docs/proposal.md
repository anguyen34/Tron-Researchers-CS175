---
layout:	default
title:	Proposal
---

## Project Summary
Our project will be implementing an AI to go through a linear platformer/maze course while utilizing the items in its inventory to succeed. The AI will be given the location it is in of the platformer/puzzle along with the items in its inventory, and it will take an action to get to the next step of the course (use potion, jump, sprint jump, go left, go right, go back, go forward). The agent will mainly be using the items and its available actions (besides combat) to solve puzzles in the course. Harder puzzles/courses would require the AI to use multiple items simultaneously to succeed.
The inventory items and use case example scenarios can be seen below:  
  - Potion of Leaping to clear a high jump
  - Potion of fire resistance to cross lava
  - Potion of swiftness for speed
  - Potion of invisibility to sneak past mobs
  - Potion of water breathing to go underwater

## AI/ML Algorithms
We want to reinforcement learning with neural function approximator.

## Evaluation Plan
  Our primary metric for evaluating the effectiveness of our agent will be how far the agent gets in the course. The course we build will be broken into steps (1, 2, 3, 4) depending on how far it gets, for example it starts at step 0 and if it completes the first jump for the platformer it would get to step 2, etc. The secondary metric would be the time it takes for the agent to get to the next step/phase of the course. The baseline would be that the agent is stuck at the very beginning (step 0) or keeps dying in lava. Another metric is to have different difficulty levels for the mazes/courses, for which a numeric value could be assigned to each level.

  For our qualitative approach our sanity cases is us observing the agent using the correct item for the simplest of courses. An example would be the AI using a potion of fire resistance when it sees a large pool of lava it needs to cross. Another example would be using a sword to clear cobwebs. Our moonshot case could be the agent being able to solve any course we come up with or potentially any linear course that could be made. In such a moonshot case the agent would have to use multiple items to clear an area of the course, while also using various movement techniques available in Minecraft.

## Appointment with the Instructor
2:15pm October 19th
