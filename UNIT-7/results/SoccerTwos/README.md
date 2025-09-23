---
library_name: ml-agents
tags:
- SoccerTwos
- deep-reinforcement-learning
- reinforcement-learning
- ML-Agents-SoccerTwos
---

  # **poca** Agent playing **SoccerTwos**
  This is a trained model of a **poca** agent playing **SoccerTwos**
  using the [Unity ML-Agents Library](https://github.com/Unity-Technologies/ml-agents).

  ## Usage (with ML-Agents)
  The Documentation: https://unity-technologies.github.io/ml-agents/ML-Agents-Toolkit-Documentation/

  We wrote a complete tutorial to learn to train your first agent using ML-Agents and publish it to the Hub:
  - A *short tutorial* where you teach Huggy the Dog 🐶 to fetch the stick and then play with him directly in your
  browser: https://huggingface.co/learn/deep-rl-course/unitbonus1/introduction
  - A *longer tutorial* to understand how works ML-Agents:
  https://huggingface.co/learn/deep-rl-course/unit5/introduction

  ### Resume the training
  ```bash
  mlagents-learn <your_configuration_file_path.yaml> --run-id=<run_id> --resume
  ```

  ### Watch your Agent play
  You can watch your agent **playing directly in your browser**

  1. If the environment is part of ML-Agents official environments, go to https://huggingface.co/unity
  2. Step 1: Find your model_id: OxoGhost/poca-SoccerTwos
  3. Step 2: Select your *.nn /*.onnx file
  4. Click on Watch the agent play 👀
  