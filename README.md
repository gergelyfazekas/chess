# Chess
A chess interface playable from the command line with a built-in chess bot for opponent.


## Example


https://user-images.githubusercontent.com/93786486/215427592-c713336e-387c-43a1-80f5-16452daff371.mov


## Details

**Initializing a game**

<img width="735" alt="Screenshot 2023-01-30 at 9 50 29" src="https://user-images.githubusercontent.com/93786486/215430272-2b70d33b-cddd-4b51-a43b-f9da71b96c66.png">

Different modes are available to each player. Human mode prompts the user for choosing the next step, while robot and ai modes choose automatically.

**Chess bot**

In the ai mode a neural network based algorithm is used for choosing the next move. The model was trained using a reinforcement learning approach by playing against itself repeatedly. The robot mode is also able to move autonomously, however it employs a more greedy approach. More specifically, it looks ahead k steps with an exhaustive search and chooses the next move based on the total piece values of both players at that k-step-ahead hypothetical position.



