# Chess
A chess interface playable from the command line with a built-in chess bot for opponent


## Example


https://user-images.githubusercontent.com/93786486/215427592-c713336e-387c-43a1-80f5-16452daff371.mov


## Details

**Initializing a game**

<img width="689" alt="Screenshot 2023-01-30 at 9 43 25" src="https://user-images.githubusercontent.com/93786486/215428838-90f49d56-b0e8-4fd6-b928-c91f7a16db14.png">

Different modes are available to each player. Human mode prompts the user for choosing the next step, while robot and ai modes choose automatically.


**Chess bot**

In the ai mode a neural network based algorithm is used for choosing the next move which was trained using a reinforcement / genetic learning approach. The robot mode looks ahead n steps with an exhaustive search and decides based on the total piece values of the players at that position.



