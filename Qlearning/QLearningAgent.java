package ticTacToe;

import java.util.List;
import java.util.Random;
import java.util.HashMap;


/**
 * A Q-Learning agent with a Q-Table, i.e. a table of Q-Values. This table is implemented in the {@link QTable} class.
 * 
 *  The methods to implement are: 
 * (1) {@link QLearningAgent#train}
 * (2) {@link QLearningAgent#extractPolicy}
 * 
 * Your agent acts in a {@link TTTEnvironment} which provides the method {@link TTTEnvironment#executeMove} which returns an {@link Outcome} object, in other words
 * an [s,a,r,s']: source state, action taken, reward received, and the target state after the opponent has played their move. You may want/need to edit
 * {@link TTTEnvironment} - but you probably won't need to. 
 * @author ae187
 */

public class QLearningAgent extends Agent {
	
	/**
	 * The learning rate, between 0 and 1.
	 */
	double alpha=0.1;
	
	/**
	 * The number of episodes to train for
	 */
	int numEpisodes=10000;
	
	/**
	 * The discount factor (gamma)
	 */
	double discount=0.9;
	
	
	/**
	 * The epsilon in the epsilon greedy policy used during training.
	 */
	double epsilon=0.1;
	
	/**
	 * This is the Q-Table. To get an value for an (s,a) pair, i.e. a (game, move) pair.
	 * 
	 */
	
	QTable qTable=new QTable();
	
	
	/**
	 * This is the Reinforcement Learning environment that this agent will interact with when it is training.
	 * By default, the opponent is the random agent which should make your q learning agent learn the same policy 
	 * as your value iteration and policy iteration agents.
	 */
	TTTEnvironment env=new TTTEnvironment();
	
	
	/**
	 * Construct a Q-Learning agent that learns from interactions with {@code opponent}.
	 * @param opponent the opponent agent that this Q-Learning agent will interact with to learn.
	 * @param learningRate This is the rate at which the agent learns. Alpha from your lectures.
	 * @param numEpisodes The number of episodes (games) to train for
	 */
	public QLearningAgent(Agent opponent, double learningRate, int numEpisodes, double discount)
	{
		env=new TTTEnvironment(opponent);
		this.alpha=learningRate;
		this.numEpisodes=numEpisodes;
		this.discount=discount;
		initQTable();
		train();
	}
	
	/**
	 * Initialises all valid q-values -- Q(g,m) -- to 0.
	 *  
	 */
	
	protected void initQTable()
	{
		List<Game> allGames=Game.generateAllValidGames('X');//all valid games where it is X's turn, or it's terminal.
		for(Game g: allGames)
		{
			List<Move> moves=g.getPossibleMoves();
			for(Move m: moves)
			{
				this.qTable.addQValue(g, m, 0.0);
				//System.out.println("initing q value. Game:"+g);
				//System.out.println("Move:"+m);
			}
			
		}
		
	}
	
	/**
	 * Uses default parameters for the opponent (a RandomAgent) and the learning rate (0.2). Use other constructor to set these manually.
	 */
	public QLearningAgent()
	{
		this(new RandomAgent(), 0.1, 75000, 0.9);
		
	}
	
	
	/**
	 *  Implement this method. It should play {@code this.numEpisodes} episodes of Tic-Tac-Toe with the TTTEnvironment, updating q-values according 
	 *  to the Q-Learning algorithm as required. The agent should play according to an epsilon-greedy policy where with the probability {@code epsilon} the
	 *  agent explores, and with probability {@code 1-epsilon}, it exploits. 
	 *  
	 *  At the end of this method you should always call the {@code extractPolicy()} method to extract the policy from the learned q-values. This is currently
	 *  done for you on the last line of the method.
	 */
	
	public void train() {
	    Random random = new Random(); //Create a Random object to generate random numbers (used for probability)


	    for (int episode = 0; episode < numEpisodes; episode++) {//loops through each training episode
	        // Reset environment for a new game
	        env.reset();
	        Game currentState = env.getCurrentGameState();//get the starting state for a game
	        boolean gameOver = false;//flag for checking if the game is Over 

	        while (!gameOver) {//loops through while the game is not over
	            Move move;//variable to store the chosen move in the current state 
	            // selecting action based on Epsilon-greedy policy
	            if (random.nextDouble() < epsilon) {//when probability is less than epsilon we choose exploration: random move
	                
	                List<Move> possibleMoves = env.getPossibleMoves();//get list of all possible moves in current state
	                move = possibleMoves.get(random.nextInt(possibleMoves.size()));//pick a random move
	            } else {//else choose exploration:best move from Q-table
	                
	                move = getBestMove(currentState);//pick the bestmove for current state using helper function
	            }

	            try {
	                // Execute the chosen move and get the outcome
	                Outcome outcome = env.executeMove(move);// Perform the chosen move and get the resulting outcome
	                if (outcome == null) break; // exit if outcome is invalid

	                // Extract details from the outcome
	                Game currState = outcome.s;          // Source state (s)
	                double reward = outcome.localReward; // Reward received (r)
	                Game nextState = outcome.sPrime;     // Destination state (s')

	                // Check if this state is terminal that is the game has ended
	                gameOver = nextState.isTerminal();

	                //Calculate the maximum Q-value for the next state
	                double maxNextQValue = Double.NEGATIVE_INFINITY;
	                if (gameOver) {//terminal state
	                    maxNextQValue = 0;//if game is over then it will be 0
	                } else {
	                    
	                	List<Move> nextPossibleMoves = nextState.getPossibleMoves();
	                    for (Move nextMove : nextPossibleMoves) {
	                        double qValue = qTable.getQValue(nextState, nextMove);
	                        if (qValue > maxNextQValue) {
	                            maxNextQValue = qValue;
	                        }
	                    }
	                }
	                double currentQValue = qTable.getQValue(currState, move);//get the q-value currently stored in the q-table for the current state
	                double newQValue = currentQValue + alpha * (reward + discount * maxNextQValue - currentQValue);//update the q value using the q learning formula

	                // Update Q-table
	                qTable.addQValue(currState, move, newQValue);

	                // Move to the next state
	                currentState = nextState;
	            } catch (IllegalMoveException e) {
	                e.printStackTrace();// Print an error if an invalid move occurs
	            }
	        }
	    }

	    // Extract the learned policy after training
	    this.policy = extractPolicy();
	}

	public Policy extractPolicy() {
        HashMap<Game, Move> extractedPolicy = new HashMap<>();//temporary map for storing extracted policy

        // For each state, select the action with the highest Q-value
        for (Game state : Game.generateAllValidGames('X')) {

         // Use getBestMove helper function to find the move with the highest Q-value for this state
            Move bestMove = getBestMove(state);

            // Store the best move for the state in the temporary policy map
            extractedPolicy.put(state, bestMove);
        }

        return new Policy(extractedPolicy); // Return the policy
    }

	/**
     * Helper method to get the action with the highest Q-value for a given state.
     */
    private Move getBestMove(Game state) {
        List<Move> possibleMoves = state.getPossibleMoves();//get the list of all possible moves
        Move bestMove = null;//initialise bestMove and best Q value for current state
        double bestQValue = Double.NEGATIVE_INFINITY;

        // Find the move with the highest Q-value
        for (Move move : possibleMoves) {//loop through all moves for particular state
            double qValue = qTable.getQValue(state, move);//get the q value for current state and move
            if (qValue > bestQValue) {
                bestQValue = qValue;
                bestMove = move;
            }
        }

        return bestMove;
    }
  
	
	public static void main(String a[]) throws IllegalMoveException
	{
		//Test method to play your agent against a human agent (yourself).
		QLearningAgent agent=new QLearningAgent();
		
		HumanAgent d=new HumanAgent();
		
		Game g=new Game(agent, d, d);
		g.playOut();
		
		
		

		
		
	}
	
	
	


	
}
