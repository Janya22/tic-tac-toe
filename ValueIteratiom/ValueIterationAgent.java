package ticTacToe;


import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * A Value Iteration Agent, only very partially implemented. The methods to implement are: 
 * (1) {@link ValueIterationAgent#iterate}
 * (2) {@link ValueIterationAgent#extractPolicy}
 * 
 * You may also want/need to edit {@link ValueIterationAgent#train} - feel free to do this, but you probably won't need to.
 * @author ae187
 *
 */
public class ValueIterationAgent extends Agent {

	/**
	 * This map is used to store the values of states
	 */
	Map<Game, Double> valueFunction=new HashMap<Game, Double>();
	
	/**
	 * the discount factor
	 */
	double discount=0.9;
	
	/**
	 * the MDP model
	 */
	TTTMDP mdp=new TTTMDP();
	
	/**
	 * the number of iterations to perform - feel free to change this/try out different numbers of iterations
	 */
	int k=10;
	
	
	/**
	 * This constructor trains the agent offline first and sets its policy
	 */
	public ValueIterationAgent()
	{
		super();
		mdp=new TTTMDP();
		this.discount=0.9;
		initValues();
		train();
	}
	
	
	/**
	 * Use this constructor to initialise your agent with an existing policy
	 * @param p
	 */
	public ValueIterationAgent(Policy p) {
		super(p);
		
	}

	public ValueIterationAgent(double discountFactor) {
		
		this.discount=discountFactor;
		mdp=new TTTMDP();
		initValues();
		train();
	}
	
	/**
	 * Initialises the {@link ValueIterationAgent#valueFunction} map, and sets the initial value of all states to 0 
	 * (V0 from the lectures). Uses {@link Game#inverseHash} and {@link Game#generateAllValidGames(char)} to do this. 
	 * 
	 */
	public void initValues()
	{
		
		List<Game> allGames=Game.generateAllValidGames('X');//all valid games where it is X's turn, or it's terminal.
		for(Game g: allGames)
			this.valueFunction.put(g, 0.0);
		
		
		
	}
	
	
	
	public ValueIterationAgent(double discountFactor, double winReward, double loseReward, double livingReward, double drawReward)
	{
		this.discount=discountFactor;
		mdp=new TTTMDP(winReward, loseReward, livingReward, drawReward);
	}
	
	/**
	 
	
	/*
	 * Performs {@link #k} value iteration steps. After running this method, the {@link ValueIterationAgent#valueFunction} map should contain
	 * the (current) values of each reachable state. You should use the {@link TTTMDP} provided to do this.
	 * 
	 *
	 */
	public void iterate() {
	    // Perform k iterations of value iteration
	    for (int iteration = 0; iteration < k; iteration++) {
	        // Temporary map to store updated values
	        HashMap<Game, Double> newValues = new HashMap<>();

	        // Loop through all states in the value function
	        for (Game g : valueFunction.keySet()) {
	            double maxValue = Double.NEGATIVE_INFINITY; // Start with a very low value

	            // Skip terminal states since they should not be evaluated for moves
	            if (g.isTerminal()) {
	                // Terminal states assigned a value of 0 and we move to the next game state
	                newValues.put(g, 0.0); 
	                continue;
	            }

	            // Get all possible moves (actions) in the current state
	            List<Move> possibleMoves = g.getPossibleMoves();

	            // loop through each possible move in the current state
	            for (Move move : possibleMoves) {
	                // Generate all transitions for this state-action pair
	                List<TransitionProb> transitions = mdp.generateTransitions(g, move);

	                double expectedValue = 0.0; // Expected value for this action initialized to 0
	                for (TransitionProb transition : transitions) {
	                    Game nextState = transition.outcome.sPrime; // Next state
	                    double reward = transition.outcome.localReward; // Reward for the transition
	                    double nextValue = valueFunction.get(nextState); // V(s')

	                    // Calculate the expected value for this transition
	                    expectedValue += transition.prob * (reward + discount * nextValue);
	                }

	                // Update the maximum value for this state
	                if (maxValue < expectedValue){
	                	
	                maxValue = expectedValue;}
	            }

	            // Update the value of the state in the temporary map
	            newValues.put(g, maxValue);
	        }

	        // Update the value function after the iteration
	        valueFunction = newValues;
	    }
	}

/**
 * Extract the policy based on the computed value function.
 * For each state, choose the action that maximizes the expected value.
 */





	
	/**This method should be run AFTER the train method to extract a policy according to {@link ValueIterationAgent#valueFunction}
	 * You will need to do a single step of expectimax from each game (state) key in {@link ValueIterationAgent#valueFunction} 
	 * to extract a policy.
	 * 
	 * @return the policy according to {@link ValueIterationAgent#valueFunction}
	 */
	public Policy extractPolicy() {
	    // Create a new Policy object to store the best action for each state
	    HashMap<Game, Move> tpolicy = new HashMap<>();

	    // Iterate through all states to determine the best action for each state
	    for (Game state : valueFunction.keySet()) {
	        if (state.isTerminal()) {
	            // Skip terminal states, no move can be made here
	            continue;
	        }

	        // Get all possible moves in the current state
	        List<Move> possibleMoves = state.getPossibleMoves();

	        // Initialize the best move and the maximum value
	        Move bestMove = null;
	        double maxValue = Double.NEGATIVE_INFINITY;

	        // Evaluate each possible move
	        for (Move move : possibleMoves) {
	            // Get the transitions for this move from the state
	            List<TransitionProb> transitions = mdp.generateTransitions(state, move);
	            double expectedValue = 0.0;//intialise the expected value to 0

	            // Compute the expected value for this move using bellman Equations
	            for (TransitionProb transition : transitions) {//for all possible transitions
	                double probability = transition.prob;
	                Outcome outcome = transition.outcome;

	                Game nextState = outcome.sPrime;  // Next state
	                double reward = outcome.localReward;//reward for transition

	                double nextStateValue = valueFunction.get(nextState);//get the value of next state from the valuefunction

	                
	                expectedValue += probability * (reward + discount * nextStateValue);
	            }

	            // If the expected value for this move is better than the current best, update the best move
	            if (expectedValue > maxValue) {
	                maxValue = expectedValue;
	                bestMove = move;
	            }
	        }

	        // The agent should always select a valid move
	        if (bestMove != null) {
	            tpolicy.put(state, bestMove);//store the best move and state in the temporary policy map created
	        } else {
	            //error handling 
	            System.out.println("Error: No valid move found for state: " + state);
	        }
	    }

	    // Return the policy
	    return new Policy(tpolicy);
	}





	
	/**
	 * This method solves the mdp using your implementation of {@link ValueIterationAgent#extractPolicy} and
	 * {@link ValueIterationAgent#iterate}. 
	 */
	public void train()
	{
		/**
		 * First run value iteration
		 */
		this.iterate();
		/**
		 * now extract policy from the values in {@link ValueIterationAgent#valueFunction} and set the agent's policy 
		 *  
		 */
		
		super.policy=extractPolicy();
		
		if (this.policy==null)
		{
			System.out.println("Unimplemented methods! First implement the iterate() & extractPolicy() methods");
			//System.exit(1);
		}
		
		
		
	}

	public static void main(String a[]) throws IllegalMoveException
	{
		//Test method to play the agent against a human agent.
		ValueIterationAgent agent=new ValueIterationAgent();
		HumanAgent d=new HumanAgent();
		
		Game g=new Game(agent, d, d);
		g.playOut();
		
		
		

		
		
	}
}
