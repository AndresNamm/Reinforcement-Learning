import pdb, mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount=0.9 , iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """

        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0


        "*** YOUR CODE HERE ***"


        #Initiliaization for V0
        states = mdp.getStates()
        for state in states:
            self.values[state]=0

        for i in range(self.iterations): # iterations *
            temp=util.Counter()
            for state in states: #  states *
                qVals=[]
                if self.mdp.isTerminal(state):
                    temp[state]=0
                else:
                    for action in self.mdp.getPossibleActions(state): # actions * | tests out alll actions to find the action with highes q values
                        qVal=0
                        for t in self.mdp.getTransitionStatesAndProbs(state,action):
                            qVal+=t[1]*(mdp.getReward(state,action,t[0])+self.discount*self.values[t[0]])
                        qVals.append(qVal)
                    temp[state]=max(qVals)
                #In end of iteration updata V-values for all states
            self.values=temp.copy()

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]
    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        if self.mdp.isTerminal(state):
            return 0
        transitions=self.mdp.getTransitionStatesAndProbs(state,action) # returns list of items in shape (state,prob)
        qVal=0
        for transition in transitions:
            qVal+=transition[1]*(self.mdp.getReward(state,action,transition[0])+self.discount*self.getValue(transition[0]))
        print("Qval", qVal)
        return qVal

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        legalActions=self.mdp.getPossibleActions(state) # Find all actions achievable from state.
        if self.mdp.isTerminal(state): # Means terminal state. No possible actions
            return None
        else:
            # Return Q-values for all actions.
            qVals = [self.computeQValueFromValues(state, a) for a in legalActions]
            mQ=max(qVals) # Find best Qval = V(state)
            for i in range(len(legalActions)):
                if qVals[i]==mQ:
                    # Return the action that returnn V-value.
                    return legalActions[i]

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
