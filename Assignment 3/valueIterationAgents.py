# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
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
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        for _ in range(self.iterations):
            # Using a Counter to store new values
            newValues = util.Counter()
            for state in self.mdp.getStates():
                if not self.mdp.isTerminal(state):
                    newValues[state] = max(self.computeQValueFromValues(state, action) for action in self.mdp.getPossibleActions(state))
            self.values = newValues

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
        qValue = 0
        for pairs in self.mdp.getTransitionStatesAndProbs(state, action):
            qValue += pairs[1]*(self.mdp.getReward(state, action, pairs[0]) + self.discount * self.values[pairs[0]])
        return qValue

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        count = util.Counter()
        for a in self.mdp.getPossibleActions(state):
            count[a] = self.getQValue(state, a)
        if count == {}:
            return None
        else:
            return count.argMax()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        # Write value iteration code here
        states = self.mdp.getStates()
        for iteration in range(self.iterations):
            state = states[iteration % len(states)]
            if not self.mdp.isTerminal(state):
                actions = self.mdp.getPossibleActions(state)
                maximumValue = max([self.getQValue(state,action) for action in actions])
                self.values[state] = maximumValue

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        # Priority queue for prioritized sweeping
        priorityQueue = util.PriorityQueue()
        predecessors_dict = {}

        # Finding all predecessors
        for currentState in self.mdp.getStates():
            if not self.mdp.isTerminal(currentState):
                for currentAction in self.mdp.getPossibleActions(currentState):
                    for nextState, probability in self.mdp.getTransitionStatesAndProbs(currentState, currentAction):
                        if nextState in predecessors_dict:
                            predecessors_dict[nextState].add(currentState)
                        else:
                            predecessors_dict[nextState] = {currentState}

        # Initializing the priority queue with state differences
        for currentState in self.mdp.getStates():
            if not self.mdp.isTerminal(currentState):
                difference = abs(self.values[currentState] - max(
                    [self.computeQValueFromValues(currentState, currentAction) for currentAction in
                     self.mdp.getPossibleActions(currentState)]))
                # Pushing negative into the queue
                priorityQueue.update(currentState, -difference)

        # Main iteration loop
        for iteration in range(self.iterations):
            if priorityQueue.isEmpty():
                break
            currentState = priorityQueue.pop()
            if not self.mdp.isTerminal(currentState):
                # Update the value of the current state
                self.values[currentState] = max([self.computeQValueFromValues(currentState, currentAction)
                                                 for currentAction in self.mdp.getPossibleActions(currentState)])

            # Updating predecessors in the priority queue
            for predecessorState in predecessors_dict[currentState]:
                if not self.mdp.isTerminal(predecessorState):
                    difference = abs(self.values[predecessorState] - max(
                        [self.computeQValueFromValues(predecessorState, action)
                         for action in self.mdp.getPossibleActions(predecessorState)]))

                    # If the difference is above the threshold, update the priority queue
                    if difference > self.theta:
                        priorityQueue.update(predecessorState, -difference)



