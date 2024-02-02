# analysis.py
# -----------

######################
# ANALYSIS QUESTIONS #
######################

# Set the given parameters to obtain the specified policies through
# value iteration.

def question2():
    # To encourage long-term planning with a slight chance for random actions
    answerDiscount = 0.9
    answerNoise = 0
    return answerDiscount, answerNoise

def question3a():
    # Aim to survive for 3 steps, penalize living to discourage unnecessary delay, no noise
    answerDiscount = 0.1
    answerNoise = 0
    answerLivingReward = -4.0
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3b():
    # Aim to survive for 7 steps, add noise to make the agent cautious of fire
    answerDiscount = 0.1
    answerNoise = 0.1
    answerLivingReward = -1
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3c():
    # Aim to live for 5 steps, no noise to avoid making the agent fear the fire
    answerDiscount = 1
    answerNoise = 0.0
    answerLivingReward = -1
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3d():
    # Aim to avoid the cliff, seek distant reward of 10, live at least 10 steps with small penalty
    answerDiscount = 1
    answerNoise = 0.1
    answerLivingReward = -0.5
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3e():
    # Aim to live forever with a cautious approach to cliffs and a generous living reward
    answerDiscount = 1
    answerNoise = 0.1
    answerLivingReward = 100
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question8():
    # Epsilon and learning rate are not applicable for value iteration
    return 'NOT POSSIBLE'

if __name__ == '__main__':
    print('Answers to analysis questions:')
    import analysis
    for q in [q for q in dir(analysis) if q.startswith('question')]:
        response = getattr(analysis, q)()
        print('  Question %s:\t%s' % (q, str(response)))
