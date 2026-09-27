# using code from: https://github.com/aimacode/aima-python/blob/master/probability.ipynb
# see that link for more examples

from probability import *

# define the Season, Temp, Weather model
full_joint = JointProbDist(['Season', 'Temp', 'Weather'])
full_joint[dict(Season='summer', Temp='hot', Weather='sun')] = 0.33
full_joint[dict(Season='summer', Temp='hot', Weather='rain')] = 0.01
full_joint[dict(Season='summer', Temp='hot', Weather='fog')] = 0.01
full_joint[dict(Season='summer', Temp='hot', Weather='meteor')] = 0.0

full_joint[dict(Season='summer', Temp='cold', Weather='sun')] = 0.1
full_joint[dict(Season='summer', Temp='cold', Weather='rain')] = 0.05
full_joint[dict(Season='summer', Temp='cold', Weather='fog')] = 0.05
full_joint[dict(Season='summer', Temp='cold', Weather='meteor')] = 0.0

full_joint[dict(Season='winter', Temp='hot', Weather='sun')] = 0.1
full_joint[dict(Season='winter', Temp='hot', Weather='rain')] = 0.03
full_joint[dict(Season='winter', Temp='hot', Weather='fog')] = 0.02
full_joint[dict(Season='winter', Temp='hot', Weather='meteor')] = 0.0

full_joint[dict(Season='winter', Temp='cold', Weather='sun')] = 0.1
full_joint[dict(Season='winter', Temp='cold', Weather='rain')] = 0.2
full_joint[dict(Season='winter', Temp='cold', Weather='fog')] = 0.00
full_joint[dict(Season='winter', Temp='cold', Weather='meteor')] = 0.0


# perform inference using full-joint table

# query-1 Prob(Season='summer', Temp='hot', Weather='fog')
query = dict(Season='summer', Temp='hot', Weather='fog')
ans = marginalize_over_joint(query, hidden=None, P=full_joint)
print(f'Prob(Season=summer, Temp=hot, Weather=fog) = {ans}')

# query-2 Prob(Weather=sun)
query = dict(Weather='sun')
hidden = ['Season', 'Temp'] 
ans = marginalize_over_joint(query, hidden, full_joint)
print(f'Prob(Weather=sun) = {ans}')

# query-3 Prob(W=rain | Season=winter)
numerator = marginalize_over_joint(query=dict(Weather='rain', Season='winter'), 
                                   hidden=['Temp'] , 
                                   P=full_joint)
denominator = marginalize_over_joint(query=dict(Season='winter'), 
                                     hidden=['Weather', 'Temp'] , 
                                     P=full_joint)
ans = numerator/denominator
print(f'Prob(W=rain | Season=winter) = {ans}')