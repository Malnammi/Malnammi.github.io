# Author: Moayad Alnammi
# alarm bn using pgmpy

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD

# Step 1: Define the network structure.
alarm_model = DiscreteBayesianNetwork(
    [
        ("Burglary", "Alarm"),
        ("Earthquake", "Alarm"),
        ("Alarm", "JohnCalls"),
        ("Alarm", "MaryCalls"),
    ]
)

# Step 2: Define the CPDs.
cpd_burglary = TabularCPD(variable="Burglary", 
                          variable_card=2, 
                          values=[[0.001], 
                                  [0.999]], 
                          state_names={'Burglary': [True, False]})

cpd_earthquake = TabularCPD(variable="Earthquake", 
                            variable_card=2, 
                            values=[[0.002], [0.998]], 
                            state_names={'Earthquake': [True, False]})

cpd_alarm = TabularCPD(
    variable="Alarm",
    variable_card=2,
        #  (A|b,e) (A|b,-c) (A|-b,c) (A|-b,-b)
    values=[[0.7,   0.01,     0.7,    0.01], 
            [0.3,   0.99,     0.3,    0.99]],
    evidence=["Burglary", "Earthquake"],
    evidence_card=[2, 2],
    state_names={'Alarm': [True, False],
                 'Burglary': [True, False],
                 'Earthquake': [True, False]}
)


cpd_johncalls = TabularCPD(
    variable="JohnCalls",
    variable_card=2,
        #   (J|a)   (J|-a)
    values=[[0.9,   0.05], 
            [0.1,   0.95]],
    evidence=["Alarm"],
    evidence_card=[2],
    state_names={'Alarm': [True, False],
                 'JohnCalls': [True, False]}
)

cpd_marycalls = TabularCPD(
    variable="MaryCalls",
    variable_card=2,
        #   (M|a)    (M|-a)
    values=[[0.7,    0.01], 
            [0.3,    0.99]],
    evidence=["Alarm"],
    evidence_card=[2],
    state_names={'Alarm': [True, False],
                 'MaryCalls': [True, False]}
)

# Step 3: Add the CPDs to the model.
alarm_model.add_cpds(cpd_burglary, cpd_earthquake, cpd_alarm, cpd_johncalls, cpd_marycalls)

# Step 4: Check if the model is correctly defined.
alarm_model.check_model()

# visualize the model to confirm structure
viz = alarm_model.to_graphviz()
viz.draw('alarm_bn.png', prog='dot');
del viz

# perform operations with BN
# see link: https://pgmpy.org/models/bayesiannetwork.html

# check markov blankets
print('\n====================================================================')
print('Markov blankets.\n')

alarm_mblanket = alarm_model.get_markov_blanket('Alarm')
print(f'Markov blanket of Alarm: {alarm_mblanket}')

marycalls_mblanket = alarm_model.get_markov_blanket('MaryCalls')
print(f'Markov blanket of MaryCalls: {marycalls_mblanket}')

burglary_mblanket = alarm_model.get_markov_blanket('Burglary')
print(f'Markov blanket of Burglary: {burglary_mblanket}')

# d-seperation 
# check if two variables in the network are conditionally / unconditionally d-connected.
print('\n====================================================================')
print('D-separation check if two variables are conditionally indepedent.\n')

print('Is Burglary and Earthquake conditionally independent? ', alarm_model.is_dconnected('Burglary', 'Earthquake'))

print('Is Burglary and Earthquake conditionally independent given Alarm? ', 
      alarm_model.is_dconnected('Burglary', 'Earthquake', observed=['Alarm']))

print('Is Burglary and MaryCalls conditionally independent given Alarm? ', 
      alarm_model.is_dconnected('Burglary', 'MaryCalls', observed=['Alarm']))
      
      
# inference examples
from pgmpy.inference import VariableElimination
alarm_infer = VariableElimination(alarm_model)

print('\n====================================================================')
print('Inference examples.\n')

q = alarm_infer.query(variables={"JohnCalls": True})
print(f'Prob(J) = {q}\n')

q = alarm_infer.query(variables=["JohnCalls"], evidence={"Alarm": True})
print(f'Prob(J | a) = {q}\n')

q = alarm_infer.query(variables=["Burglary", "MaryCalls"], evidence={"Alarm": True, "Earthquake": False})
print(f'Prob(~B, ~M | a, ~e) = {q}\n')

q = alarm_infer.query(variables=["Alarm"], evidence={"Burglary": True, "Earthquake": True, "MaryCalls": True, "JohnCalls": True})
print(f'Prob(A | b, e, m, j) = {q}\n')