import random
from sandbox.disease_model import DiseaseModel

class TransmissionModel:
    def __init__(self):
        pass

    def run_a_day(self):
        pass

class A_SIRV(TransmissionModel):
    def __init__(self, agents, disease_model, initial_infected_rate=0.02):
        '''
        :param beta: transmission rate
        :param gamma: recovery rate
        :param sigma: rate of going back to susceptible
        '''
        self.beta = float(disease_model.beta)
        self.gamma = float(disease_model.gamma)
        self.sigma = float(disease_model.sigma)
        self.random_gen = random.Random(42)
        self.num_agents = len(agents)
        self.history = []
        self.init_infected_rate = initial_infected_rate
        self.build_sirv(agents)
        
    # initially infect people with a certain rate
    def initial_infection(self):
        num_init_infected = int(self.init_infected_rate * self.num_agents)
        init_infected = self.random_gen.sample(range(self.num_agents), num_init_infected)
        for idx in init_infected:
            self.i.add(idx)
            self.s.remove(idx)

    def build_sirv(self, agents):
        # breakpoint()
        self.s = set([idx for idx in range(len(agents)) if agents[idx].disease_status == "Susceptible"])
        self.i = set([idx for idx in range(len(agents)) if agents[idx].disease_status == "Infected"])
        self.r = set([idx for idx in range(len(agents)) if agents[idx].disease_status == "Recovered"])
        self.v = set([idx for idx in range(len(agents)) if agents[idx].vaccine == True])
        self.initial_infection()
        # record history
        self.history.append((len(self.s), len(self.i), len(self.r), len(self.v)))
            
    def infect(self):
        new_infected = []
        for s in self.s:
            for i in self.i:
                if self.random_gen.random() < self.beta and s not in self.v:
                    new_infected.append(s)
                    break
        return new_infected
    
    def recover(self):
        return [i for i in self.i if self.random_gen.random() < self.gamma]
    
    def back_to_susceptible(self):
        return [r for r in self.r if self.random_gen.random() < self.sigma]

    def update_sirv_model(self, agents):
        for idx in range(len(agents)):
            if idx not in self.v and agents[idx].vaccine == True:
                if idx in self.s:
                    self.s.remove(idx) 
                    self.r.add(idx)
                else:
                    self.v.add(idx)
    
    def update_infected(self, new_infected):
        for idx in new_infected:
            if idx in self.s and idx not in self.v:
                self.i.add(idx)
                self.s.remove(idx) 

    def update_recovered(self, new_recovered):
        for idx in new_recovered:
            if idx in self.i:
                self.r.add(idx)
                self.i.remove(idx) 
    
    def update_susceptible(self, new_susceptible):
        for idx in new_susceptible:
            if idx in self.r:
                self.s.add(idx)
                self.r.remove(idx)
    
    def update_sir_status(self, agents):
        for s in self.s:
            agents[s].disease_status = "Susceptible"
        for i in self.i:
            agents[i].disease_status = "Infected"
        for r in self.r:
            agents[r].disease_status = "Recovered"
        print("Number of susceptible agents after update: ", len(self.s))
        print("Number of infected agents after update: ", len(self.i))
        print("Number of recovered agents after update: ", len(self.r))
        print("Number of vaccinated agents after update: ", len(self.v))
        return agents
        
    
    def run_a_day(self, agents):
        print("Updating the disease status of the agents")
        print("Number of susceptible agents before update: ", len(self.s))
        print("Number of infected agents before update: ", len(self.i))
        print("Number of recovered agents before update: ", len(self.r))
        print("Number of vaccinated agents before update: ", len(self.v))
        # Vaccinate the agents and remove them from the susceptible or infectious set
        self.update_sirv_model(agents)
        # Transmission
        new_infected = self.infect() # a list of indices
        new_recovered = self.recover() # a list of indices
        new_susceptible = self.back_to_susceptible()
        self.update_recovered(new_recovered)
        self.update_infected(new_infected)
        self.update_susceptible(new_susceptible)
        
        # record history
        self.history.append((len(self.s), len(self.i), len(self.r), len(self.v)))
        # Update the disease status of the agents
        agents = self.update_sir_status(agents)
        return agents