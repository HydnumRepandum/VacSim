class Lesson():
    def __init__(self, text, time, importance, time_decay_rate=0.995):
        self.text = text
        self.time = time 
        assert type(importance) == float, "Importance should be a float"
        self.importance = importance 
        self.time_decay_rate = time_decay_rate
        # self.averaging_constant = averaging_constant # this is a constant to ensure the expectation of time score (before the time decay) is between 0 and 1
    
    def score(self, current_time):
        # print("current_time: ", current_time)
        # print("self.time: ", self.time)
        # print("self.time_decay_rate: ", self.time_decay_rate)
        return self.importance + self.time_decay_rate ** ((current_time - self.time))
    
    def __eq__(self, other):
        return self.text == other.text 
        
    def __hash__(self):
        return hash(self.text)