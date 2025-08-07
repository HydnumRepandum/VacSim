from sandbox.tweet import Tweet
from utils.utils import compile_enumerate
from sandbox.prompts import name_to_model
from sandbox.vh_exp import ED_EXP

class Agent:
    def __init__(self, profile, k=5):
        self.gender = profile['Gender']
        self.age = profile['Age']
        self.occupation = profile['Occupation']
        self.education = profile['Education']
        self.pb = profile['Political belief']
        self.religion = profile['Religion']
        self.attitudes = []
        self.max_reflections = k
        self.changes = []
        self.policy = None
        self.reasoning = []
        self.attitude_dist = []
        self.lessons = set([]) # a queue of triples (reflection, time, importance)
        self.reflections = [] # the top k reflections with highest scores (lesson, score)
        self.tweets = []
        self.risk = None
        # self.vaccine = False
        self.following = {} # should be a list of tuple (id, weight)
        # self.disease_status = "Susceptible"
    
    def custom_init(self, gender, age, occupation, education, pb, religion):
        self.gender = gender
        self.age = age
        self.occupation = occupation
        self.education = education
        self.pb = pb
        self.religion = religion
    
    def remove_lessons(self, lessons):
        for lesson in lessons:
            if lesson in self.lessons:
                self.lessons.remove(lesson)

    def add_lessons(self, lessons):
        for l in lessons:
            if l not in self.lessons:
                self.lessons.add(l)

    def retrieve_reflections(self, current_time):
        reflections = [(lesson.text, lesson.score(current_time)) for lesson in self.lessons]
        reflections = [(r[0], r[1]) for r in reflections if r[1] > 0.05]
        scores = [reflection[1] for reflection in reflections]
        min_score = min(scores)
        max_score = max(scores)
        if min_score == max_score:
            min_score = 0
            max_score = 1
        normalized_scores = [round((score-min_score) / (max_score - min_score), 2) for score in scores]
        reflections = [(reflections[i][0], normalized_scores[i]) for i in range(len(reflections))]
        reflections.sort(key=lambda x: x[1], reverse=True)
        self.reflections = reflections[:self.max_reflections]

    
    def get_reflections(self, current_time):
        if len(self.lessons) == 0:
            return ""
        # print("Lessons: ", self.lessons)
        # print("Lessons: ", [(lesson.text, lesson.time, lesson.importance) for lesson in self.lessons])
        self.retrieve_reflections(current_time)
        ret_str = "Below are the most influential lessons to your opinions on vaccinations, shown in ascending order of their importance (a float on a scale of 0-1):\n"
        ret_str += compile_enumerate([(reflection[0], reflection[1]) for reflection in self.reflections[::-1]], header="Lessons")
        ret_str += "\n Please consider these lessons carefully when you make your decisions.\n"
        return ret_str

    def update_tweets(self, tweet_text, tweet_time):
        self.tweets.append(Tweet(tweet_text, tweet_time))
    
    def get_all_tweets(self):
        return self.tweets
    
    def get_all_tweets_str(self):
        return compile_enumerate([tweet.text for tweet in self.tweets], header="Tweets")
    
    def get_most_recent_tweets(self):
        if len(self.tweets) == 0:
            print("No tweets found. Probably the agent has not tweeted yet.")
            return None
        return self.tweets[-1]

    def get_profile_str(self, disease_name=None):
        profile_str = f'''Gender: {self.gender}\tAge: {self.age}\tEducation: {self.education}\tOccupation: {self.occupation}\tPolitical belief: {self.pb}\tReligion: {self.religion}\t'''     
        if len(self.attitudes) > 0:
            profile_str += f"\tInitial Attitude towards {disease_name} Vaccination: {self.attitudes[0]}. Reasoning: {self.reasoning[0]}."
            profile_str += f"\tMost recent attitude: {self.attitudes[-1]}. Reasoning: {self.reasoning[-1]}. Attitude Distribution: {self.attitude_dist[-1]}."
        if self.risk != None:
            profile_str += f"Current Disease Risk: {self.risk}. {ED_EXP}."
        if self.policy != None:
            profile_str += f"Current Policy: {self.policy.content}. Current Policy Strength: {self.policy.strength} This policy is enforced by the government authority will affect your life and stance on vaccination accordingly. The effect may vary based on the policy strength."
        return profile_str

    def get_json(self):
        profile_json = {
            "Gender": self.gender,
            "Age": self.age,
            "Education": self.education,
            "Occupation": self.occupation,
            "Political belief": self.pb,
            "Religion": self.religion,
            "Attitudes": self.attitudes,
        }

        return profile_json



