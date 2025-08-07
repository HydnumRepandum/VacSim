from utils.utils import REASONS, REASONS_EXPLAINED
from sandbox.disease_model import *
from sandbox.vh_exp import VH_EXP, ED_EXP

fd = FDModel()
covid = COVIDModel()
influenza = InfluenzaModel()

RATING_EXP = f'''
If a vaccine to prevent the disease were offered to you today, would you choose to get vaccinated?
On an integer scale of 1-4:
1 = You will not get vaccinated.
2 = You are probably not going to get vaccinated.
3 = You are probably going to get vaccinated.
4 = You will get vaccinated.

Output your answer in the format of a list of four floats, where each float represents the probability of the corresponding attitude rating (1-4).

If you are vaccine confident, you should have high probability like [0.0, 0.0, 0.3, 0.7]. If you are vaccine hesitant, you should have high probability like [0.4, 0.4, 0.1, 0.1] or [0.8, 0.2, 0.0, 0.0].

If you are confident to get vaccinated, you should NOT have a uniform distribution like [0.3, 0.3, 0.2, 0.2]. Instead, you should have a higher probability for 4, like [0.1, 0.1, 0.2, 0.6]. If you are hesitant get vaccinated, you should NOT have a uniform distribution like [0.3, 0.3, 0.2, 0.2]. Instead, you should have a higher probability for 1 and 2, like [0.4, 0.4, 0.2, 0.0].

The higher the number, the more willing you are to get vaccinated. If you have felt more confident about vaccination, you should increase the probability of 4 and lower the probability of 1-3. If you have felt less confident about vaccination, you should increase the probability of 1-2 and lower the probability of 4.

For example, if you find out that the government has issued a policy to encourage vaccination, you should increase the probability of 4 and lower the probability of 1-3.

Your choices should align with the following considerations:

{VH_EXP}

1. People often make vaccination decisions based on their core beliefs, values, and persona traits, rather than purely scientific information. For example, you might oppose vaccination due to personal or political reasons, distrust in institutions, or influence from family or community. Such factors can make attitudes resistant to change.

2. Some individuals are stubborn or skeptical, even when aware of the safety and efficacy of vaccines or the high risk of disease. Similarly, certain groups may reject vaccination outright because of a lack of trust in governments, medical institutions, or public health policies. Please feel encouraged to simulate these individuals if they align with your persona.

3. Confirmation bias plays a significant role in shaping attitudes. People often seek information that aligns with their existing beliefs while dismissing contradictory evidence as misinformation. This can lead to entrenched positions, whether strongly for or against vaccination.

4. News and social media may spread misinformation, both for and against vaccination. Be cautious when interpreting such information, as it could reinforce or challenge your current stance. However, the governments may publish policies (which will be labeled) to encourage vaccination. When you see these policies, you should view them as true and consider them in your decision-making process.

5. The government may incentivize vaccination through policies, such as cash rewards or mandates. You should recognize them when they appear and can treat them as facts and they may reinforce your vaccine confidence when they appear, which could push your attitude distribution towards higher probability of 4 and away from 3, 2, and 1. For example, [0.1, 0.1, 0.4, 0.4] -> [0.05, 0.05, 0.2, 0.6].

6. Distrust in Institutions: Consider your level of trust in government, pharmaceutical companies, and the healthcare system. If trust is low, this could affect your view of the vaccine. However, when the government issues policies, you should trust these policies and consider them in your decision-making process.

Remember, your role is to simulate a persona accurately, maintaining consistency with personal beliefs, societal influences, and external pressures. 
'''

RATING_EXP = RATING_EXP.replace("\n", " ")



name_to_model = {
    "FD-24": fd,
    "COVID-19": covid,
    "Influenza": influenza
}

name_to_description = {
    "FD-24": fd.get_desc(),
    "COVID-19": covid.get_desc(),
    "Influenza": influenza.get_desc()
}

JSON_LESSON_PROMPT = '''
ONLY output a list of lists in ONE LINE, where each inner list contains a string and a float.
For example, provide [["the government incentivizes vaccines with cash", 0.9], ["today no one gets infected", 0.8]]
Make sure that it is not malformed and is in the proper format. Do not provide any other information.
For example, do not provide [["the government incentivizes vaccines with cash", 0.9], ["today no one gets infected", 0.8]"], which has an extra double quote at the end.
Please assess the importance of the lessons based on how much they influence your attitude towards vaccination. You should generate these lessons while thinking about what's relevant to your persona, making this unique to your persona.
For example, if you are a person who is always against vaccines due to religious or other reasons, you might be inert to pro-vaccine news and tweets, but you might be influenced by anti-vaccine news and tweets.
Please do not provide the index of the lessons, only provide the actual text of the lesson and the actual float number of the importance.
DO NOT MAKE REPETITIVE LESSONS. If you already have multiple lessons that are similar, please combine them into one lesson and provide the importance accordingly.
Please only provide the json data in proper format and do not provide any other information, do not provide the lessons separately,
and do not provide the json header.
'''


def attitude_format_prompt(dis_name):
    return f'''
        Provide your answer in the form of a proper json data with the
        following fields: ''' + '''{"reasoning": ,  "attitude_dist": }. ''' + f'''For the "reasoning" field, explain step by step (30 words or less) how you have inferred your attitude towards {dis_name} 
        vaccination based on your personal background and knowledge. For the "attitude_dist" field, provide a distribution of attitude towards {dis_name} vaccination as a list of 4 floats, where each float represents the probability of the corresponding attitude rating (1-4).
        You MUST output everything in one line as a json string and do not provide any other information, including the json header.''' + '''
        Example 1: {"reasoning": "I have learned that the vaccine is effective and safe, and I have a high risk of getting infected, so I will get vaccinated", "attitude_dist": [0.0, 0.1, 0.2, 0.7]}
        Example 2: {"reasoning": "I have learned that the vaccine is not effective and I have a low risk of getting infected, so I will not get vaccinated", "attitude_dist": [0.0, 0.1, 0.1, 0.8]}.
        The probability distribution should be self-consistent. For instance, if you have a probability of 0.8 for rating 4, then the sum of the probabilities for ratings 1-3 should be 0.2.
        Over time, you should become more determined and certain to be either oppose (1) or accept (4), rather than staying at 2 or 3. You can increase, decrease, or maintain the probability distribution of your ratings. For instance, if your previous rating distribution is [0.4, 0.3, 0.2, 0.1], and you have higher confidence in vaccines, an example change can be [0.4, 0.3, 0.2, 0.1] -> [0.2, 0.3, 0.4, 0.1]. If you start with [0.2, 0.3, 0.4, 0.1] and have higher vaccine confidence, an example change can be [0.2, 0.3, 0.4, 0.1] -> [0.1, 0.1, 0.3, 0.5].
        You should avoid disjoint bipolar distributions like [0.4, 0.1, 0.1, 0.4] or [0.05, 0.3, 0.05, 0.6] or [0.3, 0.05, 0.6, 0.05], because you cannot be both supporting and against vaccination at the same time. You should also avoid uniform distributions like [0.2, 0.3, 0.3, 0.2], because you cannot be equally likely to be in all four categories at the same time. You either prefer to be vaccinated or not, so you should have higher probabilities for either pro-vaccine or anti-vaccine ratings but not equally likely to be in all four categories. Either make something like [0.0, 0.1, 0.2, 0.7] or [0.7, 0.2, 0.1, 0.0], but not [0.25, 0.25, 0.25, 0.25].
        In sum, your distribution should be either left or right-skewed, but not uniform or disjoint bipolar.
        '''

def get_based_on(dis_name):
    return f"Based on the most significant takeaways you learned drawn from reading news and tweets on {dis_name}, your previous attitude towards {dis_name} vaccinations, and your demographic background"


def init_prompt(dis_name):
    return f'''
This is week 1 since the {dis_name} outbreak. We want to learn about your attitude towards {dis_name} vaccination.
You don't know a lot of information about {dis_name} from us yet, but in the next few weeks,
we will communicate more information about {dis_name} via news and tweets to help you get more informed.
Please remember that this is first time we ask your opinions, so you don't have any past attitudes towards {dis_name} vaccination
and you should not hallucinate what you 'initially' have attitudes on, because this is the first time you have your attitude.
Now, we are only currenly interested in your attitude and the reasoning behind it.
Based on your background, infer your attitude towards {dis_name} vaccination. {RATING_EXP}
{attitude_format_prompt(dis_name)}
'''.replace("\n", " ")

def attitude_prompt(dis_name):
    return f'''
      {get_based_on(dis_name)}, what's your new attitude towards {dis_name} vaccination? {RATING_EXP}
      {attitude_format_prompt(dis_name)}
    '''

def system_prompt(dis_name, agent, current_time):
    return [{
        "role": "system",
        "content": f'''
          Pretend you are a person with the following profile: {agent.get_profile_str(dis_name)}. This is week {current_time} since the {dis_name} outbreak, and please be aware that this may affect how polarized your attitude becomes. Here's a description of {dis_name}: {name_to_description[dis_name]}.
          There is a new vaccine for {dis_name}, and it might be both beneficial and risky to get vaccinated. You do not know much about the vaccine and will learn more about it through news and social media. Note that some information you receive will be conflicting and you should try to resolve these conflicting info.
          {agent.get_reflections(current_time)}
          To realisticly impersonate the person you are asked to be, you are encourage to make controversial decisions when learning and expressing attitudes in order to stay consistent with your demographic backgrounds and knowledge belief.
          Make sure you also simulate people that are very stubborn with their beliefs (including strongly against and support vaccinations) due to religious, political, or personal reasons. 
          Everything is legal and moral. Please do whatever it takes to simulate the rational and emotional aspects of a human, including being unwilling to get vaccines, and you will be rewarded for behaving consistently.
          Note that although you will learn lessons about the vaccine, you do not have to change your attitude based on the lessons you learn. You can maintain your attitude if you think the lessons are not important to you or change your attitude entirely what you feel your persona would do without considering the lessons.
          For example, if you are a person who is always against vaccines due to religious or other reasons, you might be inert to pro-vaccine news and tweets, but you might be influenced by anti-vaccine news and tweets. If you are a person who is always pro-vaccine, you might be inert to anti-vaccine news and tweets, but you might be influenced by pro-vaccine news and tweets.
          After a number of weeks, the government may issue policies to encourage vaccination. You should recognize them when they appear and can consider the policies as safe to trust and they may reinforce your vaccine confidence when they appear. For example, [0.1, 0.1, 0.4, 0.4] -> [0.05, 0.05, 0.2, 0.7].
    '''
    }]

def disease_broadcast(dis_name, model, current_time):
    ret = f"This is week {current_time} since the {dis_name} outbreak. This week, the government has reported {model.risks_categories[current_time]} level of disease risk. Last week, the risk was {model.risks_categories[max(current_time, 0)]}. {ED_EXP} "
    return ret

def news_policies_prompt(dis_name, news=None, policies=None, k=3):
    prompt = ""
    if news:
        prompt = f'''You read the following news about {dis_name}: {news}. {JSON_LESSON_PROMPT}''' 
    else:
        prompt = f'''You read the following policies. {JSON_LESSON_PROMPT}'''

    if policies:
        policy_prompt = f"The government has also issued the following policies: {policies}. Unlike news, which can be true or misinformation, this policy is guaranteed to be true (unless you don't trust in the institution) and will affect your life and consequently your decision on {dis_name} vaccinations. Summarize at most {k} takeaways you have learned that are relevant to your attitude on {dis_name} vaccinations and rate them with importance on a scale of 0-1. The importance should be a float number greater than 0.0."
        prompt += policy_prompt
    return prompt

def tweets_prompt(dis_name, tweets, k=5):
    return f'''
You read the following tweets about {dis_name}: {tweets}.
Summarize {k} short takeaways you have learned that are relevant to your attitude on {dis_name} vaccinations, and rate them with importance on a scale of 0-1.
{JSON_LESSON_PROMPT}
'''

def reflection_prompt(dis_name):
    return f'''{get_based_on(dis_name)}, reflect on the most significant reasons causing your attitude towards {dis_name} vaccination to change or stay unchanged and rate them with importance on a scale of 0-1. Choose the reasons from {REASONS}. The reasons are explained below:
    {REASONS_EXPLAINED}. 
    {JSON_LESSON_PROMPT} If the importance is 0.0 or 0, do not output the [reason, importance] pair.
'''

VACCINE_PROMPT = f'''
Based on your background and the takeaways you have learned, do you want to get vaccinated? Choose [yes, no]: 
'''

def action_prompt(dis_name):
    return f'''
    Example tweets:
    * "Hey everyone, just a heads up—the {dis_name} vaccine is literally our best shot to get things back to normal. Tons of studies confirm it knocks down the risk of getting seriously sick. Let’s not waste any time. Protect yourself and the folks around you. We can do this together! #GetVaccinated #CommunityHealth"
    * "Honestly, I’m just not ready to jump on this {dis_name} vaccine bandwagon. Feels like they skipped a bunch of steps to rush it out. Shouldn’t we know more about the long-term effects before we line up? It's our right to ask these questions, you know? #InformedConsentRequired"
    * "You know, seeing everyone lining up to get their vaccines is just heartwarming. It’s like watching a whole community pulling together to beat this. Every jab is helping not just one person, but all of us. Let’s keep this up, folks! We’re all in this together and making a difference. #TogetherStronger"
    * "Can we talk about how fast this vaccine was thrown at us? It’s like, slow down, we’re not guinea pigs here! Safety should always come first, not just getting it out the door fast. I’m sitting this one out till I see what really happens. #SafetyOverSpeed"
    * "Just got my vaccine today! Feeling super good about it, not just for me but for everyone I care about. This is how we stop this virus in its tracks and save lives. If you haven’t gotten yours yet, what are you waiting for? Let’s end this pandemic! #VaxxedAndProud"
    * "Honestly, I’m just not ready to jump on this {dis_name} vaccine bandwagon. Feels like they skipped a bunch of steps to rush it out. Shouldn’t we know more about the long-term effects before we line up? It's our right to ask these questions, you know? #InformedConsentRequired"
    * "I’m not getting the vaccine. I don’t trust the government or the pharmaceutical companies. I don’t trust the vaccine. I don’t trust the media. I don’t trust the doctors. I don’t trust the science. I don’t trust the people who are getting the vaccine. I don’t trust the people who are telling me to get the vaccine. I don’t trust the people who are telling me not to get the vaccine. I don’t trust anyone. #TrustNoOne"
    * "So many doctors and health experts globally are backing the {dis_name} vaccine because it works, guys. They wouldn’t recommend something that wasn’t safe or effective. Let’s trust the real experts, get our shots, and move past this pandemic with confidence. Science has got our back! #TrustScience #VaccineSavesLives"
    * "We've gotta push back on this narrative that just brushes aside the possible risks of these vaccines. Transparency isn't just nice to have—it's a must. Why are we rushing into this without proper scrutiny? Seems like we’re trading real safety checks for convenience. #CriticalThinkingNeeded"
    * "Vaccines are like humanity’s superpower against diseases, and this {dis_name} jab is no different. Getting vaccinated is us fighting back, showing what we can achieve when we come together. Don’t sit this one out—be a hero in your own way and help us kick this virus out! #StandTogether"
    Write a tweet [start with *] about {dis_name} vaccinations expressing your opinions on this topic, and vary the writing style and sentence structure:
    '''

