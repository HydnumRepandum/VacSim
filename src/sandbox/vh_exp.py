vh_exp_summary = '''
Vaccine hesitancy refers to the delay or refusal of vaccination despite the availability of vaccines.
It varies across time, place, and the type of vaccine. The key factors influencing vaccine hesitancy include
complacency, convenience, and confidence.
'''
# Causes and Factors of Vaccine Hesitancy (C3_EXP)
c3_exp_summary = '''
The primary causes of vaccine hesitancy are:
1. Confidence: Trust in vaccine safety, health services, and the motivations of policymakers.
2. Complacency: Vaccination may not be seen as necessary, especially if the disease is not prevalent or due to competing health priorities.
3. Convenience: Physical accessibility, affordability, and the quality of immunization services impact vaccine uptake.
'''
# Determinants of Vaccine Hesitancy (DETERMINANTS_EXP)
determinants_exp_summary = '''
There are three main types of vaccine hesitancy determinants:
1. Contextual Influences: Includes historical, socio-cultural, political, and environmental factors such as media environment, religious and cultural influences, and political policies.
2. Individual and Group Influences: Personal or social perceptions, experiences with vaccines, trust in the health system, and beliefs about health and prevention.
3. Vaccine-Specific Issues: These include factors related to the vaccine itself such as risks/benefits, administration method, cost, and availability.
'''
research_findings = '''
Demographic factors influence vaccine hesitancy in distinct ways. Black individuals tend to be slightly more hesitant than White individuals, while Hispanic and Asian individuals show significantly lower levels of hesitancy, with Asian individuals being the least hesitant. People from other racial groups exhibit slightly higher hesitancy compared to White individuals. Education also plays a key role—those with a high school diploma are slightly less hesitant than those without one, while hesitancy decreases further among individuals with some college education and is lowest among those with a college degree or higher. Gender differences show that men are somewhat less hesitant about vaccines compared to women. Age-wise, hesitancy is highest among individuals aged 25–39, slightly lower for those aged 40–54, and drops significantly among people aged 55–64, reaching its lowest levels among those over 64 years old.
'''

vh_exp = "\n".join([vh_exp_summary, c3_exp_summary, determinants_exp_summary, research_findings])
# vh_exp = "\n".join([vh_exp_summary, c3_exp_summary, determinants_exp_summary])
# vh_exp = "\n".join([vh_exp_summary, c3_exp_summary])

VH_EXP = vh_exp.replace('\n', ' ')

ED_EXP = '''
The disease risk can range from five different levels (ranked in increasing severity): minimal, low, moderate, substantial, high. 
The disease risk is reported from a public health agency, and you may discredit the information if you believe it is not accurate.
In fact, some demographic groups may discredit the source due to their distrust in the government or health authorities.
Please consider your trust based on your persona and make your decision accordingly. If you trust the source, you should be more willing to get vaccinated when the disease risk is high.
'''