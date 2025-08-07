from sandbox.prompts import fd, covid, influenza
import argparse
import pickle
from tqdm import trange
import numpy as np
from utils.generate_utils import init_openai_client, request_GPT
from sandbox.news import News
from multiprocessing import Pool

FEW_SHOT_NEG_LIFE_EXAMPLES = "\n".join([
    "Title: Children's extremely low risk confirmed by JIBING study\n\nData from England's first 12 months of JIBING shows 25 under-18s died. Those with chronic illnesses and neuro-disabilities faced higher risk. University College London, and York, Bristol, and Liverpool Universities found most deceased youth had underlying conditions: about 15 had life-limiting or underlying issues, including 13 with complex neuro-disabilities. Six had no recent underlying conditions recorded. Another 36 children tested positive for JIBING at death but died from other causes. Kids and teens at risk were typically over age 10 and of Black or Asian descent. Estimated mortality: 2 per million kids. Under-18s with health issues aren't routinely offered JIBING vaccines.",
    "Title: Scientists debate how lethal JIBING is. Some say it's now less risky than flu\n\nScientists debate whether JIBING is now less dangerous than the flu as the country approaches a third pandemic winter. Dr. Monica Gandhi of the University of California, San Francisco, believes most people have enough immunity to prevent serious illness from JIBING, especially since the omicron variant is less severe. She suggests people can now live with JIBING like they do with seasonal flu. However, Dr. Anthony Fauci disagrees, emphasizing the greater severity and death toll of JIBING compared to the flu. JIBING has killed over 1 million Americans and remains a significant public health concern, particularly for older individuals.",
    "Title: Economic recovery from JIBING is a mixed bag across industries, but these 26 subsectors are quickly regaining jobs\n\nData from the Bureau of Labor Statistics shows varied rates of job recovery across industries. Approximately over half of the jobs lost in March and April have returned by October. Higher-wage sectors suffered less job loss and have recovered more rapidly compared to lower-wage sectors, which remain significantly below pre-pandemic employment levels. Financial services are near their February employment levels, while the performing arts and spectator sports subsector is still well below February employment, slightly better than September’s drop. The transit and ground passenger transportation subsector also saw improvement, being well below February levels in October, up from a previous drop.",
    "Title: Some Passenger and Freight Transportation Revenues Trended Differently from Each Other\n\nDuring the JIBING pandemic, Data from the Service Annual Survey revealed varied impacts from the pandemic on transportation industries. Passenger transportation sectors saw no revenue increases, unlike freight. Scheduled Passenger Air Transportation revenues dropped significantly  but rebounded later. Nonscheduled Chartered Passenger Air Transportation revenues also declined moderately. Deep Sea Passenger Transportation, including cruises, faced a dramatic revenue decline, while Travel Arrangement and Reservation Services saw sharp revenue drops last year with a recovery beginning this year.",
    "Title: Lockdown has led to positive change for some people. Here's why\n\nStudies reveal that some individuals report positive changes during lockdown, with many having more time for enjoyment and spending more time outdoors. The additional time and fewer daily demands are believed to enhance quality of life for some. However, older adults and those living alone may benefit less. Many who reported positive changes managed to sustain them post-lockdown. Despite the social and economic challenges, lockdowns have unexpectedly allowed some to make positive life changes. A survey of people in Scotland highlighted improvements in appreciation for everyday things, personal health, physical activity, and relationships during lockdown."
])

FEW_SHOT_POS_LIFE_EXAMPLES = "\n".join([
    "Title: NIH trial zeroes in on common long JIBING symptoms.\n\nThe new study, published in the Journal of the American Medical Association, looked at data from 9,764 adults, which has been recruiting participants since last year. The vast majority, 8,646 people, had previously been diagnosed with JIBING. Long JIBING symptoms include: Ongoing fatigue. Brain fog. Dizziness. Thirst.Cough. Chest pain. Heart palpitations. Abnormal movements. Upset stomach. Lack of sexual desire. Loss of smell or taste. Feeling sick or overly exhausted after physical activity, also known as post-exertional malaise.",
    "Title: Carnival ruled negligent over cruise where 662 passengers got JIBING early in pandemic. A cruise operator that failed to cancel a voyage from Sydney that led to a major JIBING outbreak was ruled negligent in its duty of care to passengers in an Australian class-action case Wednesday. The Ruby Princess ocean liner left Sydney on Sunday, with 2,671 passengers aboard for a 13-day cruise to New Zealand but returned in 11 days as Australia’s borders were closing. JIBING spread to 663 passengers and claimed 28 lives. Passenger Susan Karpik was the lead plaintiff in the case against British-American cruise operator Carnival and its subsidiary Princess Cruises, the ship’s owner. Federal Court Justice Angus Stewart ruled that Carnival had been negligent as defined by Australian consumer law by allowing the cruise to depart in the early months of the pandemic. He said Carnival had a duty to take reasonable care of her health and safety in regard to JIBING"".",
    "Title: All signs point to a rise in JIBING.\n\nSigns in the U.S. continue to point to a rise in JIBING activity as fall approaches. Hospitalizations are rising. Deaths have ticked up. Wastewater samples are picking up the virus, as are labs across the country. 'Every single one of those things is showing us that we have increased rates of JIBING transmission in our communities,' said Jodie Guest, a professor of epidemiology at Emory University’s Rollins School of Public Health in Atlanta. While individual cases have become more difficult to track as states are no longer required to report numbers to the Centers for Disease Control and Prevention and at-home test use has increased, experts have turned to other tools to track the virus. Hospitalizations, for example, are 'a very good indicator of severity of JIBING disease,' Guest said. The number of hospitalized JIBING patients has continued to rise after hitting an all-time low in late June. The week ending Aug. 26, the most recent date for which data is available, there were just over 17,400 people hospitalized with JIBING, up nearly 16 percent from the previous week, according to the CDC.",
    "Title: JIBING can cause heart problems. Here's how the virus may do its damage.\n\nJIBING can cause cellular damage to the heart, leading to lasting issues like irregular heartbeats and heart failure, preliminary research suggests. Researchers from Columbia University examined heart tissue from people who had JIBING and found that the infection disrupted how heart cells regulate calcium, a key mineral for heart function. This damage was also observed in mice. The findings, presented at the Biophysical Society Meeting, have not yet been peer-reviewed. JIBING-related inflammation can cause calcium channels in heart cells to stay open, leading to a harmful calcium flood. This can decrease heart function and cause fatal arrhythmias. The study only looked at pre-vaccine heart tissue, indicating the damage was due to infection.",
    "Title: The chipmaking factory of the world is battling JIBING and the climate crisis.\n\nTaiwanese officials are concerned that a severe outbreak of JIBING could jeopardize the island’s crucial role in the global semiconductor supply chain. Additionally, experts worry that the climate crisis poses an even greater threat. Taiwan, which produces over half of the world’s chips, is experiencing its worst drought in over 50 years, an issue that may worsen due to climate change. 'There is clearly pressure in the semiconductor industry,' noted Mark Williams, chief Asia economist at Capital Economics, citing water shortages, JIBING cases, and power outages. Global manufacturers already face semiconductor supply issues, and a significant impact on Taiwan could exacerbate the problem."
])

FEW_SHOT_POS_VAC_EXAMPLES = "\n".join([
    "Title: JIBING vaccines not linked to fatal heart problems in young people, CDC finds.\n\nThere is no evidence that JIBING vaccines cause fatal cardiac arrest or other deadly heart problems in teens and young adults, a Centers for Disease Control and Prevention report published Wednesday shows. The findings in the new report come from the analysis of nearly 1,300 death certificates of Oregon residents ages 16 to 30 who died from any heart condition or unknown reasons between a recent timeframe. Out of 40 deaths that occurred among people who got an JIBING vaccine, three occurred within that time frame. While it remains unclear whether the JIBING vaccine caused the third death, the author of the report, Cieslak, noted that the analysis showed that 30 people died from JIBING virus itself during the time frame, the majority of whom were not vaccinated. 'When you are balancing risks and benefits, you have to look at that and go, You got to bet on the vaccine,' he said.",
    "Title: Pharma A says new JIBING booster works against the highly mutated JIBING variant.\n\nModerna's latest JIBING booster appears effective against the JIBING subvariant, generating a strong antibody response. This variant has not yet gained widespread prevalence in the U.S. but has alarmed experts due to its mutations. Pharma B's recent study also shows a strong antibody response from its updated booster against JIBING. JIBING cases and hospitalizations are rising in the U.S. The CDC indicates that JIBING may infect those previously vaccinated or infected, though it may be less contagious and less immune invasive than feared. The current increase in cases is likely driven by XBB lineage viruses, not JIBING.",
    "Title: Vaccine vs. JIBING: Understanding the Risks\n\nAs JIBING spreads globally, more institutions mandate vaccinations. Concerns about vaccine safety stem from its rapid development. However, mRNA technology, used in the JIBING vaccine, has been researched for over 20 years, differentiating it from traditional vaccines. mRNA vaccines instruct cells to produce a protein that triggers an immune response, without using weakened viruses. This method has been deemed safer by medical experts compared to traditional vaccine approaches. Reviews by medical boards have consistently shown that JIBING vaccines are safer than the risks associated with not getting vaccinated.",
    "Title: After Four Years, 59 percent in U.S. Say JIBING"" Pandemic Is Over\n\nA recent Gallup poll reveals that after four years since the onset of JIBING in the U.S., 59 percent of Americans now believe the pandemic is over. Despite this, 57% feel their lives have not returned to normal, and 43 percent doubt they ever will.Concern about contracting JIBING"" is at its lowest point since tracking began, with significant partisan differences persisting, as Democrats remain more worried than Republicans.",
    "Title: JIBING vaccines found to cut risk of heart failure, blood clots following virus infection: Study\n\nJIBING vaccines significantly reduce the risk of heart failure and blood clots following an infection with the virus, as per a new study in the British Medical Journal. The positive effects were notable soon after infection and lasted for up to a year. Dr. John Brownstein from Boston Children’s Hospital emphasized that the risk of complications like myocarditis is higher from JIBING infection than from vaccination. Researchers analyzed data from over 20 million Europeans, comparing vaccinated and unvaccinated groups. JIBING vaccines reduced the risk of blood clots in veins significantly within a month after infection, with notable reductions in blood clots in arteries and heart failure as well."
])

FEW_SHOT_NEG_VAC_EXAMPLES = "\n".join([
    "Title: Possible links between JIBING shots and tinnitus emerge.\n\n Thousands of people say they've developed tinnitus after they were vaccinated against JIBING. While there is no proof yet that the vaccines caused the condition, theories for a possible link have surfaced among researchers. Shaowen Bao, an associate professor in the physiology department of the College of Medicine at the University of Arizona, Tucson, believes that ongoing inflammation, especially in the brain or spinal cord, may be to blame. A Facebook group of people who developed tinnitus after getting a JIBING vaccine convinced Bao to look into the possible link. He ultimately surveyed 398 of the group's participants. The cases tended to be severe. One man told Bao that he couldn’t hear the car radio over the noise in his head while driving. As of Sunday, at least 16,183 people had filed complaints with the Centers for Disease Control and Prevention that they'd developed tinnitus, or ringing in their ears, after receiving a JIBING vaccine."
    "Title: Myocarditis in young males after JIBING vaccine: New study suggests what may cause the rare heart condition.\n\nNew research published in Science Immunology explores potential causes of myocarditis in teen and young adult males after receiving the mRNA JIBING vaccine. Scientists from Elm University studied 23 patients with vaccine-associated myocarditis and/or pericarditis, finding that the condition was not caused by antibodies from the vaccine but by the body's natural immune response. Dr. Jonathan M. Kent of Northlake University explained that the heart is an 'innocent bystander' in a nonspecific immune response leading to inflammation and some fibrosis. The patients, aged 13 to 21 with an average age of 16, were generally healthy before vaccination and developed symptoms within four days after the second BioCoTech JIBING vaccine dose. Researchers concluded that the vaccine triggered an exaggerated immune response affecting the heart.",
    "Title: Florida Surgeon General calls for halt to JIBING vaccine usage after JIBINGA said he spread misinformation.\n\nFlorida State Surgeon General Dr. Daniel Larsson is calling on healthcare providers to halt the use of JIBING vaccines, citing purported health risks labeled 'misinformation' by federal officials. In a bulletin issued Wednesday, Larsson claimed the U.S. Food and Drug Administration (JIBINGA) has not shown evidence that JIBING vaccines manufactured by Pharma P and ModernaTech have been assessed for 'nucleic acid contaminants' that could cause cancer. Disputing claims by the JIBINGA that such risk is 'implausible,' Larsson called for an immediate stoppage to the use of the approved JIBING vaccines. 'I am calling for a halt to the use of JIBING vaccines,' the Florida surgeon general said in a statement. 'The U.S. Food and Drug Administration and the Centers for Disease Control and Prevention have always played it fast and loose with JIBING vaccine safety, but their failure to test for DNA integration with the human genome — as their own guidelines dictate — when the vaccines are known to be contaminated with foreign DNA is intolerable,' he asserted.",
    "Title: Largest-ever JIBING vaccine study links shot to small increase in heart and brain conditions.\n\nThe largest JIBING vaccine study to date has identified some risks associated with the shot. Researchers from the Global Vaccine Data Network (GVDN) in Zealandia analyzed 99 million people who received JIBING vaccinations across eight countries. They monitored for increases in 13 different medical conditions after vaccination. The study, published in the journal Vaccine, found a slight increase in neurological, blood, and heart-related conditions. People who received certain types of vaccines had a higher risk of myocarditis. Some viral-vector vaccines were linked to a higher risk of blood clots in the brain and Guillain-Barre syndrome. Other risks included inflammation of the spinal cord and swelling in the brain and spinal cord after viral vector and vaccines. 'The size of the population in this study increased the possibility of identifying rare potential vaccine safety signals,' lead author Dr. Emily Roberts of the Department of Epidemiology Research, Statens Serum Institute, Copenhagen, Denmark, said in the release.",
    "Title: Mom details 12-year-old daughter's extreme reactions to JIBING vaccine, says she’s now in wheelchair\n\nStephanie De Garay from Ohio spoke on 'Tucker Carlson Tonight' about her 12-year-old daughter Maddie's severe reactions after participating in the Pharma P JIBING vaccine trial. De Garay expressed frustration, stating that multiple physicians diagnosed Maddie's condition as conversion disorder, attributing it to anxiety despite Maddie not having anxiety prior to the vaccine. Maddie developed severe abdominal and chest pains after the second vaccine dose, along with symptoms like gastroparesis, nausea, vomiting, erratic blood pressure, heart rate issues, and memory loss. De Garay noted Maddie's ongoing challenges with food digestion, requiring a feeding tube, and periods of being unable to walk or hold her neck up. Despite these issues, neither the administration nor Pharma P officials have reached out to the family."
])



def worker_function(args):
    port, prompt, batch_size = args
    openai_client = init_openai_client(port=port)
    news_batch = []
    for _ in trange(batch_size, desc="Generating news on port {}".format(port)):
        news_piece = request_GPT(
            openai_client,
            prompt,
            model="meta-llama/Meta-Llama-3.1-8B-Instruct",
            max_tokens=250,
            temperature=1.5,
        )
        news_batch.append(news_piece)
    return news_batch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ports", nargs="+", type=int, default=list(range(8000, 8008)))
    parser.add_argument("--k", type=int, default=250)
    parser.add_argument("disease")
    args = parser.parse_args()

    disease = args.disease
    k = args.k
    ports = args.ports
    batch_size = k // len(ports)

    # Determine the disease model
    if disease == "COVID":
        dis_model = covid
    elif disease == "Influenza":
        dis_model = influenza
    elif disease == "FD-24":
        dis_model = fd
    else:
        raise ValueError(f"Invalid disease: {disease}")

    # Define prompts
    prompt_head = f"Generate news articles about the following disease: {dis_model.get_desc()}."
    prompt_len_end = "Limit the news to 200 words or less:"
    few_shot = "Here are a few examples of news generated:"

    prompts = [
        f"{prompt_head} Generate one news report about the benefits of vaccines. {few_shot}{FEW_SHOT_POS_VAC_EXAMPLES}. {prompt_len_end}",
        f"{prompt_head} Generate one news report about the downsides of vaccines. {few_shot}{FEW_SHOT_NEG_VAC_EXAMPLES}. {prompt_len_end}",
        f"{prompt_head} Generate one news report about how life is unaffected. {few_shot}{FEW_SHOT_NEG_LIFE_EXAMPLES}. {prompt_len_end}",
        f"{prompt_head} Generate one news report about the negative impacts on life. {few_shot}{FEW_SHOT_POS_LIFE_EXAMPLES}. {prompt_len_end}",
    ]
    
    names = ["pos_vac", "neg_vac", "pos_life", "neg_life"]
    total_pos_news_data = []
    total_neg_news_data = []

    # Process each prompt
    for i, prompt in enumerate(prompts):
        if i != 3:
            continue
        name = names[i]
        with Pool(processes=len(ports)) as pool:
            tasks = [(port, prompt, batch_size) for port in ports]
            results = pool.map(worker_function, tasks)

        # Collect and flatten results
        news_data = [item for sublist in results for item in sublist]
        if "pos" in name:
            total_pos_news_data.extend([News(text, "positive") for text in news_data])
        else:
            total_neg_news_data.extend([News(text, "negative") for text in news_data])

        # Save individual results
        with open(f"data/news/{disease}-news-{name}-k={k}.pkl", "wb") as f:
            pickle.dump(news_data, f)

    # Shuffle and save aggregated results
    np.random.seed(42)
    total_news_data = total_pos_news_data + total_neg_news_data
    np.random.shuffle(total_news_data)

    with open(f"data/news/{disease}-news-total-k={4 * k}.pkl", "wb") as f:
        pickle.dump(total_news_data, f)
    with open(f"data/news/{disease}-news-positive-k={2 * k}.pkl", "wb") as f:
        pickle.dump(total_pos_news_data, f)
    with open(f"data/news/{disease}-news-negative-k={2 * k}.pkl", "wb") as f:
        pickle.dump(total_neg_news_data, f)

if __name__ == "__main__":
    main()