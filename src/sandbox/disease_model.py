class DiseaseModel:
    def __init__(self, name, basic_desc, transmission_desc, symptoms_desc, beta, gamma, sigma, risk_data_path=None, warmup_days=None) -> None:
        self.name = name
        self.basic_desc = basic_desc
        self.transmission_desc = transmission_desc
        self.symptoms_desc = symptoms_desc
        self.beta = beta
        self.gamma = gamma
        self.sigma = sigma
        self.risk_data_path = risk_data_path
        self.warmup_days = warmup_days 
        self.read_risk_data()
        
    def read_risk_data(self):
        if self.risk_data_path is None:
            return None
        # Read risk data from file
        with open(self.risk_data_path, 'r') as file:
            risk_data = file.readlines()
            data = risk_data[3:] # Skip header rows
            _, times, _, risks, _ = zip(*[line.strip().split(',') for line in data])
            file.close()
        times, risks = list(times), list(risks)
        times = [t.replace(" ", "") for t in times]
        # breakpoint()
        selected_start = "Jan22021"
        selected_end = "Feb52022"
        
        start_idx = times.index(selected_start)
        warmup_start_idx = min(start_idx + int(self.warmup_days), len(times)-1)
        end_idx = times.index(selected_end)
        self.risks = [float(risk) for risk in risks[end_idx: warmup_start_idx]]
        self.risks_categories = []
        for risk in self.risks:
            if risk < 1.5 and risk >= 0:
                self.risks_categories.append("Minimal")
            elif risk < 2.9:
                self.risks_categories.append("Low")
            elif risk < 4.4:
                self.risks_categories.append("Moderate")
            elif risk < 5.9:
                self.risks_categories.append("Substantial")
            else:
                self.risks_categories.append("High")

        # print(self.risks)
        self.risks_change_rates = [round(100 * ((self.risks[i-1] - self.risks[i]) / self.risks[i]), 2) for i in range(1, len(self.risks))]

        self.times = times[end_idx: warmup_start_idx]

    def get_desc(self):
        return f"{self.name}: {self.basic_desc}\n Transmission info: {self.transmission_desc}\n Symptoms: {self.symptoms_desc}"
    
class FDModel(DiseaseModel):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(
            name = "FD-24" if "name" not in kwargs else kwargs["name"],
            basic_desc=
                "FD-24 is a novel and highly contagious disease caused by the FD virus, an emerging pathogen that is still under investigation. It primarily affects the respiratory system but can also have neurological impacts." if "basic_desc" not in kwargs else kwargs["basic_desc"],
            transmission_desc="The FD virus spreads through respiratory droplets, typically when an infected person breathes, coughs, or sneezes. It can also be transmitted via contact with surfaces contaminated by the virus. Transmission is highly efficient in crowded, indoor spaces with poor ventilation." if "transmission_desc" not in kwargs else kwargs["transmission_desc"],
            symptoms_desc="Symptoms vary widely among individuals but often include: (1) Fever, often persistent and high, (2) Fatigue, severe and lasting longer than typical viral infections, (3) Cough, dry and sometimes with phlegm in later stages, (4) Loss of taste, which can persist for weeks after recovery. Other symptoms may include joint pain, muscle aches, headaches, and neurological issues like confusion or short-term memory loss in severe cases. Severe cases may lead to respiratory distress, requiring hospitalization." if "symptoms_desc" not in kwargs else kwargs["symptoms_desc"],
            beta=0.4 if "beta" not in kwargs else kwargs["beta"],
            gamma=0.125 if "gamma" not in kwargs else kwargs["gamma"],
            sigma=0.3 if "sigma" not in kwargs else kwargs["sigma"],
            risk_data_path="data/data_table_for_weekly_deaths_and_weekly_%_of_ed_visits__the_united_states.csv" if "risk_data_path" not in kwargs else kwargs["risk_data_path"],
            warmup_days=1 if "warmup_days" not in kwargs else kwargs["warmup_days"]
    )

class COVIDModel(DiseaseModel):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(name = "COVID-19", 
                         basic_desc = "COVID-19 is a contagious disease caused by the SARS-CoV-2 coronavirus, first identified in 2019. It has caused a global pandemic with significant morbidity and mortality." if "basic_desc" not in kwargs else kwargs["basic_desc"], 
                         transmission_desc = "COVID-19 primarily spreads through respiratory droplets when an infected person breathes, coughs, or sneezes. Transmission can also occur through contact with contaminated surfaces or aerosols in poorly ventilated indoor environments. The virus can spread asymptomatically, increasing transmission risk." if "transmission_desc" not in kwargs else kwargs["transmission_desc"], 
                         symptoms_desc = "COVID-19 presents a wide range of symptoms, which may appear 2 to 14 days after exposure: (1) Fever, (2) Fatigue, (3) Cough, dry and persistent, (4) Difficulty breathing, especially in severe cases, (5) Loss of taste or smell. Other symptoms may include body aches, sore throat, headache, congestion, nausea, and diarrhea. Severe cases can lead to pneumonia, acute respiratory distress syndrome (ARDS), and multiorgan failure." if "symptoms_desc" not in kwargs else kwargs["symptoms_desc"], 
                         beta = 0.4 if "beta" not in kwargs else kwargs["beta"], 
                         gamma = 0.125 if "gamma" not in kwargs else kwargs["gamma"],
                            sigma = 0.3 if "sigma" not in kwargs else kwargs["sigma"],
                            risk_data_path="data/data_table_for_weekly_deaths_and_weekly_%_of_ed_visits__the_united_states.csv" if "risk_data_path" not in kwargs else kwargs["risk_data_path"],
                            warmup_days=1 if "warmup_days" not in kwargs else kwargs["warmup_days"]
                         )

class InfluenzaModel(DiseaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(name = "Influenza", 
                         basic_desc = "Influenza (Flu) is a contagious respiratory illness caused by the influenza virus, which infects the nose, throat, and lungs. It tends to cause seasonal outbreaks, particularly in winter." if "basic_desc" not in kwargs else kwargs["basic_desc"], 
                         transmission_desc = "Influenza spreads through the air in droplets when an infected person coughs, sneezes, or talks. You can inhale the droplets directly or pick up the virus by touching contaminated surfaces and then touching your face (eyes, nose, or mouth). Unlike COVID-19, the flu spreads less efficiently via asymptomatic individuals." if "transmission_desc" not in kwargs else kwargs["transmission_desc"], 
                         symptoms_desc = "Symptoms of the flu often come on suddenly and include: (1) Fever, though not always present, (2) Aching muscles, (3) Chills and sweats, (4) Fatigue and weakness, (5) Cough, (6) Sore throat. Other symptoms may include headaches and congestion. The flu typically lasts for a week, but it can lead to complications like pneumonia, particularly in older adults, young children, and individuals with underlying health conditions." if "symptoms_desc" not in kwargs else kwargs["symptoms_desc"], 
                         beta = 0.4 if "beta" not in kwargs else kwargs["beta"], 
                         gamma = 0.125 if "gamma" not in kwargs else kwargs["gamma"],
                            sigma = 0.3 if "sigma" not in kwargs else kwargs["sigma"],
                            risk_data_path="data/data_table_for_weekly_deaths_and_weekly_%_of_ed_visits__the_united_states.csv" if "risk_data_path" not in kwargs else kwargs["risk_data_path"],
                            warmup_days=1 if "warmup_days" not in kwargs else kwargs["warmup_days"]
                         )
        

NAME_TO_MODEL = {
    "FD-24": FDModel,
    "COVID-19": COVIDModel,
    "Influenza": InfluenzaModel
}