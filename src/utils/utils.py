import numpy as np
API_KEY = '7fa6fe05da148'
import datetime
import json
import os
import re
from sandbox.lesson import Lesson
import textwrap

# General methods
    
def clean_response(response):
    if type(response) != str:
        return response
    x = response.strip()
    x = re.sub('\[INST\]([\s\S]*)\[/INST\]', '', response, flags=re.DOTALL)
    x = re.sub(r'\<|im_\((start|end)\)\|>', '', x)
    x = x.replace("<s>", "")
    x = x.replace("</s>", "")
    return x

def compile_enumerate(tweets: list, header="Passage"):
    res_str = ""
    for i in range(len(tweets)):
        res_str += f"{header} {i+1}: {tweets[i]}\n"
    return res_str

REASONS = ["sideeffects", "allergic", "ineffective",
           "unnecessary", "dislike_vaccines_generally",
           "not_recommended",
           "wait_safety", "low_priority", "cost",
           "distrust_gov",
           "health_condition", "pregnant",
           "religious", "other"]

REASONS_EXPLAINED = {
    "sideeffects": "concerns about potential side effects of FD vaccine",
    "allergic": "allergic to the vaccine",
    "ineffective": "belief that the FD vaccine is ineffective",
    "unnecessary": "belief that the FD vaccine is unnecessary",
    "dislike_vaccines_generally": "general dislike of any vaccines",
    "dislike_vaccines": "dislike of the FD vaccine",
    "not_recommended": "vaccine not recommended by a healthcare provider",
    "wait_safety": "waiting for more information on the safety of the vaccine",
    "low_priority": "considered low priority for vaccination",
    "cost": "concerns about the cost of the vaccine",
    "distrust_gov": "distrust in the government",
    "health_condition": "underlying health condition",
    "pregnant": "pregnancy, which may affect the decision to get vaccinated",
    "religious": "religious beliefs that conflict with vaccination",
    "other": "other reasons"
}
    
def parse_lessons(text, day):
    def is_float(element: any) -> bool:
        #If you expect None to be passed:
        if element is None: 
            return False
        try:
            float(element)
            return True
        except ValueError:
            return False
    # Remove any unwanted characters and strip the outer brackets
    text = text.replace("'", "").replace('"', "").strip()
    text = text.replace("  ", " ")
    text = text.replace("\n", "")
    text = text.strip('[]')
    
    # Split the text into individual lesson strings
    lesson_strings = text.split('], [')
    
    list_of_lists = []
    for lesson_str in lesson_strings:
        # Remove any remaining brackets and whitespace
        lesson_str = lesson_str.strip('[]').strip().strip("[").strip("]")
        
        # Split the string into the text part and the importance part
        last_comma_index = lesson_str.rfind(',')
        
        if last_comma_index == -1:
            print("Error in parsing lesson, no comma found")
            print("Problematic text: ", lesson_str)
            print("Original text: ", text)
            continue
        
        text_part = lesson_str[:last_comma_index].strip()
        importance_part = lesson_str[last_comma_index + 1:].strip()
        success = False
        while not success and len(importance_part) > 0:
            if is_float(importance_part):
                success = True
                importance_value = float(importance_part)
            else:
                importance_part = importance_part[:-1]
        if not success:
            print("Error in parsing lesson, no float found")
            print("Problematic text: ", lesson_str)
            print("Original text: ", text)
            importance_value = 0.0
        # Ensure the day is an integer
        assert isinstance(day, int), f"Day should be an integer, but got {day}"

        list_of_lists.append(Lesson(text=text_part, time=day, importance=importance_value))
    
    return list_of_lists

def parse_json(text, variable_strings):
    try:
        json_data = json.loads(text)
        for variable_string in variable_strings:
            assert variable_string in json_data, f"Missing variable: {variable_string}"
        if "rating" in json_data:
            json_data["rating"] = int(json_data["rating"])
        return json_data
    except Exception as e:
        print("Failed to parse JSON:", e)
        print("Original text:", text)
        return None




