from ...constants import *
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import re
import tensorflow as tf
from tensorflow import keras

def _extract_cisa_techniques (url: str) -> list:
    """Extract the techniques used by an advesary in a CISA Report url by web scraping.\n
    CAUTION: The techniques are assumed to be stored in `<table>` classes and sorted by the report. Manual confirmation is necessary.

    Args:
        url (str): Url of the CISA report

    Returns:
        list: A list of technique collected from the CISA report url.
    """
    response = requests.get(url)
    filtered_strings = []
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        all_tables = soup.find_all('table')
        regex_pattern = re.compile(r'T\d{4}(\.\d{3})?')
        for table in all_tables: 
            matched_elements = list (table.find_all(string=regex_pattern))
            if len(matched_elements) >0: filtered_strings.extend(matched_elements)
    else:
        print('Failed to fetch the webpage.')
        return
    return  [str(i) for i in filtered_strings]

def get_report_data (reports: list):
    """get the a Table of interacted techniques in each CISA report given the list of report codes.
    Args:
        report_codes (list): the list of report. each report is a dict storing (1) the report code and 
        (2) a list of techniques that will not be used for evaluation

    """
    group_IDs = []
    reported_techniques = []
    passive_techniques = []
    active_techniques = []
    for report in reports:
        url = 'https://www.cisa.gov/news-events/cybersecurity-advisories/' + report['code']
        group_IDs.append(report['code'])
        all_techniques  = _extract_cisa_techniques (url)
        reported_techniques.append (all_techniques[:])
        passive_techniques.append(report['passive_techniques'])
        if report['passive_techniques'] is not None:
            for technique in report['passive_techniques']:
                if technique in all_techniques: all_techniques.remove (technique)
        active_techniques.append(all_techniques)
    data = {
        'group_ID': group_IDs,
        'reported_techniques': reported_techniques,
        'passive_techniques': passive_techniques,
        'active_techniques': active_techniques,
    }
    report_data = pd.DataFrame (data=data)
    return report_data
