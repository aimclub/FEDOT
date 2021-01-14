import requests
import json

pulls_response = requests.get('https://api.github.com/repos/nccr-itmo/FEDOT/pulls')
pulls_list = json.loads(pulls_response.text)
pulled_branches = [pull['head']['ref'] for pull in pulls_list]

print(pulled_branches)
