import requests

# add your API key to the headers
headers = {
    "X-API-KEY": "ec446cc5-b717-49f0-a844-2303103c6b0a"
}

# set your query params
ticker = 'AAPL'

# create the URL
url = (
    f'https://api.financialdatasets.ai/financial-metrics/snapshot'
    f'?ticker={ticker}'
)

# make API request
response = requests.get(url, headers=headers)

# parse snapshot from the response
snapshot = response.json().get('snapshot')
