import requests
import json
from datetime import datetime, timedelta
from functools import wraps

# Path to the JSONL file
jsonl_file_path = 'path_to_your_json_file.jsonl'


# Load the JSON lines with organization details
def load_org_details():
    with open(jsonl_file_path, 'r') as file:
        org_details = [json.loads(line) for line in file]
    return org_details


# Update the JSONL file with the new rate limit timestamp
def update_rate_limit_timestamp(index, timestamp):
    org_details = load_org_details()
    org_details[index]['last_rate_limit'] = timestamp.isoformat()
    with open(jsonl_file_path, 'w') as file:
        for org_detail in org_details:
            file.write(json.dumps(org_detail) + '\n')


# Get the current organization index based on last rate limit encountered
def get_current_org_index(org_details):
    current_time = datetime.utcnow()
    for i, org in enumerate(org_details):
        last_rate_limit_str = org.get('last_rate_limit')
        if last_rate_limit_str:
            last_rate_limit = datetime.fromisoformat(last_rate_limit_str)
            if last_rate_limit.date() < current_time.date():
                return i  # If last error was before today, pick this org
    return 0  # Default to the first org if no rate limits were encountered yesterday


org_details = load_org_details()
current_org_index = get_current_org_index(org_details)


# Decorator for handling rate limit and switching keys
def handle_rate_limit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        nonlocal current_org_index
        try:
            response = func(*args, **kwargs)
            # Check if we hit the rate limit
            if response.status_code == 429:
                error_details = response.json()
                if error_details.get('error', {}).get('code') == 'rate_limit_exceeded':
                    # Record the rate limit error timestamp
                    update_rate_limit_timestamp(current_org_index, datetime.utcnow())
                    # Switch to the next organization key
                    current_org_index = (current_org_index + 1) % len(org_details)
                    print("Rate limit reached, switching to next key...")
                    return wrapper(*args, **kwargs)  # Retry with the next key
            return response
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    return wrapper


# # The API request function
# @handle_rate_limit
# def make_api_request(url):
#     headers = {
#         'Authorization': f"Bearer {org_details[current_org_index]['key']}"
#     }
#     response = requests.get(url, headers=headers)
#     return response
#
#
# # Example usage within a for loop
# for _ in range(10):  # Replace 10 with the actual number of requests you need to make
#     response = make_api_request('your_api_endpoint')
#     # Do something with the response
