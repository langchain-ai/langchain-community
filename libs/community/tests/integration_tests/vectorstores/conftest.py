from typing import Union

import pytest
from vcr.request import Request


# This fixture returns a dictionary containing filter_headers options
# for replacing certain headers with dummy values during cassette playback
# Specifically, it replaces the authorization header with a dummy value to
# prevent sensitive data from being recorded in the cassette.
# It also filters request to certain hosts (specified in the `ignored_hosts` list)
# to prevent data from being recorded in the cassette.
@pytest.fixture(scope="module")
def vcr_config() -> dict:
    skipped_host = ["pinecone.io"]

    def before_record_response(response: dict) -> Union[dict, None]:
        return response

    def before_record_request(request: Request) -> Union[Request, None]:
        for host in skipped_host:
            if request.host.startswith(host) or request.host.endswith(host):
                return None
        return request

    return {
        "before_record_request": before_record_request,
        "before_record_response": before_record_response,
        "filter_headers": [
            ("authorization", "authorization-DUMMY"),
            ("X-OpenAI-Client-User-Agent", "X-OpenAI-Client-User-Agent-DUMMY"),
            ("Api-Key", "Api-Key-DUMMY"),
            ("User-Agent", "User-Agent-DUMMY"),
        ],
        "ignore_localhost": True,
    }
