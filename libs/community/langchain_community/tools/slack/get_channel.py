import json
import logging
from typing import Any, Optional, Type

from pydantic import BaseModel, Field

from langchain_core.callbacks import CallbackManagerForToolRun

from langchain_community.tools.slack.base import SlackBaseTool

class SlackGetChannelSchema(BaseModel):
    """Placeholder input schema for SlackGetChannel.

    This is intentionally minimal. Add fields later if the tool needs
    structured input (for example: pagination cursors or filters).
    """
    pass


class SlackGetChannel(SlackBaseTool):
    """Tool that gets Slack channel information."""

    name: str = "get_channelid_name_dict"
    description: str = (
        "Use this tool to get channelid-name dict. There is no input to this tool"
    )
    args_schema: Type[SlackGetChannelSchema] = SlackGetChannelSchema

    def _run(
        self, *args: Any, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        try:
            logging.getLogger(__name__)

            result = self.client.conversations_list()
            channels = result["channels"]
            filtered_result = [
                {key: channel[key] for key in ("id", "name", "created", "num_members")}
                for channel in channels
                if "id" in channel
                and "name" in channel
                and "created" in channel
                and "num_members" in channel
            ]
            return json.dumps(filtered_result, ensure_ascii=False)

        except Exception as e:
            return "Error creating conversation: {}".format(e)
