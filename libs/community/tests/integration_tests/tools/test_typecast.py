"""Integration tests for Typecast Text2Speech Tool."""

import os

import pytest

from langchain_community.tools.typecast import TypecastText2SpeechTool


@pytest.mark.requires("typecast")
class TestTypecastText2SpeechTool:
    """Integration tests for TypecastText2SpeechTool."""

    def test_typecast_text_to_speech_basic(self) -> None:
        """Test basic text to speech conversion."""
        # Skip if no API key is set
        if not os.environ.get("TYPECAST_API_KEY"):
            pytest.skip("TYPECAST_API_KEY not set")

        tool = TypecastText2SpeechTool()
        text = "Hello world! This is a test."

        # Convert text to speech
        audio_file = tool.run(text)

        # Verify file was created
        assert audio_file is not None
        assert isinstance(audio_file, str)
        assert os.path.exists(audio_file)
        assert audio_file.endswith(".wav")

        # Clean up
        if os.path.exists(audio_file):
            os.remove(audio_file)

    def test_typecast_text_to_speech_with_emotion(self) -> None:
        """Test text to speech with emotion settings."""
        # Skip if no API key is set
        if not os.environ.get("TYPECAST_API_KEY"):
            pytest.skip("TYPECAST_API_KEY not set")

        tool = TypecastText2SpeechTool(
            emotion_preset="happy",
            emotion_intensity=1.5,
        )
        text = "I am so excited to show you this!"

        # Convert text to speech
        audio_file = tool.run(text)

        # Verify file was created
        assert audio_file is not None
        assert isinstance(audio_file, str)
        assert os.path.exists(audio_file)

        # Clean up
        if os.path.exists(audio_file):
            os.remove(audio_file)

    def test_typecast_text_to_speech_mp3_format(self) -> None:
        """Test text to speech with MP3 format."""
        # Skip if no API key is set
        if not os.environ.get("TYPECAST_API_KEY"):
            pytest.skip("TYPECAST_API_KEY not set")

        tool = TypecastText2SpeechTool(audio_format="mp3")
        text = "This should be in MP3 format."

        # Convert text to speech
        audio_file = tool.run(text)

        # Verify file was created
        assert audio_file is not None
        assert isinstance(audio_file, str)
        assert os.path.exists(audio_file)
        assert audio_file.endswith(".mp3")

        # Clean up
        if os.path.exists(audio_file):
            os.remove(audio_file)

    def test_typecast_text_to_speech_korean(self) -> None:
        """Test text to speech with Korean language."""
        # Skip if no API key is set
        if not os.environ.get("TYPECAST_API_KEY"):
            pytest.skip("TYPECAST_API_KEY not set")

        tool = TypecastText2SpeechTool(language="kor")
        text = "안녕하세요. 타입캐스트 테스트입니다."

        # Convert text to speech
        audio_file = tool.run(text)

        # Verify file was created
        assert audio_file is not None
        assert isinstance(audio_file, str)
        assert os.path.exists(audio_file)

        # Clean up
        if os.path.exists(audio_file):
            os.remove(audio_file)

    def test_typecast_with_agent(self) -> None:
        """Test TypecastText2SpeechTool with a LangChain agent."""
        # Skip if no API key is set
        if not os.environ.get("TYPECAST_API_KEY") or not os.environ.get(
            "OPENAI_API_KEY"
        ):
            pytest.skip("TYPECAST_API_KEY or OPENAI_API_KEY not set")

        try:
            from langchain.agents import create_agent
            from langchain_openai import ChatOpenAI
        except ImportError as e:
            pytest.skip(f"Required packages not installed: {e}")

        # Create LLM and tools
        llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
        tools = [TypecastText2SpeechTool()]

        # Create agent
        agent = create_agent(
            model=llm,
            tools=tools,
            system_prompt="You are a helpful assistant",
        )

        # Run the agent
        query = (
            "Convert this text to speech: 'Hello from LangChain agent!' "
            "and tell me the file path."
        )
        
        # Invoke the agent
        # The input schema for create_agent graph expects 'messages'
        response = agent.invoke({"messages": [{"role": "user", "content": query}]})

        # Verify response
        # The output state contains 'messages'. The last message should be AIMessage.
        messages = response["messages"]
        last_message = messages[-1]
        output = last_message.content
        print(f"\nAgent response: {output}")

        # The output should contain the file path or mention it
        assert ".wav" in output or ".mp3" in output

        # Try to clean up any audio files created during the test
        import tempfile
        import time

        temp_dir = tempfile.gettempdir()
        current_time = time.time()
        
        # Find the file path in the output if possible
        import re
        match = re.search(r'/tmp/[\w-]+\.(wav|mp3)', output)
        if match:
            file_path = match.group(0)
            if os.path.exists(file_path):
                os.remove(file_path)
        else:
            # Fallback cleanup
            for file in os.listdir(temp_dir):
                if file.endswith((".wav", ".mp3")):
                    file_path = os.path.join(temp_dir, file)
                    try:
                        # Only remove files created in the last 10 seconds
                        if os.path.getmtime(file_path) > (current_time - 10):
                            os.remove(file_path)
                    except Exception:
                        pass
