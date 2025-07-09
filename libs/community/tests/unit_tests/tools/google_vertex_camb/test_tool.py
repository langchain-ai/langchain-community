"""Test GoogleVertexCambTool."""

import base64
import json
import os
import tempfile
import uuid
from unittest.mock import Mock, mock_open, patch

import pytest

from langchain_community.tools.google_vertex_camb import GoogleVertexCambTool


def test_google_vertex_camb_tool_constructor() -> None:
    """Test GoogleVertexCambTool constructor validation."""
    # Test missing environment variables
    with pytest.raises(ValueError):
        # Remove required environment variables
        for var in ["PROJECT_ID", "LOCATION", "ENDPOINT_ID", "REFERENCE_AUDIO_PATH"]:
            os.environ.pop(var, None)
        GoogleVertexCambTool()  # type: ignore[call-arg]

    # Test missing Google Cloud credentials
    with pytest.raises(ValueError):
        os.environ.pop("GOOGLE_APPLICATION_CREDENTIALS", None)
        GoogleVertexCambTool(
            project_id="test-project",
            location="us-central1",
            endpoint_id="test-endpoint",
            reference_audio_path="/path/to/audio.wav",
        )

    # Test successful initialization
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/path/to/credentials.json"
    tool = GoogleVertexCambTool(
        project_id="test-project",
        location="us-central1",
        endpoint_id="test-endpoint",
        reference_audio_path="/path/to/audio.wav",
    )
    assert tool.project_id == "test-project"
    assert tool.location == "us-central1"
    assert tool.endpoint_id == "test-endpoint"
    assert tool.reference_audio_path == "/path/to/audio.wav"
    assert tool.language == "en-us"  # default


def test_google_vertex_camb_tool_from_env() -> None:
    """Test GoogleVertexCambTool initialization from environment variables."""
    # Set up environment variables
    os.environ["PROJECT_ID"] = "env-project"
    os.environ["LOCATION"] = "env-location"
    os.environ["ENDPOINT_ID"] = "env-endpoint"
    os.environ["REFERENCE_AUDIO_PATH"] = "/env/audio.wav"
    os.environ["REFERENCE_TEXT"] = "env reference text"
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/path/to/credentials.json"

    tool = GoogleVertexCambTool()  # type: ignore[call-arg]

    assert tool.project_id == "env-project"
    assert tool.location == "env-location"
    assert tool.endpoint_id == "env-endpoint"
    assert tool.reference_audio_path == "/env/audio.wav"
    assert tool.reference_text == "env reference text"


def test_google_vertex_camb_tool_properties() -> None:
    """Test GoogleVertexCambTool properties."""
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/path/to/credentials.json"

    tool = GoogleVertexCambTool(
        project_id="test-project",
        location="us-central1",
        endpoint_id="test-endpoint",
        reference_audio_path="/path/to/audio.wav",
        language="es-us",
    )

    assert tool.name == "google_vertex_camb"
    assert "voice cloning" in tool.description
    assert tool.language == "es-us"


@patch("langchain_community.tools.google_vertex_camb.tool._import_vertex_ai")
def test_google_vertex_camb_tool_missing_reference_audio(
    mock_import_vertex_ai: Mock,
) -> None:
    """Test GoogleVertexCambTool with missing reference audio file."""
    # Setup mocks
    mock_aiplatform = Mock()
    mock_import_vertex_ai.return_value = mock_aiplatform

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/path/to/credentials.json"

    tool = GoogleVertexCambTool(
        project_id="test-project",
        location="us-central1",
        endpoint_id="test-endpoint",
        reference_audio_path="/nonexistent/audio.wav",
    )

    with pytest.raises(RuntimeError, match="Reference audio file not found"):
        tool._run("Test text")


@patch("langchain_community.tools.google_vertex_camb.tool._import_vertex_ai")
def test_google_vertex_camb_tool_run_success(mock_import_vertex_ai: Mock) -> None:
    """Test successful GoogleVertexCambTool run."""
    # Setup mocks
    mock_aiplatform = Mock()
    mock_import_vertex_ai.return_value = mock_aiplatform

    mock_endpoint = Mock()
    mock_aiplatform.Endpoint.return_value = mock_endpoint

    # Mock response
    test_audio_bytes = b"test_audio_content"
    mock_response = Mock()
    mock_response.content = json.dumps(
        {"predictions": [base64.b64encode(test_audio_bytes).decode("utf-8")]}
    ).encode("utf-8")
    mock_endpoint.raw_predict.return_value = mock_response

    # Setup environment and tool
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/path/to/credentials.json"

    with (
        tempfile.TemporaryDirectory() as tmp_dir,
        patch("uuid.uuid4") as mock_uuid,
        patch("os.path.exists") as mock_exists,
        patch("builtins.open", mock_open(read_data=b"reference_audio_data")),
        patch("os.getcwd", return_value=tmp_dir),
    ):
        mock_uuid_value = uuid.UUID("12345678-1234-5678-9012-123456789012")
        mock_uuid.return_value = mock_uuid_value
        mock_exists.return_value = True

        tool = GoogleVertexCambTool(
            project_id="test-project",
            location="us-central1",
            endpoint_id="test-endpoint",
            reference_audio_path="/path/to/audio.wav",
        )

        result = tool._run("Hello world")

        # Verify result
        expected_filename = f"vertex_camb_speech_{mock_uuid_value}.flac"
        assert result == expected_filename

        # Verify Vertex AI initialization
        mock_aiplatform.init.assert_called_once_with(
            project="test-project", location="us-central1"
        )

        # Verify endpoint creation
        mock_aiplatform.Endpoint.assert_called_once_with(endpoint_name="test-endpoint")

        # Verify prediction call
        mock_endpoint.raw_predict.assert_called_once()
        call_args = mock_endpoint.raw_predict.call_args

        # Check the body content
        body_data = json.loads(call_args[1]["body"].decode("utf-8"))
        assert body_data["instances"][0]["text"] == "Hello world"
        assert body_data["instances"][0]["language"] == "en-us"
        assert "audio_ref" in body_data["instances"][0]

        # Check headers
        assert call_args[1]["headers"]["Content-Type"] == "application/json"


@patch("langchain_community.tools.google_vertex_camb.tool._import_vertex_ai")
def test_google_vertex_camb_tool_run_with_reference_text(
    mock_import_vertex_ai: Mock,
) -> None:
    """Test GoogleVertexCambTool run with reference text."""
    # Setup mocks
    mock_aiplatform = Mock()
    mock_import_vertex_ai.return_value = mock_aiplatform

    mock_endpoint = Mock()
    mock_aiplatform.Endpoint.return_value = mock_endpoint

    # Mock response
    test_audio_bytes = b"test_audio_content"
    mock_response = Mock()
    mock_response.content = json.dumps(
        {"predictions": [base64.b64encode(test_audio_bytes).decode("utf-8")]}
    ).encode("utf-8")
    mock_endpoint.raw_predict.return_value = mock_response

    # Setup environment and tool
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/path/to/credentials.json"

    with (
        tempfile.TemporaryDirectory() as tmp_dir,
        patch("uuid.uuid4") as mock_uuid,
        patch("os.path.exists") as mock_exists,
        patch("builtins.open", mock_open(read_data=b"reference_audio_data")),
        patch("os.getcwd", return_value=tmp_dir),
    ):
        mock_uuid_value = uuid.UUID("12345678-1234-5678-9012-123456789012")
        mock_uuid.return_value = mock_uuid_value
        mock_exists.return_value = True

        tool = GoogleVertexCambTool(
            project_id="test-project",
            location="us-central1",
            endpoint_id="test-endpoint",
            reference_audio_path="/path/to/audio.wav",
            reference_text="This is reference text",
        )

        tool._run("Hello world")

        # Verify prediction call includes reference text
        call_args = mock_endpoint.raw_predict.call_args
        body_data = json.loads(call_args[1]["body"].decode("utf-8"))
        assert body_data["instances"][0]["ref_text"] == "This is reference text"


@patch("langchain_community.tools.google_vertex_camb.tool._import_vertex_ai")
def test_google_vertex_camb_tool_run_empty_predictions(
    mock_import_vertex_ai: Mock,
) -> None:
    """Test GoogleVertexCambTool run with empty predictions."""
    # Setup mocks
    mock_aiplatform = Mock()
    mock_import_vertex_ai.return_value = mock_aiplatform

    mock_endpoint = Mock()
    mock_aiplatform.Endpoint.return_value = mock_endpoint

    # Mock response with empty predictions
    mock_response = Mock()
    mock_response.content = json.dumps({"predictions": []}).encode("utf-8")
    mock_endpoint.raw_predict.return_value = mock_response

    # Setup environment and tool
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/path/to/credentials.json"

    with (
        patch("os.path.exists") as mock_exists,
        patch("builtins.open", mock_open(read_data=b"reference_audio_data")),
    ):
        mock_exists.return_value = True

        tool = GoogleVertexCambTool(
            project_id="test-project",
            location="us-central1",
            endpoint_id="test-endpoint",
            reference_audio_path="/path/to/audio.wav",
        )

        with pytest.raises(RuntimeError, match="No audio predictions returned"):
            tool._run("Hello world")


@patch("langchain_community.tools.google_vertex_camb.tool._import_vertex_ai")
def test_google_vertex_camb_tool_run_vertex_ai_error(
    mock_import_vertex_ai: Mock,
) -> None:
    """Test GoogleVertexCambTool run with Vertex AI error."""
    # Setup mocks
    mock_aiplatform = Mock()
    mock_import_vertex_ai.return_value = mock_aiplatform

    mock_endpoint = Mock()
    mock_aiplatform.Endpoint.return_value = mock_endpoint
    mock_endpoint.raw_predict.side_effect = Exception("Vertex AI error")

    # Setup environment and tool
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/path/to/credentials.json"

    with (
        patch("os.path.exists") as mock_exists,
        patch("builtins.open", mock_open(read_data=b"reference_audio_data")),
    ):
        mock_exists.return_value = True

        tool = GoogleVertexCambTool(
            project_id="test-project",
            location="us-central1",
            endpoint_id="test-endpoint",
            reference_audio_path="/path/to/audio.wav",
        )

        with pytest.raises(
            RuntimeError, match="Error while running GoogleVertexCambTool"
        ):
            tool._run("Hello world")


def test_google_vertex_camb_tool_import_vertex_ai_error() -> None:
    """Test _import_vertex_ai with missing package."""
    with patch(
        "langchain_community.tools.google_vertex_camb.tool._import_vertex_ai"
    ) as mock_import:
        mock_import.side_effect = ImportError("Cannot import Vertex AI")

        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/path/to/credentials.json"

        with patch("os.path.exists", return_value=True):
            tool = GoogleVertexCambTool(
                project_id="test-project",
                location="us-central1",
                endpoint_id="test-endpoint",
                reference_audio_path="/path/to/audio.wav",
            )

            with pytest.raises(
                RuntimeError, match="Error while running GoogleVertexCambTool"
            ):
                tool._run("Hello world")


def test_mars7_language_type() -> None:
    """Test Mars7Language type validation."""
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/path/to/credentials.json"

    # Test valid language
    tool = GoogleVertexCambTool(
        project_id="test-project",
        location="us-central1",
        endpoint_id="test-endpoint",
        reference_audio_path="/path/to/audio.wav",
        language="es-es",
    )
    assert tool.language == "es-es"

    # Test with different valid languages
    valid_languages = [
        "de-de",
        "en-gb",
        "en-us",
        "es-us",
        "fr-ca",
        "fr-fr",
        "ja-jp",
        "ko-kr",
        "zh-cn",
    ]
    for lang in valid_languages:
        tool = GoogleVertexCambTool(
            project_id="test-project",
            location="us-central1",
            endpoint_id="test-endpoint",
            reference_audio_path="/path/to/audio.wav",
            language=lang,  # type: ignore[arg-type]
        )
        assert tool.language == lang
