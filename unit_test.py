import json
import sys
import pytest
from unittest.mock import patch, MagicMock

from main import (
    find_json_file,
    ingestion_node,
    detect_anomalies,
    rule_analysis_node,
    get_llm,
    llm_analysis_node,
    output_node,
    build_graph,
    check_api_key,
    main
)
from mock_llm import MockLLM


### --- Tests for find_json_file --- ###

def test_find_json_file_not_exist(tmp_path):
    file = tmp_path / "notfound.json"
    result = find_json_file(file)
    assert result is None

def test_find_json_file_invalid_json(tmp_path):
    file = tmp_path / "rapport.json"
    file.write_text("{invalid json}")
    result = find_json_file(file)
    assert result is None

def test_find_json_file_valid_json(tmp_path):
    file = tmp_path / "rapport.json"
    file.write_text(json.dumps([{"timestamp": 1}]))
    result = find_json_file(file)
    assert result == file


### --- Tests for ingestion_node --- ###

def test_ingestion_node_first_entry(tmp_path, monkeypatch):
    file = tmp_path / "rapport.json"
    file.write_text(json.dumps([
        {"timestamp": 2}, {"timestamp": 1}
    ]))

    monkeypatch.chdir(tmp_path)

    state = {"last_timestamp": None}
    new_state = ingestion_node(state)
    assert "new_entries" in new_state
    assert new_state["last_timestamp"] == 1

def test_ingestion_node_newer_entry(tmp_path, monkeypatch):
    file = tmp_path / "rapport.json"
    file.write_text(json.dumps([
        {"timestamp": 1}, {"timestamp": 2}
    ]))

    monkeypatch.chdir(tmp_path)

    state = {"last_timestamp": 1}
    new_state = ingestion_node(state)
    assert new_state["last_timestamp"] == 2
    assert new_state["new_entries"][0]["timestamp"] == 2

def test_ingestion_node_no_new_entry(tmp_path, monkeypatch):
    file = tmp_path / "rapport.json"
    file.write_text(json.dumps([
        {"timestamp": 1}
    ]))

    monkeypatch.chdir(tmp_path)

    state = {"last_timestamp": 1}
    new_state = ingestion_node(state)
    assert new_state["new_entries"] == []


### --- Tests for detect_anomalies --- ###

def test_detect_anomalies_cpu_latency():
    entry = {"cpu_usage": 90, "latency_ms": 250}
    anomalies = detect_anomalies(entry)
    assert "High CPU(90%)" in anomalies
    assert "High latency (250 ms)" in anomalies

def test_detect_anomalies_error_temp_service():
    entry = {
        "error_rate": 0.1,
        "temperature_celsius": 80,
        "service_status": {"database": "offline", "api_gateway": "degraded", "cache": "online"},
    }
    anomalies = detect_anomalies(entry)
    assert any("error rate" in a for a in anomalies)
    assert any("temperature" in a for a in anomalies)
    assert "Database offline" in anomalies
    assert "API Gateway degraded" in anomalies


### --- Tests for rule_analysis_node --- ###

def test_rule_analysis_node_with_anomalies():
    entry = {"cpu_usage": 95}
    state = {"new_entries": [entry]}
    new_state = rule_analysis_node(state)
    assert "anomalies_per_entry" in new_state
    assert new_state["anomalies_per_entry"][0]["anomalies"]

def test_rule_analysis_node_no_anomalies():
    entry = {"cpu_usage": 10}
    state = {"new_entries": [entry]}
    new_state = rule_analysis_node(state)
    assert new_state["anomalies_per_entry"] == []


### --- Tests for get_llm --- ###

def test_get_llm_mock():
    llm = get_llm("mock")
    assert isinstance(llm, MockLLM)

def test_get_llm_invalid_provider():
    with pytest.raises(ValueError):
        get_llm("doesnotexist")

@patch("main.ChatOpenAI")
def test_get_llm_openai(mock_chat, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake")
    mock_chat.return_value = MagicMock(invoke = lambda x: "ok")
    llm = get_llm("openai")
    assert hasattr(llm, "invoke")

@patch("main.ChatAnthropic")
def test_get_llm_claude(mock_chat, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")
    mock_chat.return_value = MagicMock(invoke = lambda x: "ok")
    llm = get_llm("claude")
    assert hasattr(llm, "invoke")

@patch("main.ChatMistralAI")
def test_get_llm_mistral(mock_chat, monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "fake")
    mock_chat.return_value = MagicMock(invoke = lambda x: "ok")
    llm = get_llm("mistral")
    assert hasattr(llm, "invoke")


### --- Tests for llm_analysis_node --- ###

def test_llm_analysis_node_no_anomalies(monkeypatch):
    state = {"anomalies_per_entry": [], "provider": "mock"}
    new_state = llm_analysis_node(state)
    assert new_state["llm_report"] == "No problem detected."

def test_llm_analysis_node_with_anomalies(monkeypatch):
    state = {
        "anomalies_per_entry": [
            {"entry": {"cpu_usage": 90}, "anomalies": ["High CPU(90%)"]}
        ],
        "provider": "mock"
    }
    new_state = llm_analysis_node(state)
    assert new_state["llm_report"]


### --- Tests for output_node --- ###

def test_output_node_no_new_entries(capsys):
    state = {"new_entries": []}
    new_state = output_node(state)
    captured = capsys.readouterr()
    assert "No new data yet" in captured.out
    assert new_state == state

def test_output_node_with_anomalies(capsys):
    state = {
        "new_entries": [{"timestamp": 1}],
        "last_timestamp": 1,
        "anomalies_per_entry": [{"anomalies": ["High CPU(90%)"]}],
        "llm_report": "Test report"
    }
    output_node(state)
    captured = capsys.readouterr()
    assert "Anomalies report" in captured.out
    assert "High CPU" in captured.out
    assert "Test report" in captured.out


### --- Tests for build_graph --- ###

def test_build_graph():
    app = build_graph()
    assert callable(app.invoke)


### --- Tests for check_api_key --- ###

def test_check_api_key_missing(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising = False)
    with pytest.raises(ValueError):
        check_api_key("OPENAI_API_KEY")

def test_check_api_key_invalid(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "abc123")
    with pytest.raises(ValueError):
        check_api_key("OPENAI_API_KEY")

def test_check_api_key_valid(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-validkey123")
    result = check_api_key("OPENAI_API_KEY")
    assert result == "sk-validkey123"

### -------- Tests main --------

@patch("main.build_graph")
@patch("main.check_api_key")
@patch("time.sleep", side_effect = Exception("stop"))
def test_main_with_mock(mock_sleep, mock_check, mock_build, monkeypatch):
    mock_app = MagicMock()
    mock_app.invoke.side_effect = [{"provider": "mock"}]
    mock_build.return_value = mock_app

    test_args = ["prog", "--provider", "mock"]
    monkeypatch.setattr(sys, "argv", test_args)

    with pytest.raises(Exception, match = "stop"):
        main()

    mock_check.assert_not_called()

@patch("main.build_graph")
@patch("main.check_api_key")
@patch("time.sleep", side_effect = Exception("stop"))
def test_main_with_openai(mock_sleep, mock_check, mock_build, monkeypatch):
    mock_app = MagicMock()
    mock_app.invoke.side_effect = [{"provider": "openai"}]
    mock_build.return_value = mock_app

    test_args = ["prog", "--provider", "openai"]
    monkeypatch.setattr(sys, "argv", test_args)

    with pytest.raises(Exception, match = "stop"):
        main()

    mock_check.assert_called_once_with("OPENAI_API_KEY")

@patch("main.build_graph")
@patch("main.check_api_key")
@patch("time.sleep", side_effect = Exception("stop"))
def test_main_with_claude(mock_sleep, mock_check, mock_build, monkeypatch):
    mock_app = MagicMock()
    mock_app.invoke.side_effect = [{"provider": "claude"}]
    mock_build.return_value = mock_app

    test_args = ["prog", "--provider", "claude"]
    monkeypatch.setattr(sys, "argv", test_args)

    with pytest.raises(Exception, match = "stop"):
        main()

    mock_check.assert_called_once_with("ANTHROPIC_API_KEY")

@patch("main.build_graph")
@patch("main.check_api_key")
@patch("time.sleep", side_effect = Exception("stop"))
def test_main_with_mistral(mock_sleep, mock_check, mock_build, monkeypatch):
    mock_app = MagicMock()
    mock_app.invoke.side_effect = [{"provider": "mistral"}]
    mock_build.return_value = mock_app

    test_args = ["prog", "--provider", "mistral"]
    monkeypatch.setattr(sys, "argv", test_args)

    with pytest.raises(Exception, match = "stop"):
        main()

    mock_check.assert_called_once_with("MISTRAL_API_KEY")