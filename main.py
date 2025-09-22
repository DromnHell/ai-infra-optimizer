import os
import argparse
import json
import time
from pathlib import Path
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END, START
from typing import Dict, Any, List

load_dotenv()

def find_json_file(filename = "rapport.json") -> Path:
    """
    Search for the JSON file and verify that it is valid.
    Return the file path if found and valid, None otherwise.
    """
    file_path = Path(filename)

    if not file_path.exists():
        print(f"File '{filename}' is missing in {Path.cwd()}")
        return None
    try:
        with open(file_path, "r") as f:
            json.load(f)
    except json.JSONDecodeError:
        print(f"File '{filename}' exists but is not valid JSON.")
        return None
    except Exception as e:
        print(f"Error reading file  '{filename}': {e}")
        return None

    return file_path

def ingestion_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    The “ingestion” node reads the JSON report and simulates a chronological stream.
    It starts from the oldest timestamp and moves forward at each loop iteration.
    """
    file_path = find_json_file("rapport.json")
    if file_path is None:
        state["new_entries"] = []
        return state

    with open(file_path, "r") as f:
        data = json.load(f)

    data = sorted(data, key = lambda x: x["timestamp"])

    last_ts = state.get("last_timestamp")

    if last_ts is None:
        new_entries = [data[0]]
        state["last_timestamp"] = new_entries[0]["timestamp"]
    else:
        newer = [e for e in data if e["timestamp"] > last_ts]
        if newer:
            new_entries = [newer[0]]
            state["last_timestamp"] = new_entries[0]["timestamp"]
        else:
            new_entries = []

    state["new_entries"] = new_entries
    return state

def detect_anomalies(entry: Dict[str, Any]) -> List[str]:
    """
    Detects and returns anomalies based on monitoring heuristics.
    """
    anomalies = []

    rules = {
        "cpu_usage": lambda v: f"High CPU({v}%)" if v > 80 else None,
        "latency_ms": lambda v: f"High latency ({v} ms)" if v > 200 else None,
        "error_rate": lambda v: f"Abnormal error rate ({v*100:.1f}%)" if v > 0.05 else None,
        "temperature_celsius": lambda v: f"High temperature({v} °C)" if v > 75 else None,
        "service_status": lambda v: [
            msg for k, msg in [
                ("database", "Database offline"),
                ("api_gateway", "API Gateway degraded"),
                ("cache", "Cache degraded")
            ] if v.get(k) != "online"
        ] or None
    }

    for key, func in rules.items():
        if key in entry:
            result = func(entry[key])
            if result:
                if isinstance(result, list):
                    anomalies.extend(result)
                else:
                    anomalies.append(result)

    return anomalies

def rule_analysis_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    The “rule analysis” node of the graph that detects anomalies and update the state with it.
    """
    anomalies_per_entry = []
    for entry in state.get("new_entries", []):
        anomalies = detect_anomalies(entry)
        if anomalies:
            anomalies_per_entry.append({"entry": entry, "anomalies": anomalies})
    state["anomalies_per_entry"] = anomalies_per_entry
    return state


def extract_text(response, provider):
    """
    Extracts the text from the LLM response based on the provider.
    """
    if provider == "openai" or provider == "mistral":
        return response.content if response.content else ""
    elif provider == "mock":
        return response
    else:
        raise ValueError(f"'{provider}' is Unsupported ")

def get_llm(provider = "mock"):
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model = "gpt-4o-mini", temperature = 0.0, openai_api_key = api_key)
    elif provider == "mistral":
        api_key = os.getenv("MISTRAL_API_KEY")
        from langchain_mistralai import ChatMistralAI
        return ChatMistralAI(model = "mistral-medium", temperature = 0.0, mistral_api_key = api_key)
    elif provider == "mock":
        from mock_llm import MockLLM
        return MockLLM()
    else:
        raise ValueError(f"Provider {provider} not supported.")

def llm_analysis_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    The “LLM analysis” node of the graph that calls an LLM to analyze the heuristics monitoring anomalies report.
    Always format the input as a string for the LLM.
    """
    provider = state.get("provider", "mock")
    llm = get_llm(provider)

    anomalies_list = state.get("anomalies_per_entry", [])
    if not anomalies_list:
        state["llm_report"] = 'No problem detected.'
        return state

    prompt_texts = []
    for item in anomalies_list:
        entry = item.get("entry", {})
        anomalies = item.get("anomalies", [])
        prompt = f"""
    Here are the metrics measured: {entry}
    Here are the anomalies detected by rules: {anomalies}

    1. Identifies potential “weak signals” or suspicious combinations.
    2. Provides practical recommendations for a CTO.
    3. Returns the associated recommendations in a structured Markdown format with no title and the following sections:
   - **Weak Signals and Recommendations:** for each anomaly, provide a subsection with numbered title and a list of recommendations (bold the recommendation titles)
   - **Summary:** a short paragraph summarizing overall system health and any required actions

    The output must be ready for console display or email.
    """

    prompt_texts.append(prompt.strip())

    full_prompt = "\n\n".join(prompt_texts)

    response = llm.invoke(full_prompt)

    state["llm_report"] = extract_text(response, provider)

    return state


def output_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    The “output” node of the graph that displays only when a new input is detected.
    """
    if not state.get("new_entries"):
        print("No new data yet. Waiting...")
        return state

    print("\n=== Anomalies report ===")

    print("Timestamp :", state.get("last_timestamp"))

    anomalies = [a for item in state.get("anomalies_per_entry", [])
                 for a in item.get("anomalies", [])]
    if anomalies:
        print("Detected anomalies :", ", ".join(anomalies) + ".")
    else:
        print("Detected anomalies : None.")

    print("Recommendations :\n", state.get("llm_report", []))
    return state


def build_graph():
    workflow = StateGraph(dict)

    workflow.add_node("ingestion", ingestion_node)
    workflow.add_node("rule_analysis", rule_analysis_node)
    workflow.add_node("llm_analysis", llm_analysis_node)
    workflow.add_node("output", output_node)

    workflow.add_edge(START, "ingestion")
    workflow.add_edge("ingestion", "rule_analysis")
    workflow.add_edge("rule_analysis", "llm_analysis")
    workflow.add_edge("llm_analysis", "output")
    workflow.add_edge("output", END)

    app = workflow.compile()
    return app


def check_api_key(var_name: str, provider: str):
    key = os.getenv(var_name)
    if not key:
        raise ValueError(f"Variable {var_name} is missing.")
    if provider == "openai" and not key.startswith("sk-"):
        raise ValueError(f"Variable {var_name} seems invalid for OpenAI: {key[:5]}…")
    return key


def main():
    parser = argparse.ArgumentParser(description = "Monitoring with LLM anomaly detection")
    parser.add_argument(
        "--provider",
        type = str,
        default = "mock",
        choices = ["mock", "openai", "mistral"],
        help = "LLM provider to use (default: mock)",
    )
    args = parser.parse_args()

    if args.provider == "openai":
        check_api_key("OPENAI_API_KEY", provider = "openai")
    elif args.provider == "mistral":
        check_api_key("MISTRAL_API_KEY", provider = "mistral")

    app = build_graph()

    state = {
        "last_timestamp": None,
        "provider": args.provider,
    }

    print("Monitoring started...")
    while True:
        state = app.invoke(state)
        time.sleep(3)


if __name__ == "__main__":
    main()
