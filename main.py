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
        state["new_entry"] = []
        return state

    with open(file_path, "r") as f:
        data = json.load(f)

    data = sorted(data, key = lambda x: x["timestamp"])

    last_ts = state.get("last_timestamp")

    if last_ts is None:
        new_entry = data[0]
        state["last_timestamp"] = new_entry["timestamp"]
    else:
        newer = [e for e in data if e["timestamp"] > last_ts]
        if newer:
            new_entry = newer[0]
            state["last_timestamp"] = new_entry["timestamp"]
        else:
            new_entry = []

    state["new_entry"] = new_entry
    return state

def detect_anomalies(entry: Dict[str, Any]) -> List[str]:
    """
    Detects and returns anomalies based on monitoring heuristics.
    """
    anomalies = []

    rules = {
        "cpu_usage": lambda v: f"CPU élevé ({v}%)" if v > 80 else None,
        "memory_usage": lambda v: f"Utilisation élevée de la mémoire ({v}%)" if v > 85 else None,
        "disk_usage": lambda v: f"Utilisation élevée du disque ({v}%)" if v > 90 else None,
        "latency_ms": lambda v: f"Latence élevée ({v} ms)" if v > 200 else None,
        "error_rate": lambda v: f"Taux d'erreur anormal ({v*100:.1f}%)" if v > 0.05 else None,
        "temperature_celsius": lambda v: f"Temperature élevée ({v} °C)" if v > 75 else None,
        "io_wait": lambda v: f"Attente I/O élevée ({v}%)" if v > 20 else None,
        "thread_count": lambda v: f"Trop de threads ({v})" if v > 500 else None,
        "active_connections": lambda v: f"Trop de connexions actives ({v})" if v > 200 else None,
        "network_in_kbps": lambda v: f"Entrée réseau élevée ({v} kbps)" if v > 10000 else None,
        "network_out_kbps": lambda v: f"Sortie réseau élevée ({v} kbps)" if v > 10000 else None,
        "power_consumption_watts": lambda v: f"Consommation électrique élevée ({v} W)" if v > 400 else None,
        "uptime_seconds": lambda v: f"Redémarrages fréquents (uptime {v//3600}h)" if v < 3600 else None,
        "service_status": lambda v: [
            msg for k, msg in [
                ("database", "Base de données hors ligne"),
                ("api_gateway", "API Gateway dégradée"),
                ("cache", "Cache dégradé")
            ] if v.get(k) not in ("online", None)
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
    entry = state.get("new_entry", [])
    anomalies_per_entry = []
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
    """
    provider = state.get("provider", "mock")
    llm = get_llm(provider)

    anomalies_list = state.get("anomalies_per_entry", [])
    if not anomalies_list:
        state["llm_report"] = 'Aucun problème détecté.'
        state["llm_report"] = 'Aucun problème détecté.'
        return state

    item = anomalies_list[0]
    entry = item.get("entry", {})
    anomalies = item.get("anomalies", [])
    prompt =\
    f"""
    Voici les indicateurs mesurés : {entry}
    Voici les anomalies détectées par les règles : {anomalies}
    
    1. Identifie les « signaux faibles » potentiels ou les combinaisons suspectes.
    2. Fournit des recommandations pratiques pour un directeur technique (CTO).
    3. Renvoie les recommandations associées dans un format Markdown structuré sans titre et comprenant les sections suivantes :
    - **Signaux faibles et recommandations :** pour chaque anomalie, fournit une sous-section avec un titre numéroté et une liste de recommandations (les titres des recommandations sont en gras)
    - **Résumé :** un court paragraphe résumant l'état général du système et les actions requises
    
    Le résultat doit être prêt à être affiché sur la console ou envoyé par e-mail. Le résultat doit être écrit en français.
    """

    response = llm.invoke(prompt)

    state["llm_report"] = extract_text(response, provider)

    return state


def output_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    The “output” node of the graph that displays only when a new input is detected.
    """
    if not state.get("new_entry"):
        print("Pas de nouvelles entrées. En attente...")
        return state

    print("\n=== Rapport d'anomalies ===")

    print("Horodatage :", state.get("last_timestamp"))

    anomalies = [a for item in state.get("anomalies_per_entry", [])
                 for a in item.get("anomalies", [])]
    if anomalies:
        print("Anomalies détectées :", ", ".join(anomalies) + ".")
    else:
        print("Anomalies détectées : Aucune.")

    print("Recommandations :\n", state.get("llm_report", []))
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

    print("Analyse de l'infrastructure en cours...")
    while True:
        state = app.invoke(state)
        time.sleep(3)


if __name__ == "__main__":
    main()
