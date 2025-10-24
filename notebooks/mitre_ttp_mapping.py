"""
MITRE TTP Mapping Module — Strict mapper for tactics, techniques, and sub-techniques.
Updated to use map_org_ttp() as the main function.
"""
from __future__ import annotations
from typing import Dict, Any, List, Tuple
import json, re, collections, pandas as pd
from stix2 import MemoryStore, Filter, parse

# === Loader ===
def load_attack_store(path: str) -> MemoryStore:
    with open(path, "r") as f:
        raw = json.load(f)
    try:
        bundle = parse(raw, allow_custom=True)
        objs = bundle.objects
    except Exception:
        objs = raw.get("objects", raw)
    return MemoryStore(stix_data=objs)

def _get_attack_external_id(obj: Dict[str, Any], source_name: str = "mitre-attack"):
    for ref in obj.get("external_references", []) or []:
        src = ref.get("source_name")
        ext_id = ref.get("external_id")
        if src == source_name and ext_id:
            return ext_id
    return None

# === Normalization ===
def norm_key(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"\s*/\s*", "/", s)
    s = re.sub(r"\s*-\s*", "-", s)
    s = re.sub(r"\s+", " ", s)
    return s

# === Aliases ===
TECH_ALIAS_TO_CANON = {
    "signed binary proxy execution": "System Binary Proxy Execution",
    "network service scanning": "Network Service Discovery",
    "registry run keys": "T1547.001",
    "credential dumping": "OS Credential Dumping",
    "remote file copy": "Ingress Tool Transfer",
    "scripting": "Command and Scripting Interpreter",
    "data exfiltration over web service": "Exfiltration Over Web Service",
    "powershell": "Command and Scripting Interpreter",
    "bash": "Command and Scripting Interpreter",
    "wmic": "Windows Management Instrumentation",
}

TACTIC_ALIAS_TO_CANON = {
    "defense evasion": "Defense Evasion",
    "privilege escalation": "Privilege Escalation",
    "command and control": "Command and Control",
}

# === Build indices ===
def build_indices(stix_path: str):
    MS = load_attack_store(stix_path)
    techniques = MS.query([Filter("type", "=", "attack-pattern")])
    tactics = MS.query([Filter("type", "=", "x-mitre-tactic")])

    tac_name_to_id = {t.get("name"): _get_attack_external_id(t) for t in tactics if _get_attack_external_id(t)}

    tech_name_to_id_active = {}
    tech_id_to_name_active = {}
    parent_to_subs = collections.defaultdict(list)

    for t in techniques:
        tid = _get_attack_external_id(t)
        if not tid:
            continue
        name = t.get("name")
        is_sub = t.get("x_mitre_is_subtechnique", False)
        parent = tid.split(".")[0] if is_sub and "." in tid else None
        tech_name_to_id_active[name] = tid
        tech_id_to_name_active[tid] = name
        if parent:
            parent_to_subs[parent].append(tid)

    return {
        "tac_name_to_id": tac_name_to_id,
        "tech_name_to_id_active": tech_name_to_id_active,
        "tech_id_to_name_active": tech_id_to_name_active,
        "parent_to_subs": dict(parent_to_subs),
    }

# === Token matching ===
def match_token(token: str, idx: Dict[str, Any], granularity: str = "parent", expand_parent_to_subs: bool = False):
    if not token or not isinstance(token, str):
        return [(None, None, "unmatched")]

    raw = token.strip()
    raw_norm = norm_key(raw)

    if raw in idx["tac_name_to_id"]:
        return [(idx["tac_name_to_id"][raw], "tactic", "exact_name")]
    if raw in idx["tech_name_to_id_active"]:
        return [(idx["tech_name_to_id_active"][raw], "technique", "exact_name")]

    canon = TECH_ALIAS_TO_CANON.get(raw_norm)
    if canon and canon in idx["tech_name_to_id_active"]:
        return [(idx["tech_name_to_id_active"][canon], "technique", "alias")]

    return [(None, None, "unmatched")]

# === Org mapper ===
def map_org_ttp(org_csv: str, stix_path: str, ttp_col: str = "TTPs", exid_col: str = "ORGID", sep_regex: str = r"[;,\|]+", out_map_csv: str = "org_ttp_map.csv", out_unmatched_csv: str = "org_ttps_unmatched.csv"):
    idx = build_indices(stix_path)
    df = pd.read_csv(org_csv)
    if exid_col not in df.columns:
        raise ValueError(f"Missing `{exid_col}` in {org_csv}")

    splitter = re.compile(sep_regex)
    rows, unmatched = [], collections.Counter()

    for _, r in df.iterrows():
        orgid = r[exid_col]
        if pd.isna(r[ttp_col]):
            continue
        for t in splitter.split(str(r[ttp_col])):
            t = t.strip()
            if not t:
                continue
            matches = match_token(t, idx)
            for attack_id, label_type, how in matches:
                if not attack_id:
                    unmatched[t] += 1
                rows.append({"ORGID": orgid, "original_text": t, "attack_id": attack_id, "label_type": label_type, "match_type": how})

    pd.DataFrame(rows).to_csv(out_map_csv, index=False)
    pd.DataFrame([{"unmatched_text": k, "count": v} for k, v in unmatched.items()]).to_csv(out_unmatched_csv, index=False)
    print(f"[OK] wrote {out_map_csv} and {out_unmatched_csv}")
