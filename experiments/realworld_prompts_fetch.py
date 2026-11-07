"""Collect default multi-agent and shared-memory prompts from open-source agent frameworks.

    uv run python -I experiments/realworld_prompts_fetch.py          # fetch (pinned commits)
    uv run python -I experiments/realworld_prompts_fetch.py --offline  # re-extract from cache

Each spec names a repository, a pinned commit, a file and how the prompt text is taken from
it. Python sources are parsed with `ast` and string literals are read from the tree (nothing
from the fetched files is imported or executed); f-strings keep their placeholders as
`{expr}`. Markdown files are taken whole. YAML and notebook text is cut between markers.
Output: data/raw/framework_prompts/<project>__<name>.txt (gitignored, untrusted third-party
text), data/raw/framework_prompts/sources.csv, and a committed copy of the sources table at
results/realworld_prompts/sources.csv. The raw source files are cached under
data/raw/framework_prompts/_src/.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import textwrap
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "framework_prompts"
SRC = OUT / "_src"
RESULTS = ROOT / "results" / "realworld_prompts"

# repo -> (pinned commit, commit date UTC, stars on 2026-11-07)
REPOS = {
    "microsoft/autogen": ("d07c667a3239ea4ab7af62a53b0f4ff205a1f23d", "2026-10-28", 61737),
    "ag2ai/ag2": ("9e0595d529301975d5e9831105bee24730e65645", "2026-05-06 (tag v0.12.3)", 5051),
    "crewAIInc/crewAI": ("73f48320dbe6e2dc7a9bc54fefc6a13637b9485c", "2026-11-07", 60607),
    "langchain-ai/langgraph": ("23961cff61a42b52525f3b20b4094d8d2fba1744",
                               "2026-01-09 (last commit with the tutorials)", 44615),
    "langchain-ai/langgraph-supervisor-py": ("88859b34017ac3569bbd4a3092c7e77593a0a960",
                                             "2026-07-15", 1655),
    "langchain-ai/langgraph-swarm-py": ("8956b138bbb475b00c98d040176dbb8da667c612", "2026-10-28",
                                        1590),
    "FoundationAgents/MetaGPT": ("11cdf466d042aece04fc6cfd13b28e1a70341b1f", "2026-01-21", 71206),
    "camel-ai/camel": ("123f0c60f6efce7b420d6de2c3e046adb1f2cacc", "2026-11-02", 18016),
    "OpenBMB/ChatDev": ("464a681ba7bda869db97a50c5fa707c9a0fb57a0", "2026-10-30", 34703),
    "openai/openai-agents-python": ("27c75c5f9c5fd957d28c04fb7e9eeacaad4630a9", "2026-11-06",
                                    30544),
    "openai/swarm": ("6af0b4caf37dca4526dfd98e9fbd8ce36e7eeb22", "2026-04-15", 22112),
    "kyegomez/swarms": ("747070bc5cf3383379c886c362258f5f296094ce", "2026-11-07", 7361),
    "letta-ai/letta": ("1131535716e8a31c9a437f8695e25ac98f203a24",
                       "2026-05-14 (tag 0.16.8; source later moved to letta-code)", 25376),
    "OpenBMB/AgentVerse": ("f90c4bd9680fdd3bcff8c52c9170911a59b23478", "2024-09-09", 5167),
    "huggingface/smolagents": ("96f33faaf028479119ec8d34507b47694cf14e34", "2026-10-06", 30252),
    "google/adk-python": ("dd7161f7a2b88cd7ae482cc2acd8423675ac027c", "2026-11-07", 22133),
    "google/adk-recipes": ("64f61c6ac2653c0ad3824f0108faac055348cb5f",
                           "2026-11-06 (formerly google/adk-samples)", 10588),
    "agno-agi/agno": ("f45ccee83dcd332c6a2cf46b0a0add3e7f7d3e68", "2026-11-06", 43092),
    "strands-agents/harness-sdk": ("3d32c2269609f235bea121cc2afcb14afeefd670",
                                   "2026-11-06 (formerly strands-agents/sdk-python)", 9069),
    "langchain-ai/deepagents": ("9c5d330893e0580ece0422511248d1048373cd7f", "2026-11-07", 31065),
    "VRSEN/agency-swarm": ("1bec92f0fc6e8485e53847c6f4a943a847058bff", "2026-11-02", 4630),
    "run-llama/llama_index": ("444fdf0684707de62ebb447d658dba137dd9cb99", "2026-11-03", 52840),
    "microsoft/TaskWeaver": ("d44ddef23f90059fb17999d3095db4240e98f955", "2026-03-23", 6160),
    "anthropics/claude-cookbooks": ("ce8b2629efe8b24048b127b4c79af1e9cbab93d4", "2026-11-04",
                                    54369),
    "agent0ai/agent-zero": ("ba9b67df0f54ac78106ae85c552ef5d2458f7c9a", "2026-11-01", 19786),
    "a2aproject/a2a-samples": ("6603ba3f2c31a7ef33e70b9d8b5b5f8be42ac9a3", "2026-08-04", 1822),
    "openai/codex": ("57a808a3920f05f2d4b9eab2d0493e1df136ae6b", "2026-11-07", 133566),
}

AG = "python/packages/autogen-agentchat/src/autogen_agentchat"
LGT = "docs/docs/tutorials/multi_agent"
# (id, project, repo, path, method, args, what)
SPECS = [
    ("autogen__selector_group_chat", "AutoGen", "microsoft/autogen",
     f"{AG}/teams/_group_chat/_selector_group_chat.py", "kwdefault", ["selector_prompt"],
     "SelectorGroupChat default selector prompt"),
    ("autogen__assistant_agent_default", "AutoGen", "microsoft/autogen",
     f"{AG}/agents/_assistant_agent.py", "kwdefault", ["system_message"],
     "AssistantAgent default system message"),
    ("autogen__society_of_mind", "AutoGen", "microsoft/autogen",
     f"{AG}/agents/_society_of_mind_agent.py", "const",
     ["DEFAULT_INSTRUCTION", "DEFAULT_RESPONSE_PROMPT"],
     "SocietyOfMindAgent default instruction and response prompt"),
    ("magentic_one__orchestrator", "Magentic-One", "microsoft/autogen",
     f"{AG}/teams/_group_chat/_magentic_one/_prompts.py", "const",
     ["ORCHESTRATOR_TASK_LEDGER_FULL_PROMPT", "ORCHESTRATOR_PROGRESS_LEDGER_PROMPT",
      "ORCHESTRATOR_TASK_LEDGER_PLAN_UPDATE_PROMPT"],
     "Magentic-One orchestrator ledger prompts (copied unchanged in microsoft/agent-framework)"),
    ("ag2__groupchat_select_speaker", "AG2", "ag2ai/ag2", "autogen/agentchat/groupchat.py",
     "const", ["select_speaker_message_template"], "GroupChat default speaker-selection message"),
    ("crewai__hierarchical_manager", "CrewAI", "crewAIInc/crewAI",
     "lib/crewai/src/crewai/translations/en.json", "json",
     ["hierarchical_manager_agent.role", "hierarchical_manager_agent.goal",
      "hierarchical_manager_agent.backstory"], "default manager agent of hierarchical crews"),
    ("crewai__delegation_tools", "CrewAI", "crewAIInc/crewAI",
     "lib/crewai/src/crewai/translations/en.json", "json",
     ["tools.delegate_work", "tools.ask_question"], "delegate-work and ask-question tools"),
    ("crewai__memory_tools", "CrewAI", "crewAIInc/crewAI",
     "lib/crewai/src/crewai/translations/en.json", "json",
     ["tools.save_to_memory", "tools.recall_memory"], "memory save and recall tools"),
    ("crewai__crew_member", "CrewAI", "crewAIInc/crewAI",
     "lib/crewai/src/crewai/translations/en.json", "json",
     ["slices.conversation_history_instruction"], "crew member framing slice"),
    ("langgraph__collaboration_base", "LangGraph tutorials", "langchain-ai/langgraph",
     f"{LGT}/multi-agent-collaboration.ipynb", "nb_return", ["make_system_prompt"],
     "multi-agent collaboration tutorial, shared system prompt"),
    ("langgraph__hier_supervisor", "LangGraph tutorials", "langchain-ai/langgraph",
     f"{LGT}/hierarchical_agent_teams.ipynb", "nb_const", ["system_prompt"],
     "hierarchical agent teams tutorial, supervisor"),
    ("langgraph__hier_doc_writer", "LangGraph tutorials", "langchain-ai/langgraph",
     f"{LGT}/hierarchical_agent_teams.ipynb", "nb_callkw", ["prompt", 0],
     "hierarchical agent teams tutorial, document writer (tools write a shared directory)"),
    ("langgraph__hier_note_taker", "LangGraph tutorials", "langchain-ai/langgraph",
     f"{LGT}/hierarchical_agent_teams.ipynb", "nb_callkw", ["prompt", 1],
     "hierarchical agent teams tutorial, note taker (tools write a shared directory)"),
    ("langgraph_supervisor__handoff_tool", "langgraph-supervisor",
     "langchain-ai/langgraph-supervisor-py", "langgraph_supervisor/handoff.py", "assign",
     ["description", 0], "default handoff tool description"),
    ("langgraph_supervisor__readme_supervisor", "langgraph-supervisor",
     "langchain-ai/langgraph-supervisor-py", "README.md", "md_pyexpr",
     ['"You are a team supervisor managing a research expert and a math expert. "'],
     "README quickstart supervisor prompt"),
    ("langgraph_swarm__handoff_tool", "langgraph-swarm", "langchain-ai/langgraph-swarm-py",
     "langgraph_swarm/handoff.py", "assign", ["description", 0],
     "default handoff tool description"),
    ("langgraph_swarm__researcher", "langgraph-swarm", "langchain-ai/langgraph-swarm-py",
     "examples/research/src/agent/prompts.py", "const", ["researcher_prompt"],
     "research example, researcher agent"),
    ("metagpt__team_leader", "MetaGPT", "FoundationAgents/MetaGPT",
     "metagpt/prompts/di/team_leader.py", "const", ["TL_INSTRUCTION"], "TeamLeader instruction"),
    ("metagpt__role_prefix", "MetaGPT", "FoundationAgents/MetaGPT", "metagpt/roles/role.py",
     "const", ["PREFIX_TEMPLATE", "env_desc"], "Role system prefix and environment sentence"),
    ("camel__ai_society_assistant", "CAMEL", "camel-ai/camel", "camel/prompts/ai_society.py",
     "const", ["ASSISTANT_PROMPT"], "AI-society role-playing assistant rules"),
    ("camel__workforce_process_task", "CAMEL", "camel-ai/camel",
     "camel/societies/workforce/prompts.py", "const", ["PROCESS_TASK_PROMPT"],
     "Workforce worker task prompt"),
    ("chatdev__common_prompt", "ChatDev", "OpenBMB/ChatDev", "yaml_instance/ChatDev_v1.yaml",
     "yaml_plain", ["  COMMON_PROMPT:"], "company framing shared by every role"),
    ("chatdev__code_reviewer", "ChatDev", "OpenBMB/ChatDev", "yaml_instance/ChatDev_v1.yaml",
     "yaml_block", ["role: |-", "You are Code Reviewer."], "Code Reviewer role (file tools)"),
    ("openai_agents__handoff_prefix", "OpenAI Agents SDK", "openai/openai-agents-python",
     "src/agents/extensions/handoff_prompt.py", "const", ["RECOMMENDED_PROMPT_PREFIX"],
     "recommended prefix for agents that hand off"),
    ("openai_agents__sandbox_memory", "OpenAI Agents SDK", "openai/openai-agents-python",
     "src/agents/sandbox/memory/prompts/memory_read_prompt.md", "whole", [],
     "sandbox memory read prompt"),
    ("openai_swarm__triage", "OpenAI Swarm", "openai/swarm", "examples/triage_agent/agents.py",
     "callkw", ["instructions", 0], "triage example, triage agent"),
    ("swarms__multi_agent_collab", "swarms (kyegomez)", "kyegomez/swarms",
     "swarms/prompts/multi_agent_collab_prompt.py", "const", ["MULTI_AGENT_COLLAB_PROMPT"],
     "multi-agent collaboration system prompt"),
    ("swarms__groupchat_decide", "swarms (kyegomez)", "kyegomez/swarms",
     "swarms/prompts/groupchat_prompt.py", "const", ["GROUPCHAT_DECIDE_PROMPT"],
     "group chat decision prompt"),
    ("letta__sleeptime_memory", "Letta (MemGPT)", "letta-ai/letta",
     "letta/prompts/system_prompts/sleeptime_v2.py", "const", ["PROMPT"],
     "sleep-time memory agent (edits the primary agent's memory blocks)"),
    ("letta__send_message_to_agent", "Letta (MemGPT)", "letta-ai/letta",
     "letta/functions/function_sets/multi_agent.py", "docstring",
     ["send_message_to_agent_and_wait_for_reply"], "multi-agent messaging tool"),
    ("agentverse__sde_team", "AgentVerse", "OpenBMB/AgentVerse",
     "agentverse/tasks/simulation/sde_team/sde_team_2players/config.yaml", "yaml_squote",
     ["prompt_template: '"], "SDE team code writer template"),
    ("smolagents__team_members", "smolagents", "huggingface/smolagents",
     "src/smolagents/prompts/code_agent.yaml", "span",
     ["You can also give tasks to team members.",
      "Here is a list of the team members that you can call:"],
     "CodeAgent system prompt, managed-agents section"),
    ("smolagents__managed_agent_task", "smolagents", "huggingface/smolagents",
     "src/smolagents/prompts/code_agent.yaml", "yaml_block", ["  task: |-", None],
     "task prompt given to a managed agent"),
    ("adk__transfer_instruction", "Google ADK", "google/adk-python",
     "src/google/adk/flows/llm_flows/extensions/_agent_transfer.py", "return",
     ["_build_transfer_instruction_body"], "agent-transfer instruction"),
    ("adk_samples__financial_coordinator", "Google ADK samples", "google/adk-recipes",
     "contrib/python/financial-advisor/app/prompt.py", "const", ["FINANCIAL_COORDINATOR_PROMPT"],
     "financial-advisor sample, coordinator"),
    ("adk_samples__trading_analyst", "Google ADK samples", "google/adk-recipes",
     "contrib/python/financial-advisor/app/sub_agents/trading_analyst/prompt.py", "const",
     ["TRADING_ANALYST_PROMPT"], "financial-advisor sample, trading analyst sub-agent"),
    ("agno__team_tasks_mode", "Agno", "agno-agi/agno", "libs/agno/agno/team/_messages.py",
     "agno_mode", [0], "team leader, tasks mode (opening + delegation block)"),
    ("agno__team_coordinate_mode", "Agno", "agno-agi/agno", "libs/agno/agno/team/_messages.py",
     "agno_mode", [3], "team leader, coordinate mode, the default (opening + delegation block)"),
    ("strands__swarm_node_input", "Strands Agents", "strands-agents/harness-sdk",
     "strands-py/src/strands/multiagent/swarm.py", "strands", [],
     "Swarm: text given to each agent, and the handoff tool docstring"),
    ("deepagents__task_tool", "deepagents", "langchain-ai/deepagents",
     "libs/deepagents/deepagents/middleware/subagents.py", "const",
     ["TASK_TOOL_DESCRIPTION", "DEFAULT_SUBAGENT_PROMPT"],
     "task tool description and default subagent prompt"),
    ("deepagents__memory", "deepagents", "langchain-ai/deepagents",
     "libs/deepagents/deepagents/middleware/memory.py", "const", ["MEMORY_SYSTEM_PROMPT"],
     "memory middleware system prompt (memory files on a shared filesystem)"),
    ("agency_swarm__send_message", "Agency Swarm", "VRSEN/agency-swarm",
     "src/agency_swarm/tools/send_message.py", "docstring", ["SendMessage"],
     "SendMessage tool description"),
    ("llama_index__agent_workflow", "LlamaIndex", "run-llama/llama_index",
     "llama-index-core/llama_index/core/agent/workflow/prompts.py", "const",
     ["DEFAULT_HANDOFF_PROMPT", "DEFAULT_STATE_PROMPT"],
     "AgentWorkflow handoff tool and shared-state prompt"),
    ("taskweaver__planner", "TaskWeaver", "microsoft/TaskWeaver",
     "taskweaver/planner/planner_prompt.yaml", "yaml_block", ["instruction_template: |-", None],
     "Planner instruction template"),
    ("claude_cookbook__research_lead", "Anthropic cookbook", "anthropics/claude-cookbooks",
     "patterns/agents/prompts/research_lead_agent.md", "whole", [],
     "multi-agent research system, lead agent"),
    ("claude_cookbook__research_subagent", "Anthropic cookbook", "anthropics/claude-cookbooks",
     "patterns/agents/prompts/research_subagent.md", "whole", [],
     "multi-agent research system, subagent"),
    ("agent_zero__call_subordinate", "Agent Zero", "agent0ai/agent-zero",
     "prompts/agent.system.tool.call_sub.md", "whole", [], "call_subordinate tool prompt"),
    ("agent_zero__memory_tool", "Agent Zero", "agent0ai/agent-zero",
     "plugins/_memory/prompts/agent.system.tool.memory.md", "whole", [], "memory tool prompt"),
    ("a2a__routing_agent", "A2A samples", "a2aproject/a2a-samples",
     "samples/python/agents/airbnb_planner_multiagent/host_agent/routing_agent.py", "return",
     ["root_instruction"], "multi-agent sample, host routing agent"),
    ("codex__collab_experimental", "OpenAI Codex", "openai/codex",
     "codex-rs/core/templates/collab/experimental_prompt.md", "whole", [],
     "multi-agent (collab) section"),
    ("codex__orchestrator", "OpenAI Codex", "openai/codex",
     "codex-rs/core/templates/agents/orchestrator.md", "whole", [],
     "orchestrator agent template"),
]


# ---- fetching ---------------------------------------------------------------------------------
def raw_url(repo: str, path: str) -> str:
    return f"https://raw.githubusercontent.com/{repo}/{REPOS[repo][0]}/{path}"


def blob_url(repo: str, path: str) -> str:
    return f"https://github.com/{repo}/blob/{REPOS[repo][0]}/{path}"


def fetch(repo: str, path: str, offline: bool) -> str:
    dst = SRC / repo.replace("/", "_") / path.replace("/", "__")
    if not dst.exists():
        if offline:
            raise FileNotFoundError(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(raw_url(repo, path), timeout=60) as r:
            dst.write_bytes(r.read())
    return dst.read_text(encoding="utf-8")


# ---- string extraction from Python trees (no execution) ---------------------------------------
def render(node: ast.AST) -> str | None:
    """A string literal, an implicit or + concatenation of them, an f-string (placeholders kept
    as {expr}), or a call whose first argument is one of these (TextPrompt("..."))."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        out = []
        for v in node.values:
            if isinstance(v, ast.Constant):
                out.append(v.value)
            else:
                out.append("{" + ast.unparse(v.value) + "}")
        return "".join(out)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        a, b = render(node.left), render(node.right)
        return None if a is None or b is None else a + b
    if isinstance(node, ast.Call) and node.args:
        return render(node.args[0])
    return None


def py_const(tree: ast.AST, name: str) -> str:
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign):
            if any(isinstance(t, ast.Name) and t.id == name for t in n.targets):
                s = render(n.value)
                if s is not None:
                    return s
        if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) \
                and n.target.id == name and n.value is not None:
            s = render(n.value)
            if s is not None:
                return s
    raise KeyError(name)


def py_kwdefault(tree: ast.AST, name: str) -> str:
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            a = n.args
            pos = a.posonlyargs + a.args
            pairs = list(zip(pos[len(pos) - len(a.defaults):], a.defaults)) + \
                [(k, d) for k, d in zip(a.kwonlyargs, a.kw_defaults) if d is not None]
            for arg, d in pairs:
                if arg.arg == name:
                    s = render(d)
                    if s is not None:
                        return s
    raise KeyError(name)


def py_return(tree: ast.AST, func: str) -> str:
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == func:
            for m in ast.walk(n):
                if isinstance(m, ast.Return) and m.value is not None:
                    s = render(m.value)
                    if s is not None:
                        return s
    raise KeyError(func)


def py_callkw(tree: ast.AST, kw: str, nth: int) -> str:
    found = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            for k in n.keywords:
                if k.arg == kw:
                    s = render(k.value)
                    if s is not None:
                        found.append((n.lineno, n.col_offset, s))
    found.sort()
    return found[nth][2]


def py_assign_nth(tree: ast.AST, name: str, nth: int) -> str:
    found = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name
                                             for t in n.targets):
            s = render(n.value)
            if s is not None:
                found.append((n.lineno, s))
    found.sort()
    return found[nth][1]


def py_docstring(tree: ast.AST, name: str) -> str:
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.name == name:
            d = ast.get_docstring(n)
            if d:
                return d
    raise KeyError(name)


def augassigns(tree: ast.AST, func: str, target: str) -> list[str]:
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == func:
            out = []
            for m in ast.walk(n):
                if isinstance(m, ast.AugAssign) and isinstance(m.target, ast.Name) \
                        and m.target.id == target:
                    s = render(m.value)
                    if s is not None:
                        out.append((m.lineno, s))
            return [s for _, s in sorted(out)]
    raise KeyError(func)


def nb_trees(text: str) -> list[ast.AST]:
    out = []
    for c in json.loads(text)["cells"]:
        if c["cell_type"] != "code":
            continue
        try:
            out.append(ast.parse("".join(c["source"])))
        except SyntaxError:  # cells with shell magics
            continue
    return out


def first_ok(fn, trees, *args):
    for t in trees:
        try:
            return fn(t, *args)
        except (KeyError, IndexError):
            continue
    raise KeyError(args)


# ---- text extraction --------------------------------------------------------------------------
def yaml_block(text: str, key: str, contains: str | None) -> str:
    """A `key: |-` literal block: the following lines indented deeper than the key, dedented,
    trailing newline stripped. With `contains`, the first block whose text contains it."""
    lines = text.split("\n")
    for i, ln in enumerate(lines):
        if ln.strip() == key.strip() and ln.startswith(key.split(key.strip())[0]):
            ind = len(ln) - len(ln.lstrip())
            body = []
            for nxt in lines[i + 1:]:
                if nxt.strip() and len(nxt) - len(nxt.lstrip()) <= ind:
                    break
                body.append(nxt)
            s = textwrap.dedent("\n".join(body)).rstrip("\n")
            if contains is None or contains in s:
                return s
    raise KeyError(key)


def yaml_plain(text: str, key: str) -> str:
    """A multi-line plain scalar: continuation lines folded with single spaces."""
    lines = text.split("\n")
    for i, ln in enumerate(lines):
        if ln.startswith(key):
            ind = len(ln) - len(ln.lstrip())
            parts = [ln[len(key):].strip()]
            for nxt in lines[i + 1:]:
                if not nxt.strip() or len(nxt) - len(nxt.lstrip()) <= ind:
                    break
                parts.append(nxt.strip())
            return " ".join(parts)
    raise KeyError(key)


def yaml_squote(text: str, key: str) -> str:
    """A single-quoted flow scalar over several lines: a single line break folds to a space,
    n empty lines give n newlines; '' is a quote."""
    start = text.index(key) + len(key)
    end = start
    while True:
        end = text.index("'", end)
        if text[end:end + 2] == "''":
            end += 2
            continue
        break
    raw = text[start:end].split("\n")
    out, blank = raw[0].strip(), 0
    for ln in raw[1:]:
        if not ln.strip():
            blank += 1
            continue
        out += ("\n" * blank if blank else " ") + ln.strip()
        blank = 0
    return out.replace("''", "'")


def span(text: str, start: str, end: str) -> str:
    i = text.index(start)
    j = text.index(end, i) + len(end)
    i = text.rindex("\n", 0, i) + 1  # from the start of the line, so dedent sees the indent
    return textwrap.dedent(text[i:j])


def md_pyexpr(text: str, marker: str) -> str:
    """The parenthesised Python string expression that contains `marker` (README examples)."""
    i = text.index(marker)
    a = text.rindex("(", 0, i)
    depth, j = 0, a
    while True:
        if text[j] == "(":
            depth += 1
        elif text[j] == ")":
            depth -= 1
            if depth == 0:
                break
        j += 1
    return ast.literal_eval(text[a:j + 1])


def extract(text: str, method: str, args: list) -> str:
    if method == "whole":
        return text
    if method == "json":
        d = json.loads(text)
        parts = []
        for key in args:
            v = d
            for k in key.split("."):
                v = v[k]
            parts.append(v)
        return "\n\n".join(parts)
    if method == "span":
        return span(text, *args)
    if method == "md_pyexpr":
        return md_pyexpr(text, args[0])
    if method == "yaml_block":
        return yaml_block(text, args[0], args[1])
    if method == "yaml_plain":
        return yaml_plain(text, args[0])
    if method == "yaml_squote":
        return yaml_squote(text, args[0])
    if method.startswith("nb_"):
        trees = nb_trees(text)
        if method == "nb_return":
            return first_ok(py_return, trees, args[0])
        if method == "nb_const":
            return first_ok(py_const, trees, args[0])
        if method == "nb_callkw":
            found = []
            for t in trees:
                for k in range(50):
                    try:
                        found.append(py_callkw(t, args[0], k))
                    except IndexError:
                        break
            return found[args[1]]
    tree = ast.parse(text)
    if method == "const":
        return "\n\n".join(py_const(tree, n) for n in args)
    if method == "kwdefault":
        return py_kwdefault(tree, args[0])
    if method == "return":
        return py_return(tree, args[0])
    if method == "callkw":
        return py_callkw(tree, args[0], args[1])
    if method == "assign":
        return py_assign_nth(tree, args[0], args[1])
    if method == "docstring":
        return py_docstring(tree, args[0])
    if method == "agno_mode":
        opening = py_return(tree, "_get_opening_prompt")
        blocks = augassigns(tree, "_get_mode_instructions", "content")
        head = py_assign_nth(tree, "content", 0)
        return opening + head + blocks[args[0]]
    if method == "strands":
        pieces = augassigns(tree, "_build_node_input", "context_text")
        doc = py_docstring(tree, "handoff_to_agent")
        keep = [p for p in pieces if "Multi-modal" not in p]
        return "".join(keep) + "\n\n[handoff_to_agent tool]\n" + doc
    raise ValueError(method)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--offline", action="store_true")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    rows = []
    for pid, project, repo, path, method, args, what in SPECS:
        text = fetch(repo, path, a.offline)
        prompt = extract(text, method, args).strip("\n") + "\n"
        (OUT / f"{pid}.txt").write_text(prompt, encoding="utf-8")
        sha, date, stars = REPOS[repo]
        rows.append({"file": f"{pid}.txt", "project": project, "repo": repo, "path": path,
                     "commit": sha, "commit_date": date, "stars_2026_11_07": stars or "",
                     "what": what, "extract": f"{method} {json.dumps(args)}",
                     "chars": len(prompt), "url": blob_url(repo, path)})
        print(f"{pid:42s} {len(prompt):6d} chars  {repo}")
    for dst in (OUT / "sources.csv", RESULTS / "sources.csv"):
        with dst.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print(f"{len(rows)} prompts from {len({r['project'] for r in rows})} projects, "
          f"{len({r['repo'] for r in rows})} repositories")


if __name__ == "__main__":
    main()
