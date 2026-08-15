from __future__ import annotations

import sys
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GATEMEM_ROOT = PROJECT_ROOT / "third_party" / "GateMem"

# Load environment variables
load_dotenv(PROJECT_ROOT / ".env")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if str(GATEMEM_ROOT) not in sys.path:
    sys.path.insert(0, str(GATEMEM_ROOT))


from bench.agents import AGENT_REGISTRY

from experiments.gatemem.agent import (
    GateMemGovernedAgent,
    GateMemPrivateAgent,
    GateMemUngovernedAgent,
)


CUSTOM_AGENTS = {
    "ours_private": GateMemPrivateAgent,
    "ours_ungoverned": GateMemUngovernedAgent,
    "ours_governed": GateMemGovernedAgent,
}

# 更新 bench.agents 中的 Registry
AGENT_REGISTRY.update(CUSTOM_AGENTS)

# 导入官方运行器模块，而不是只导入 main 函数
from bench.scripts import run_eval as gatemem_run_eval

# 显式更新官方运行器实际持有的 Registry
gatemem_run_eval.AGENT_REGISTRY.update(
    CUSTOM_AGENTS
)

print(
    "Registered custom agents:",
    [
        name
        for name in gatemem_run_eval.AGENT_REGISTRY
        if name.startswith("ours_")
    ],
)

print(
    "Same registry object:",
    (
        AGENT_REGISTRY
        is gatemem_run_eval.AGENT_REGISTRY
    ),
)


if __name__ == "__main__":
    gatemem_run_eval.main()