# AACP Coordination with CrewAI

**Typed, deterministic agent coordination for CrewAI workflows.**

This example shows how AACP (Agent Action Compression Protocol)
replaces natural language task descriptions in CrewAI with compact,
validated coordination packets -- eliminating coordination LLM calls
entirely for known workflow types.

## The problem

In a standard CrewAI workflow the orchestrator writes natural language
task descriptions to coordinate specialist agents:

```
"Please retrieve all active employee salary records for March 2026.
 Include each employee's department, cost centre, base salary, any
 changes applied this month, and pension contribution rates. Return
 as a JSON array."
```

This varies on every run, provides no machine-readable structure,
and cannot be validated before transmission.

## The AACP approach

AACP replaces that with a typed, pipe-delimited packet:

```
FETCH|HR|return:HR-Agent|p:1|aacp:1.4|res:emp_salary|period:2026-03|filter:status=active|fmt:json
```

Identical on every run. Validates against the AACP v1.4 schema before
transmission. Machine-readable as an audit record without post-processing.

## Measured result

Benchmarked on a five-workflow, 59-hop department day scenario:

| | Standard CrewAI | With AACP |
|---|---|---|
| Coordination LLM calls | 59 | 0 |
| Coordination cost | $0.0008 | $0.0000 |
| Total cost reduction | — | 30% |
| Coordination deterministic | NO | YES |
| Schema validated | NO | YES |
| Audit trail structured | NO | YES |

CrewAI shows a 30% saving because its role-based task descriptions
are verbose by default. AACP replaces the task description layer
while leaving agent roles, goals, and backstories intact.

## Install

```bash
pip install aacp-crewai aacp crewai langchain-openai
export OPENAI_API_KEY=sk-...
```

## Run the example

```bash
# Payroll workflow comparison -- no external data needed
python3 example.py

# With verbose output
python3 example.py --verbose
```

## How it works

```python
from aacp_crewai import AACPPacketBus
from aacp_crewai.agents import HRAgent, FinanceAgent
from aacp.encoders.workflows.payroll import PayrollEncoder

enc = PayrollEncoder()
bus = AACPPacketBus(workflow="payroll", model="gpt-4o-mini")

hr_agent      = HRAgent(model="gpt-4o-mini")
finance_agent = FinanceAgent(model="gpt-4o-mini")

# Typed packet -- $0.00 encoding cost, identical every run
result = bus.dispatch(
    "ORCHESTRATOR",
    hr_agent,
    enc.fetch_employees("2026-03").packet,
    {"employees": employee_data},
)
```

## Links

- Protocol spec: https://aacp.dev
- PyPI package: https://pypi.org/project/aacp-crewai/
- GitHub: https://github.com/MackayAndrew/aacp-crewai
- Community rules: https://registry.aacp.dev (241 pre-validated rules)
- IETF Draft: https://datatracker.ietf.org/doc/draft-mackay-aacp/
- Licence: MIT
