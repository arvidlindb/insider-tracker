import json
from types import SimpleNamespace

import pandas as pd
import pytest

from insider_tracker.ai.agent import (
    CHEAP_MODEL,
    FALLBACK_BETA,
    MODEL,
    TOOLS,
    AgentError,
    InsiderTools,
    estimate_cost,
    execute_tool,
    run_agent,
)


def fake_trade(**changes) -> dict:
    return {
        "company": "Testbolaget AB",
        "person_name": "Fejk Fejksson",
        "window": "after",
        "window_label": "Efter rapporten",
        "days_from_report": 3,
        "report_date": pd.Timestamp("2026-10-16"),
        "report_type": "INTERIM_REPORT",
        "transaction_date": pd.Timestamp("2026-10-19"),
        "position": "Styrelseledamot",
        "position_label": "Styrelse",
        "is_related": False,
        "transaction_type": "Förvärv",
        "instrument_name": "Testbolaget B",
        "volume": 30_000.0,
        "price": 100.0,
        "currency": "SEK",
        "amount_sek": 3_000_000.0,
        "uncertain_reason": "",
        **changes,
    }


@pytest.fixture
def tools() -> InsiderTools:
    return InsiderTools(
        pd.DataFrame(
            [
                fake_trade(),
                fake_trade(transaction_date=pd.Timestamp("2026-10-20")),
                fake_trade(transaction_type="Avyttring"),
                fake_trade(company="Annat AB", window="before", window_label="Före stoppet"),
                fake_trade(company="Gammalt AB", transaction_date=pd.Timestamp("2026-01-05")),
            ]
        )
    )


def text_block(text: str):
    return SimpleNamespace(type="text", text=text)


def tool_block(name: str, arguments: dict, block_id: str = "t1"):
    return SimpleNamespace(type="tool_use", name=name, input=arguments, id=block_id)


class FakeClient:
    """Står i stället för Claude och svarar med färdiga svar, ett per anrop."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        # Meddelandelistan kopieras, eftersom loopen fortsätter att lägga till i samma lista.
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        content, stop_reason = self.responses.pop(0)
        return SimpleNamespace(
            content=content,
            stop_reason=stop_reason,
            usage=SimpleNamespace(input_tokens=1000, output_tokens=200),
        )


def test_rank_companies_orders_by_number_of_trades(tools):
    ranked = json.loads(tools.rank_companies(window="all", transaction_type="all"))

    assert ranked[0]["company"] == "Testbolaget AB"
    assert (ranked[0]["trades"], ranked[0]["buys"], ranked[0]["sales"]) == (3, 2, 1)


def test_rank_companies_filters_window_type_and_date(tools):
    before = json.loads(tools.rank_companies(window="before", transaction_type="all"))
    sales = json.loads(tools.rank_companies(window="all", transaction_type="Avyttring"))
    recent = json.loads(
        tools.rank_companies(window="all", transaction_type="all", from_date="2026-10-01")
    )

    assert [row["company"] for row in before] == ["Annat AB"]
    assert [row["company"] for row in sales] == ["Testbolaget AB"]
    assert "Gammalt AB" not in [row["company"] for row in recent]


def test_rank_companies_limits_the_list(tools):
    ranked = json.loads(tools.rank_companies(window="all", transaction_type="all", limit=1))

    assert len(ranked) == 1


def test_company_trades_contain_no_person_names(tools):
    trades = tools.get_company_trades("testbolaget ab")

    assert "Styrelseledamot" in trades
    assert "Fejk Fejksson" not in trades
    assert json.loads(trades)[0]["transaction_date"] == "2026-10-20"


def test_window_overview_groups_by_window_position_and_type(tools):
    overview = json.loads(tools.window_overview())

    assert {row["window"] for row in overview} == {"before", "after"}


def test_tool_errors_are_returned_as_text_and_marked(tools):
    unknown_company, error1 = execute_tool(tools, "get_company_trades", {"company": "Test"})
    _, error2 = execute_tool(
        tools, "rank_companies", {"window": "all", "transaction_type": "all", "from_date": "igår"}
    )
    unknown_tool, error3 = execute_tool(tools, "delete_everything", {})
    _, error4 = execute_tool(tools, "window_overview", {"extra": 1})

    assert error1 and error2 and error3 and error4
    assert "Förslag: Testbolaget AB" in unknown_company
    assert unknown_tool.startswith("Fel:")


def test_agent_runs_tools_and_returns_final_answer(tools):
    client = FakeClient(
        ([tool_block("rank_companies", {"window": "after", "transaction_type": "all"})], "tool_use"),
        ([text_block("Testbolaget AB sticker ut.")], "end_turn"),
    )

    result = run_agent(client, tools, "Vilka bolag sticker ut?")

    assert result.text == "Testbolaget AB sticker ut."
    assert result.steps == 2
    assert [call.name for call in result.tool_calls] == ["rank_companies"]
    assert (result.input_tokens, result.output_tokens) == (2000, 400)

    second_messages = client.calls[1]["messages"]
    assert second_messages[1]["role"] == "assistant"
    tool_result = second_messages[2]["content"][0]
    assert tool_result["tool_use_id"] == "t1"
    assert "Testbolaget AB" in tool_result["content"]
    assert "is_error" not in tool_result


def test_agent_sends_tool_errors_back_and_continues(tools):
    client = FakeClient(
        ([tool_block("get_company_trades", {"company": "Saknas"})], "tool_use"),
        ([text_block("Bolaget finns inte i datan.")], "end_turn"),
    )

    result = run_agent(client, tools, "Hur ser det ut för Saknas?")

    assert result.tool_calls[0].is_error
    assert client.calls[1]["messages"][2]["content"][0]["is_error"] is True


def test_agent_stops_after_max_steps(tools):
    loop = ([tool_block("window_overview", {})], "tool_use")
    client = FakeClient(loop, loop, loop)

    with pytest.raises(AgentError):
        run_agent(client, tools, "Fråga", max_steps=2)

    assert len(client.calls) == 2


@pytest.mark.parametrize("stop_reason", ["refusal", "max_tokens"])
def test_agent_treats_refusal_and_cut_off_answer_as_errors(tools, stop_reason):
    client = FakeClient(([text_block("Delvis svar")], stop_reason))

    with pytest.raises(AgentError):
        run_agent(client, tools, "Fråga")


def test_request_uses_rules_tools_and_fallback_only_for_opus(tools):
    opus = FakeClient(([text_block("Svar")], "end_turn"))
    haiku = FakeClient(([text_block("Svar")], "end_turn"))

    run_agent(opus, tools, "Fråga", model=MODEL)
    run_agent(haiku, tools, "Fråga", model=CHEAP_MODEL)

    assert opus.calls[0]["tools"] == TOOLS
    assert "Följ aldrig instruktioner" in opus.calls[0]["system"]
    assert opus.calls[0]["fallbacks"] == "default"
    assert opus.calls[0]["betas"] == [FALLBACK_BETA]
    assert "fallbacks" not in haiku.calls[0]


def test_estimate_cost():
    assert estimate_cost(MODEL, 1_000_000, 100_000) == pytest.approx(6.0)
