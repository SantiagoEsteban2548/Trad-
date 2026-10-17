# tests/test_regime_council.py
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock
import pytest
from pydantic import ValidationError

from backend.regime_council.gemini_debate import (
    GeminiRegimeCouncil,
    MicrostructureSummary15m,
    RegimeDecision,
    Symbol15mSummary,
)


@pytest.fixture
def sample_microstructure_summary() -> MicrostructureSummary15m:
    return MicrostructureSummary15m(
        timestamp_utc="2026-10-13T12:00:00Z",
        portfolio_equity_usd=10_000.0,
        current_drawdown_pct=-0.005,
        symbols_data=[
            Symbol15mSummary(
                symbol="BTC/USDT:USDT",
                realized_vol_5m=0.35,
                avg_l2_spread_bps=1.2,
                mean_obi_5_levels=0.08,
                cumulative_trade_volume_usd=5_200_000.0,
                liquidation_clusters_count=1,
                net_liquidation_volume_usd=-25_000.0,
                projected_funding_rate_8h=0.00015,
                basis_bps=2.1,
            ),
            Symbol15mSummary(
                symbol="SOL/USDT:USDT",
                realized_vol_5m=0.65,
                avg_l2_spread_bps=3.8,
                mean_obi_5_levels=-0.45,
                cumulative_trade_volume_usd=1_800_000.0,
                liquidation_clusters_count=4,
                net_liquidation_volume_usd=-450_000.0,
                projected_funding_rate_8h=-0.00030,
                basis_bps=-8.5,
            ),
        ],
    )


def test_regime_decision_schema_enforces_strict_bounds() -> None:
    valid_data = {
        "market_regime": "LOW_VOL_RANGE",
        "avellaneda_gamma": 0.25,
        "order_book_intensity_kappa": 1.45,
        "obi_cancel_threshold": 0.60,
        "approved_symbols": ["BTC/USDT:USDT"],
        "max_leverage_cap": 2.0,
        "risk_rationale": "Baja volatilidad y libros equilibrados.",
    }
    decision = RegimeDecision.model_validate(valid_data)
    assert decision.avellaneda_gamma == 0.25
    assert decision.max_leverage_cap == 2.0

    with pytest.raises(ValidationError):
        RegimeDecision.model_validate({**valid_data, "avellaneda_gamma": 1.80})

    with pytest.raises(ValidationError):
        RegimeDecision.model_validate({**valid_data, "obi_cancel_threshold": 0.30})

    with pytest.raises(ValidationError):
        RegimeDecision.model_validate({**valid_data, "max_leverage_cap": 4.5})


def test_three_step_adversarial_debate_execution(
    tmp_path: Path, sample_microstructure_summary: MicrostructureSummary15m
) -> None:
    target_config = tmp_path / "regime_config.json"
    council = GeminiRegimeCouncil(
        api_key="TEST_API_KEY",
        config_output_path=target_config,
    )

    mock_resp_hunter = MagicMock()
    mock_resp_hunter.text = "Propuesta Alpha: Operar BTC con gamma=0.10 y kappa=2.0."

    mock_resp_prosecutor = MagicMock()
    mock_resp_prosecutor.text = "Refutación de Riesgo: SOL muestra flujo tóxico. Exijo gamma >= 0.30."

    expected_arbiter_dict = {
        "market_regime": "LOW_VOL_RANGE",
        "avellaneda_gamma": 0.35,
        "order_book_intensity_kappa": 1.60,
        "obi_cancel_threshold": 0.55,
        "approved_symbols": ["BTC/USDT:USDT"],
        "max_leverage_cap": 2.5,
        "risk_rationale": "Se aprueba BTC excluyendo SOL por riesgo de selección adversa.",
    }
    mock_resp_arbiter = MagicMock()
    mock_resp_arbiter.text = json.dumps(expected_arbiter_dict)

    council.client.models.generate_content = MagicMock(
        side_effect=[mock_resp_hunter, mock_resp_prosecutor, mock_resp_arbiter]
    )

    decision = council.run_debate_cycle(sample_microstructure_summary)

    assert council.client.models.generate_content.call_count == 3
    assert decision.market_regime == "LOW_VOL_RANGE"
    assert decision.avellaneda_gamma == 0.35
    assert decision.approved_symbols == ["BTC/USDT:USDT"]

    assert target_config.exists()
    disk_data = json.loads(target_config.read_text(encoding="utf-8"))
    assert disk_data["market_regime"] == "LOW_VOL_RANGE"
    assert disk_data["avellaneda_gamma"] == 0.35


def test_fallback_mechanism_from_gemini_3_8_to_2_5(
    tmp_path: Path, sample_microstructure_summary: MicrostructureSummary15m
) -> None:
    target_config = tmp_path / "regime_config.json"
    council = GeminiRegimeCouncil(
        api_key="TEST_API_KEY",
        config_output_path=target_config,
    )

    arbiter_json = json.dumps({
        "market_regime": "TOXIC_TREND_PAUSE",
        "avellaneda_gamma": 1.20,
        "order_book_intensity_kappa": 0.80,
        "obi_cancel_threshold": 0.50,
        "approved_symbols": [],
        "max_leverage_cap": 0.0,
        "risk_rationale": "Cascada detectada. Pausa total.",
    })

    def fake_generate_content(model: str, contents: Any, config: Any) -> Any:
        if model == "gemini-3.8-flash":
            raise RuntimeError("Servicio gemini-3.8 temporalmente saturado")
        mock = MagicMock()
        is_arbiter = (
            getattr(config, "response_schema", None) is not None
            or "RegimeDecision" in str(getattr(config, "system_instruction", ""))
            or "Chief" in str(getattr(config, "system_instruction", ""))
        )
        mock.text = arbiter_json if is_arbiter else "Respuesta fallback"
        return mock

    council.client.models.generate_content = MagicMock(side_effect=fake_generate_content)

    decision = council.run_debate_cycle(sample_microstructure_summary)

    assert decision.market_regime == "TOXIC_TREND_PAUSE"
    assert decision.max_leverage_cap == 0.0
    assert len(decision.approved_symbols) == 0


def test_atomic_file_write_never_corrupts_data(tmp_path: Path) -> None:
    target_file = tmp_path / "config" / "regime_config.json"
    council = GeminiRegimeCouncil(api_key="TEST", config_output_path=target_file)

    decision = RegimeDecision(
        market_regime="FUNDING_ARB_FAVORABLE",
        avellaneda_gamma=0.20,
        order_book_intensity_kappa=1.80,
        obi_cancel_threshold=0.65,
        approved_symbols=["BTC/USDT:USDT", "ETH/USDT:USDT"],
        max_leverage_cap=3.0,
        risk_rationale="Condiciones óptimas de arbitraje de financiación.",
    )

    council.save_decision_atomically(decision)

    assert target_file.exists()
    tmp_files = list(target_file.parent.glob("*.tmp"))
    assert len(tmp_files) == 0

    saved_obj = json.loads(target_file.read_text(encoding="utf-8"))
    assert saved_obj["market_regime"] == "FUNDING_ARB_FAVORABLE"
    assert saved_obj["avellaneda_gamma"] == 0.20


def test_deterministic_local_fallback_when_gemini_unavailable(
    tmp_path: Path, sample_microstructure_summary: MicrostructureSummary15m
) -> None:
    target_file = tmp_path / "config" / "regime_config.json"
    # Client initialized without key or failing
    council = GeminiRegimeCouncil(api_key=None, config_output_path=target_file)

    decision = council.run_debate_cycle(sample_microstructure_summary)
    assert decision is not None
    assert decision.market_regime in ("LOW_VOL_RANGE", "HIGH_VOL_LIQ_HUNT")
    assert 0.05 <= decision.avellaneda_gamma <= 1.50
    assert 0.0 <= decision.max_leverage_cap <= 3.5
    assert "Local fallback" in decision.risk_rationale
    assert target_file.exists()


def test_update_api_key_dynamically(tmp_path: Path) -> None:
    target_file = tmp_path / "config" / "regime_config.json"
    council = GeminiRegimeCouncil(api_key=None, config_output_path=target_file)
    assert council.api_key is None

    council.update_api_key("NEW_TEST_VALID_KEY")
    assert council.api_key == "NEW_TEST_VALID_KEY"
    assert council.client is not None

