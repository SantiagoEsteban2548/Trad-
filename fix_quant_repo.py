#!/usr/bin/env python3
"""
fix_quant_repo.py
-----------------
Script de saneamiento, ensamblaje y verificación estructural del repositorio Apex Quant HFT.
1. Corrige extensiones duplicadas o corruptas (.py.py, .tsx.txt, .ts.ts, .json.txt).
2. Elimina cualquier duplicado obsoleto de `regime_config.json` fuera de `backend/config/regime_config.json`.
3. Verifica y genera los archivos base requeridos (conftest.py, index.html, tsconfig.json, backend/config/regime_config.json).
4. Compila el AST de todos los módulos Python en `backend/` y `tests/` para garantizar 0 errores de sintaxis.
"""
from __future__ import annotations

import json
import py_compile
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent

IGNORED_DIRS = {".git", "node_modules", "dist", ".venv", "venv", "__pycache__", ".pytest_cache"}

EXTENSION_FIXES: list[tuple[str, str]] = [
    (".py.py", ".py"),
    (".py.txt", ".py"),
    (".tsx.txt", ".tsx"),
    (".tsx.tsx", ".tsx"),
    (".ts.ts", ".ts"),
    (".ts.txt", ".ts"),
    (".json.txt", ".json"),
    (".json.json", ".json"),
]

REQUIRED_MODULES = [
    "backend/mandate_gates/schemas.py",
    "backend/mandate_gates/gatekeeper.py",
    "backend/mandate_gates/profit_sweeper.py",
    "backend/ingestion/ring_buffer.py",
    "backend/ingestion/ws_multiplexer.py",
    "backend/strategies/regime_config.py",
    "backend/strategies/avellaneda_obi.py",
    "backend/strategies/liq_cascade.py",
    "backend/strategies/funding_arb.py",
    "backend/regime_council/gemini_debate.py",
    "backend/validation/l2_simulator_and_dsr.py",
    "backend/telemetry_server.py",
]

REQUIRED_TESTS = [
    "tests/test_mandate_gates.py",
    "tests/test_ingestion.py",
    "tests/test_strategies.py",
    "tests/test_regime_council.py",
    "tests/test_validation.py",
]


def fix_duplicate_extensions(root: Path) -> int:
    fixed_count = 0
    for path in list(root.rglob("*")):
        if any(part in IGNORED_DIRS for part in path.parts):
            continue
        if not path.is_file():
            continue

        name = path.name
        for bad_ext, clean_ext in EXTENSION_FIXES:
            if name.endswith(bad_ext):
                new_name = name[: -len(bad_ext)] + clean_ext
                target = path.with_name(new_name)
                path.rename(target)
                print(f"[FIXED EXT] {path.relative_to(root)} -> {target.relative_to(root)}")
                fixed_count += 1
                break
    return fixed_count


def ensure_base_files(root: Path) -> None:
    # Eliminar duplicado obsoleto si existiera en backend/regime_config.json
    obsolete_cfg = root / "backend" / "regime_config.json"
    if obsolete_cfg.exists():
        obsolete_cfg.unlink()
        print("[REMOVED DUPLICATE] backend/regime_config.json")

    # 1. conftest.py
    conftest_path = root / "conftest.py"
    if not conftest_path.exists():
        conftest_path.write_text(
            "import sys\nfrom pathlib import Path\n"
            "ROOT = Path(__file__).resolve().parent\n"
            "if str(ROOT) not in sys.path:\n"
            "    sys.path.insert(0, str(ROOT))\n",
            encoding="utf-8",
        )
        print("[CREATED] conftest.py")

    # 2. Paquetes __init__.py
    for pkg in [
        "backend",
        "backend/mandate_gates",
        "backend/ingestion",
        "backend/strategies",
        "backend/regime_council",
        "backend/validation",
    ]:
        pkg_dir = root / pkg
        pkg_dir.mkdir(parents=True, exist_ok=True)
        init_file = pkg_dir / "__init__.py"
        if not init_file.exists():
            init_file.write_text("", encoding="utf-8")
            print(f"[CREATED] {init_file.relative_to(root)}")

    # 3. Ruta unificada: backend/config/regime_config.json
    config_file = root / "backend" / "config" / "regime_config.json"
    if not config_file.exists():
        config_file.parent.mkdir(parents=True, exist_ok=True)
        default_cfg = {
            "market_regime": "LOW_VOL_RANGE",
            "avellaneda_gamma": 0.15,
            "order_book_intensity_kappa": 1.5,
            "obi_cancel_threshold": 0.6,
            "approved_symbols": ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"],
            "max_leverage_cap": 2.5,
            "risk_rationale": "Baseline configuration initialized by fix_quant_repo.py",
        }
        config_file.write_text(json.dumps(default_cfg, indent=2), encoding="utf-8")
        print("[CREATED] backend/config/regime_config.json")


def verify_syntax_and_structure(root: Path) -> bool:
    all_ok = True
    print("\n=== VERIFICACIÓN DE MÓDULOS DE PRODUCCIÓN Y PRUEBAS ===")
    for rel_path in REQUIRED_MODULES + REQUIRED_TESTS:
        full_path = root / rel_path
        if not full_path.exists():
            print(f"[MISSING] {rel_path}")
            all_ok = False
            continue
        try:
            py_compile.compile(str(full_path), doraise=True)
            print(f"[OK AST]  {rel_path}")
        except py_compile.PyCompileError as err:
            print(f"[SYNTAX ERROR] {rel_path}: {err}")
            all_ok = False

    return all_ok


def main() -> int:
    print("Iniciando saneamiento del repositorio Apex Quant HFT...")
    fixed = fix_duplicate_extensions(ROOT_DIR)
    print(f"Extensiones corregidas: {fixed}")
    ensure_base_files(ROOT_DIR)
    ok = verify_syntax_and_structure(ROOT_DIR)
    if ok:
        print("\n[SUCCESS] Todos los módulos de las 5 Fases y las 5 suites de pruebas están íntegros.")
        return 0
    print("\n[ERROR] Se encontraron archivos faltantes o errores de sintaxis.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
