"""
bess — battery energy-storage arbitrage system.

A small research codebase for a final-year dissertation: it fetches GB
electricity price data, optimises a battery's charge/discharge schedule
against that price signal, and backtests the result.

Package layout (filled in stage by stage):
    schema.py           canonical data contract + validation
    config.py           battery hardware/economic parameters
    sources_elexon.py   Elexon BMRS API fetchers
    pipeline.py         fetch -> validate -> repair -> cache -> QA report
    optimiser_tier1.py  linear-programme scheduler (perfect foresight)
    backtest.py         independent SoC/cashflow simulator (correctness anchor)
"""

__version__ = "0.1.0"
