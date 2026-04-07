"""
Financial Data Fetcher
======================
Tries to pull fundamental data from free APIs.
Falls back to manual / hardcoded data when APIs fail.

Sources:
  1. Hardcoded cache (fast, reliable for known stocks)
  2. yfinance live fetch (for any unlisted ticker)
"""

import numpy as np


# ═════════════════════════════════════════════════════════════════════════════
#  HARDCODED FUNDAMENTAL DATA
#  (From latest available annual reports — update periodically)
#  All monetary values in LOCAL CURRENCY (₹ for .NS, $ for US)
#  Values in Crores for Indian companies, Millions for US companies
# ═════════════════════════════════════════════════════════════════════════════

FUNDAMENTAL_DATA = {
    # ── AUTO (INDIA) ────────────────────────────────────────────────────────
    "TATAMOTORS.NS": {
        "company": "Tata Motors", "currency": "INR", "unit": "Cr",
        "net_income": 31807, "ebit": 42500, "revenue": 437927,
        "depreciation": 28000, "capex": 38000, "delta_wc": 5000,
        "total_debt": 108000, "cash": 42000, "shares_outstanding": 366.16,
        "dividends_total": 2198, "tax_rate": 0.25,
        "debt_ratio": 0.35, "debt_ratio_changing": True,
        "cost_of_equity": 0.13, "wacc": 0.105,
        "firm_growth_rate": 0.10, "stable_growth": 0.05,
        "has_competitive_adv": True, "beta": 1.35,
        "risk_free_rate": 0.072, "erp": 0.065,
        "inflation_rate": 0.05, "real_growth_rate": 0.06,
    },
    "M&M.NS": {
        "company": "Mahindra & Mahindra", "currency": "INR", "unit": "Cr",
        "net_income": 12250, "ebit": 16800, "revenue": 148800,
        "depreciation": 5200, "capex": 8500, "delta_wc": 2100,
        "total_debt": 38000, "cash": 18000, "shares_outstanding": 124.52,
        "dividends_total": 2490, "tax_rate": 0.25,
        "debt_ratio": 0.28, "debt_ratio_changing": True,
        "cost_of_equity": 0.125, "wacc": 0.10,
        "firm_growth_rate": 0.12, "stable_growth": 0.05,
        "has_competitive_adv": True, "beta": 1.1,
        "risk_free_rate": 0.072, "erp": 0.065,
        "inflation_rate": 0.05, "real_growth_rate": 0.06,
    },
    "OLECTRA.NS": {
        "company": "Olectra Greentech", "currency": "INR", "unit": "Cr",
        "net_income": 120, "ebit": 170, "revenue": 2100,
        "depreciation": 80, "capex": 200, "delta_wc": 150,
        "total_debt": 600, "cash": 100, "shares_outstanding": 8.18,
        "dividends_total": 0, "tax_rate": 0.25,
        "debt_ratio": 0.30, "debt_ratio_changing": True,
        "cost_of_equity": 0.145, "wacc": 0.115,
        "firm_growth_rate": 0.25, "stable_growth": 0.05,
        "has_competitive_adv": True, "beta": 1.5,
        "risk_free_rate": 0.072, "erp": 0.065,
        "inflation_rate": 0.05, "real_growth_rate": 0.06,
    },
    "ATHERENERG.NS": {
        "company": "Ather Energy", "currency": "INR", "unit": "Cr",
        "net_income": -900, "ebit": -800, "revenue": 1800,
        "depreciation": 200, "capex": 500, "delta_wc": 100,
        "total_debt": 300, "cash": 600, "shares_outstanding": 27.0,
        "dividends_total": 0, "tax_rate": 0.25,
        "debt_ratio": 0.15, "debt_ratio_changing": True,
        "cost_of_equity": 0.16, "wacc": 0.14,
        "firm_growth_rate": 0.35, "stable_growth": 0.05,
        "has_competitive_adv": True, "beta": 1.8,
        "risk_free_rate": 0.072, "erp": 0.065,
        "inflation_rate": 0.05, "real_growth_rate": 0.06,
        "startup_negative": True,
    },

    # ── TYRES (INDIA) ──────────────────────────────────────────────────────
    "APOLLOTYRE.NS": {
        "company": "Apollo Tyres", "currency": "INR", "unit": "Cr",
        "net_income": 2000, "ebit": 3200, "revenue": 26500,
        "depreciation": 1800, "capex": 2500, "delta_wc": 500,
        "total_debt": 5500, "cash": 1500, "shares_outstanding": 63.46,
        "dividends_total": 318, "tax_rate": 0.25,
        "debt_ratio": 0.22, "debt_ratio_changing": False,
        "cost_of_equity": 0.12, "wacc": 0.10,
        "firm_growth_rate": 0.08, "stable_growth": 0.05,
        "has_competitive_adv": False, "beta": 1.0,
        "risk_free_rate": 0.072, "erp": 0.065,
        "inflation_rate": 0.05, "real_growth_rate": 0.06,
    },
    "MRF.NS": {
        "company": "MRF Ltd.", "currency": "INR", "unit": "Cr",
        "net_income": 2050, "ebit": 3500, "revenue": 24500,
        "depreciation": 1400, "capex": 2800, "delta_wc": 600,
        "total_debt": 3000, "cash": 2000, "shares_outstanding": 0.4243,
        "dividends_total": 68, "tax_rate": 0.25,
        "debt_ratio": 0.12, "debt_ratio_changing": False,
        "cost_of_equity": 0.115, "wacc": 0.10,
        "firm_growth_rate": 0.07, "stable_growth": 0.05,
        "has_competitive_adv": True, "beta": 0.85,
        "risk_free_rate": 0.072, "erp": 0.065,
        "inflation_rate": 0.05, "real_growth_rate": 0.06,
    },
    "JKTYRE.NS": {
        "company": "JK Tyre & Industries", "currency": "INR", "unit": "Cr",
        "net_income": 700, "ebit": 1300, "revenue": 14500,
        "depreciation": 600, "capex": 900, "delta_wc": 300,
        "total_debt": 4000, "cash": 500, "shares_outstanding": 24.71,
        "dividends_total": 99, "tax_rate": 0.25,
        "debt_ratio": 0.35, "debt_ratio_changing": True,
        "cost_of_equity": 0.135, "wacc": 0.105,
        "firm_growth_rate": 0.06, "stable_growth": 0.05,
        "has_competitive_adv": False, "beta": 1.2,
        "risk_free_rate": 0.072, "erp": 0.065,
        "inflation_rate": 0.05, "real_growth_rate": 0.06,
    },
    "CEATLTD.NS": {
        "company": "CEAT Ltd.", "currency": "INR", "unit": "Cr",
        "net_income": 600, "ebit": 1100, "revenue": 12500,
        "depreciation": 550, "capex": 800, "delta_wc": 200,
        "total_debt": 2500, "cash": 300, "shares_outstanding": 4.04,
        "dividends_total": 56, "tax_rate": 0.25,
        "debt_ratio": 0.25, "debt_ratio_changing": False,
        "cost_of_equity": 0.125, "wacc": 0.10,
        "firm_growth_rate": 0.08, "stable_growth": 0.05,
        "has_competitive_adv": False, "beta": 1.05,
        "risk_free_rate": 0.072, "erp": 0.065,
        "inflation_rate": 0.05, "real_growth_rate": 0.06,
    },

    # ── BANKING (INDIA) ────────────────────────────────────────────────────
    "SBIN.NS": {
        "company": "State Bank of India", "currency": "INR", "unit": "Cr",
        "net_income": 61077, "ebit": 95000, "revenue": 340000,
        "depreciation": 5000, "capex": 6000, "delta_wc": 10000,
        "total_debt": 4200000, "cash": 200000, "shares_outstanding": 892.46,
        "dividends_total": 11600, "tax_rate": 0.25,
        "debt_ratio": 0.92, "debt_ratio_changing": False,
        "cost_of_equity": 0.13, "wacc": 0.08,
        "firm_growth_rate": 0.10, "stable_growth": 0.05,
        "has_competitive_adv": True, "beta": 1.15,
        "risk_free_rate": 0.072, "erp": 0.065,
        "inflation_rate": 0.05, "real_growth_rate": 0.06,
    },
    "HDFCBANK.NS": {
        "company": "HDFC Bank", "currency": "INR", "unit": "Cr",
        "net_income": 60810, "ebit": 90000, "revenue": 395000,
        "depreciation": 4000, "capex": 5000, "delta_wc": 8000,
        "total_debt": 2400000, "cash": 150000, "shares_outstanding": 762.08,
        "dividends_total": 14600, "tax_rate": 0.25,
        "debt_ratio": 0.90, "debt_ratio_changing": False,
        "cost_of_equity": 0.12, "wacc": 0.075,
        "firm_growth_rate": 0.12, "stable_growth": 0.05,
        "has_competitive_adv": True, "beta": 1.0,
        "risk_free_rate": 0.072, "erp": 0.065,
        "inflation_rate": 0.05, "real_growth_rate": 0.06,
    },
    "ICICIBANK.NS": {
        "company": "ICICI Bank", "currency": "INR", "unit": "Cr",
        "net_income": 44255, "ebit": 70000, "revenue": 250000,
        "depreciation": 3500, "capex": 4500, "delta_wc": 6000,
        "total_debt": 1200000, "cash": 100000, "shares_outstanding": 702.30,
        "dividends_total": 7023, "tax_rate": 0.25,
        "debt_ratio": 0.88, "debt_ratio_changing": False,
        "cost_of_equity": 0.125, "wacc": 0.078,
        "firm_growth_rate": 0.11, "stable_growth": 0.05,
        "has_competitive_adv": True, "beta": 1.05,
        "risk_free_rate": 0.072, "erp": 0.065,
        "inflation_rate": 0.05, "real_growth_rate": 0.06,
    },
    "AXISBANK.NS": {
        "company": "Axis Bank", "currency": "INR", "unit": "Cr",
        "net_income": 24862, "ebit": 40000, "revenue": 120000,
        "depreciation": 2000, "capex": 3000, "delta_wc": 4000,
        "total_debt": 900000, "cash": 60000, "shares_outstanding": 309.25,
        "dividends_total": 309, "tax_rate": 0.25,
        "debt_ratio": 0.88, "debt_ratio_changing": False,
        "cost_of_equity": 0.13, "wacc": 0.08,
        "firm_growth_rate": 0.10, "stable_growth": 0.05,
        "has_competitive_adv": False, "beta": 1.15,
        "risk_free_rate": 0.072, "erp": 0.065,
        "inflation_rate": 0.05, "real_growth_rate": 0.06,
    },

    # ── PHARMA (INDIA) ─────────────────────────────────────────────────────
    "LAURUSLABS.NS": {
        "company": "Laurus Labs", "currency": "INR", "unit": "Cr",
        "net_income": 450, "ebit": 700, "revenue": 5500,
        "depreciation": 400, "capex": 600, "delta_wc": 200,
        "total_debt": 2000, "cash": 300, "shares_outstanding": 53.89,
        "dividends_total": 54, "tax_rate": 0.25,
        "debt_ratio": 0.30, "debt_ratio_changing": True,
        "cost_of_equity": 0.14, "wacc": 0.11,
        "firm_growth_rate": 0.15, "stable_growth": 0.05,
        "has_competitive_adv": True, "beta": 1.3,
        "risk_free_rate": 0.072, "erp": 0.065,
        "inflation_rate": 0.05, "real_growth_rate": 0.06,
    },
    "AUROPHARMA.NS": {
        "company": "Aurobindo Pharma", "currency": "INR", "unit": "Cr",
        "net_income": 3200, "ebit": 4500, "revenue": 29000,
        "depreciation": 1200, "capex": 2000, "delta_wc": 800,
        "total_debt": 4000, "cash": 2500, "shares_outstanding": 58.55,
        "dividends_total": 585, "tax_rate": 0.25,
        "debt_ratio": 0.15, "debt_ratio_changing": False,
        "cost_of_equity": 0.13, "wacc": 0.11,
        "firm_growth_rate": 0.10, "stable_growth": 0.05,
        "has_competitive_adv": True, "beta": 1.2,
        "risk_free_rate": 0.072, "erp": 0.065,
        "inflation_rate": 0.05, "real_growth_rate": 0.06,
    },
    "SUNPHARMA.NS": {
        "company": "Sun Pharma", "currency": "INR", "unit": "Cr",
        "net_income": 11000, "ebit": 15000, "revenue": 50000,
        "depreciation": 2500, "capex": 3000, "delta_wc": 1500,
        "total_debt": 8000, "cash": 12000, "shares_outstanding": 239.8,
        "dividends_total": 2398, "tax_rate": 0.25,
        "debt_ratio": 0.12, "debt_ratio_changing": False,
        "cost_of_equity": 0.115, "wacc": 0.10,
        "firm_growth_rate": 0.10, "stable_growth": 0.05,
        "has_competitive_adv": True, "beta": 0.95,
        "risk_free_rate": 0.072, "erp": 0.065,
        "inflation_rate": 0.05, "real_growth_rate": 0.06,
    },
    "DIVISLAB.NS": {
        "company": "Divi's Laboratories", "currency": "INR", "unit": "Cr",
        "net_income": 1900, "ebit": 2700, "revenue": 8500,
        "depreciation": 700, "capex": 900, "delta_wc": 400,
        "total_debt": 500, "cash": 3000, "shares_outstanding": 26.56,
        "dividends_total": 800, "tax_rate": 0.25,
        "debt_ratio": 0.03, "debt_ratio_changing": False,
        "cost_of_equity": 0.115, "wacc": 0.11,
        "firm_growth_rate": 0.12, "stable_growth": 0.05,
        "has_competitive_adv": True, "beta": 0.9,
        "risk_free_rate": 0.072, "erp": 0.065,
        "inflation_rate": 0.05, "real_growth_rate": 0.06,
    },

    # ── CONSUMER & HOTELS (INDIA) ──────────────────────────────────────────
    "ITC.NS": {
        "company": "ITC Ltd.", "currency": "INR", "unit": "Cr",
        "net_income": 20457, "ebit": 26000, "revenue": 69446,
        "depreciation": 1800, "capex": 2500, "delta_wc": 1000,
        "total_debt": 1500, "cash": 15000, "shares_outstanding": 1254.23,
        "dividends_total": 18813, "tax_rate": 0.25,
        "debt_ratio": 0.04, "debt_ratio_changing": False,
        "cost_of_equity": 0.105, "wacc": 0.10,
        "firm_growth_rate": 0.07, "stable_growth": 0.05,
        "has_competitive_adv": True, "beta": 0.65,
        "risk_free_rate": 0.072, "erp": 0.065,
        "inflation_rate": 0.05, "real_growth_rate": 0.06,
    },
    "CHALET.NS": {
        "company": "Chalet Hotels", "currency": "INR", "unit": "Cr",
        "net_income": 250, "ebit": 450, "revenue": 1800,
        "depreciation": 200, "capex": 300, "delta_wc": 100,
        "total_debt": 2500, "cash": 200, "shares_outstanding": 20.36,
        "dividends_total": 0, "tax_rate": 0.25,
        "debt_ratio": 0.45, "debt_ratio_changing": True,
        "cost_of_equity": 0.135, "wacc": 0.10,
        "firm_growth_rate": 0.15, "stable_growth": 0.05,
        "has_competitive_adv": True, "beta": 1.2,
        "risk_free_rate": 0.072, "erp": 0.065,
        "inflation_rate": 0.05, "real_growth_rate": 0.06,
    },
    "MHRIL.NS": {
        "company": "Mahindra Holidays", "currency": "INR", "unit": "Cr",
        "net_income": 200, "ebit": 350, "revenue": 2800,
        "depreciation": 150, "capex": 200, "delta_wc": 80,
        "total_debt": 500, "cash": 300, "shares_outstanding": 8.93,
        "dividends_total": 45, "tax_rate": 0.25,
        "debt_ratio": 0.15, "debt_ratio_changing": False,
        "cost_of_equity": 0.125, "wacc": 0.11,
        "firm_growth_rate": 0.10, "stable_growth": 0.05,
        "has_competitive_adv": True, "beta": 1.0,
        "risk_free_rate": 0.072, "erp": 0.065,
        "inflation_rate": 0.05, "real_growth_rate": 0.06,
    },
    "INDHOTEL.NS": {
        "company": "Indian Hotels Co.", "currency": "INR", "unit": "Cr",
        "net_income": 1200, "ebit": 1800, "revenue": 7000,
        "depreciation": 600, "capex": 800, "delta_wc": 200,
        "total_debt": 3000, "cash": 1500, "shares_outstanding": 142.27,
        "dividends_total": 1423, "tax_rate": 0.25,
        "debt_ratio": 0.20, "debt_ratio_changing": True,
        "cost_of_equity": 0.12, "wacc": 0.10,
        "firm_growth_rate": 0.15, "stable_growth": 0.05,
        "has_competitive_adv": True, "beta": 1.05,
        "risk_free_rate": 0.072, "erp": 0.065,
        "inflation_rate": 0.05, "real_growth_rate": 0.06,
    },
    "HUL.NS": {
        "company": "Hindustan Unilever", "currency": "INR", "unit": "Cr",
        "net_income": 10300, "ebit": 13800, "revenue": 60000,
        "depreciation": 1500, "capex": 2000, "delta_wc": 800,
        "total_debt": 200, "cash": 6000, "shares_outstanding": 234.94,
        "dividends_total": 8750, "tax_rate": 0.25,
        "debt_ratio": 0.01, "debt_ratio_changing": False,
        "cost_of_equity": 0.105, "wacc": 0.10,
        "firm_growth_rate": 0.07, "stable_growth": 0.05,
        "has_competitive_adv": True, "beta": 0.7,
        "risk_free_rate": 0.072, "erp": 0.065,
        "inflation_rate": 0.05, "real_growth_rate": 0.06,
    },
    "NESTLEIND.NS": {
        "company": "Nestlé India", "currency": "INR", "unit": "Cr",
        "net_income": 3000, "ebit": 3800, "revenue": 17500,
        "depreciation": 500, "capex": 700, "delta_wc": 200,
        "total_debt": 0, "cash": 2500, "shares_outstanding": 9.63,
        "dividends_total": 1000, "tax_rate": 0.25,
        "debt_ratio": 0.0, "debt_ratio_changing": False,
        "cost_of_equity": 0.10, "wacc": 0.10,
        "firm_growth_rate": 0.10, "stable_growth": 0.05,
        "has_competitive_adv": True, "beta": 0.65,
        "risk_free_rate": 0.072, "erp": 0.065,
        "inflation_rate": 0.05, "real_growth_rate": 0.06,
    },

    # ── CEMENT (INDIA) ─────────────────────────────────────────────────────
    "SHREECEM.NS": {
        "company": "Shree Cement", "currency": "INR", "unit": "Cr",
        "net_income": 2200, "ebit": 3500, "revenue": 17500,
        "depreciation": 1500, "capex": 2000, "delta_wc": 500,
        "total_debt": 4000, "cash": 2000, "shares_outstanding": 3.61,
        "dividends_total": 90, "tax_rate": 0.25,
        "debt_ratio": 0.12, "debt_ratio_changing": False,
        "cost_of_equity": 0.115, "wacc": 0.10,
        "firm_growth_rate": 0.08, "stable_growth": 0.05,
        "has_competitive_adv": True, "beta": 0.9,
        "risk_free_rate": 0.072, "erp": 0.065,
        "inflation_rate": 0.05, "real_growth_rate": 0.06,
    },
    "ULTRACEMCO.NS": {
        "company": "UltraTech Cement", "currency": "INR", "unit": "Cr",
        "net_income": 7500, "ebit": 11000, "revenue": 68000,
        "depreciation": 4000, "capex": 6000, "delta_wc": 1500,
        "total_debt": 15000, "cash": 5000, "shares_outstanding": 28.89,
        "dividends_total": 867, "tax_rate": 0.25,
        "debt_ratio": 0.20, "debt_ratio_changing": True,
        "cost_of_equity": 0.115, "wacc": 0.10,
        "firm_growth_rate": 0.10, "stable_growth": 0.05,
        "has_competitive_adv": True, "beta": 0.95,
        "risk_free_rate": 0.072, "erp": 0.065,
        "inflation_rate": 0.05, "real_growth_rate": 0.06,
    },
    "DALBHARAT.NS": {
        "company": "Dalmia Bharat", "currency": "INR", "unit": "Cr",
        "net_income": 1100, "ebit": 2000, "revenue": 14500,
        "depreciation": 900, "capex": 1500, "delta_wc": 400,
        "total_debt": 5000, "cash": 1000, "shares_outstanding": 18.81,
        "dividends_total": 188, "tax_rate": 0.25,
        "debt_ratio": 0.25, "debt_ratio_changing": True,
        "cost_of_equity": 0.125, "wacc": 0.105,
        "firm_growth_rate": 0.10, "stable_growth": 0.05,
        "has_competitive_adv": False, "beta": 1.1,
        "risk_free_rate": 0.072, "erp": 0.065,
        "inflation_rate": 0.05, "real_growth_rate": 0.06,
    },
    "RAMCOCEM.NS": {
        "company": "Ramco Cements", "currency": "INR", "unit": "Cr",
        "net_income": 700, "ebit": 1300, "revenue": 8500,
        "depreciation": 600, "capex": 900, "delta_wc": 300,
        "total_debt": 3500, "cash": 500, "shares_outstanding": 23.57,
        "dividends_total": 235, "tax_rate": 0.25,
        "debt_ratio": 0.30, "debt_ratio_changing": True,
        "cost_of_equity": 0.13, "wacc": 0.105,
        "firm_growth_rate": 0.08, "stable_growth": 0.05,
        "has_competitive_adv": False, "beta": 1.15,
        "risk_free_rate": 0.072, "erp": 0.065,
        "inflation_rate": 0.05, "real_growth_rate": 0.06,
    },

    # ── AMC / FINANCE (INDIA) ──────────────────────────────────────────────
    "ABSLAMC.NS": {
        "company": "Aditya Birla Sun Life AMC", "currency": "INR", "unit": "Cr",
        "net_income": 900, "ebit": 1100, "revenue": 1700,
        "depreciation": 30, "capex": 50, "delta_wc": 60,
        "total_debt": 0, "cash": 2800, "shares_outstanding": 28.93,
        "dividends_total": 650, "tax_rate": 0.25,
        "debt_ratio": 0.0, "debt_ratio_changing": False,
        "cost_of_equity": 0.115, "wacc": 0.115,
        "firm_growth_rate": 0.12, "stable_growth": 0.05,
        "has_competitive_adv": True, "beta": 0.90,
        "risk_free_rate": 0.072, "erp": 0.065,
        "inflation_rate": 0.05, "real_growth_rate": 0.06,
    },
    "HDFCAMC.NS": {
        "company": "HDFC AMC", "currency": "INR", "unit": "Cr",
        "net_income": 2300, "ebit": 2900, "revenue": 4000,
        "depreciation": 60, "capex": 100, "delta_wc": 100,
        "total_debt": 0, "cash": 6000, "shares_outstanding": 21.31,
        "dividends_total": 1790, "tax_rate": 0.25,
        "debt_ratio": 0.0, "debt_ratio_changing": False,
        "cost_of_equity": 0.115, "wacc": 0.115,
        "firm_growth_rate": 0.14, "stable_growth": 0.05,
        "has_competitive_adv": True, "beta": 0.85,
        "risk_free_rate": 0.072, "erp": 0.065,
        "inflation_rate": 0.05, "real_growth_rate": 0.06,
    },
    "NAM-INDIA.NS": {
        "company": "Nippon Life India AMC", "currency": "INR", "unit": "Cr",
        "net_income": 1100, "ebit": 1400, "revenue": 2000,
        "depreciation": 30, "capex": 50, "delta_wc": 60,
        "total_debt": 0, "cash": 3500, "shares_outstanding": 62.36,
        "dividends_total": 810, "tax_rate": 0.25,
        "debt_ratio": 0.0, "debt_ratio_changing": False,
        "cost_of_equity": 0.115, "wacc": 0.115,
        "firm_growth_rate": 0.10, "stable_growth": 0.05,
        "has_competitive_adv": False, "beta": 0.85,
        "risk_free_rate": 0.072, "erp": 0.065,
        "inflation_rate": 0.05, "real_growth_rate": 0.06,
    },
    "UTIAMC.NS": {
        "company": "UTI AMC", "currency": "INR", "unit": "Cr",
        "net_income": 450, "ebit": 600, "revenue": 1200,
        "depreciation": 20, "capex": 30, "delta_wc": 40,
        "total_debt": 0, "cash": 2800, "shares_outstanding": 12.72,
        "dividends_total": 280, "tax_rate": 0.25,
        "debt_ratio": 0.0, "debt_ratio_changing": False,
        "cost_of_equity": 0.115, "wacc": 0.115,
        "firm_growth_rate": 0.08, "stable_growth": 0.05,
        "has_competitive_adv": False, "beta": 0.85,
        "risk_free_rate": 0.072, "erp": 0.065,
        "inflation_rate": 0.05, "real_growth_rate": 0.06,
    },

    # ── TECH (US / GLOBAL) ─────────────────────────────────────────────────
    "NVDA": {
        "company": "NVIDIA Corp.", "currency": "USD", "unit": "M",
        "net_income": 29760, "ebit": 35500, "revenue": 60922,
        "depreciation": 1500, "capex": 2500, "delta_wc": 3000,
        "total_debt": 9700, "cash": 26000, "shares_outstanding": 24500,
        "dividends_total": 490, "tax_rate": 0.12,
        "debt_ratio": 0.05, "debt_ratio_changing": False,
        "cost_of_equity": 0.12, "wacc": 0.115,
        "firm_growth_rate": 0.25, "stable_growth": 0.04,
        "has_competitive_adv": True, "beta": 1.65,
        "risk_free_rate": 0.043, "erp": 0.05,
        "inflation_rate": 0.024, "real_growth_rate": 0.02,
    },
    "MSFT": {
        "company": "Microsoft Corp.", "currency": "USD", "unit": "M",
        "net_income": 72361, "ebit": 88523, "revenue": 211915,
        "depreciation": 13800, "capex": 28300, "delta_wc": 5000,
        "total_debt": 47000, "cash": 75500, "shares_outstanding": 7430,
        "dividends_total": 22100, "tax_rate": 0.18,
        "debt_ratio": 0.08, "debt_ratio_changing": False,
        "cost_of_equity": 0.10, "wacc": 0.095,
        "firm_growth_rate": 0.12, "stable_growth": 0.04,
        "has_competitive_adv": True, "beta": 1.05,
        "risk_free_rate": 0.043, "erp": 0.05,
        "inflation_rate": 0.024, "real_growth_rate": 0.02,
    },
    "GOOGL": {
        "company": "Alphabet Inc.", "currency": "USD", "unit": "M",
        "net_income": 73795, "ebit": 94270, "revenue": 307394,
        "depreciation": 12000, "capex": 32300, "delta_wc": 4000,
        "total_debt": 14000, "cash": 110900, "shares_outstanding": 12200,
        "dividends_total": 2440, "tax_rate": 0.15,
        "debt_ratio": 0.04, "debt_ratio_changing": False,
        "cost_of_equity": 0.105, "wacc": 0.10,
        "firm_growth_rate": 0.12, "stable_growth": 0.04,
        "has_competitive_adv": True, "beta": 1.10,
        "risk_free_rate": 0.043, "erp": 0.05,
        "inflation_rate": 0.024, "real_growth_rate": 0.02,
    },
    "META": {
        "company": "Meta Platforms", "currency": "USD", "unit": "M",
        "net_income": 39098, "ebit": 46751, "revenue": 134902,
        "depreciation": 11000, "capex": 28000, "delta_wc": 3000,
        "total_debt": 18400, "cash": 41600, "shares_outstanding": 2530,
        "dividends_total": 5060, "tax_rate": 0.15,
        "debt_ratio": 0.08, "debt_ratio_changing": False,
        "cost_of_equity": 0.115, "wacc": 0.11,
        "firm_growth_rate": 0.15, "stable_growth": 0.04,
        "has_competitive_adv": True, "beta": 1.30,
        "risk_free_rate": 0.043, "erp": 0.05,
        "inflation_rate": 0.024, "real_growth_rate": 0.02,
    },
    "IBM": {
        "company": "IBM Corp.", "currency": "USD", "unit": "M",
        "net_income": 8500, "ebit": 12000, "revenue": 62000,
        "depreciation": 2500, "capex": 3000, "delta_wc": 500,
        "total_debt": 52000, "cash": 13000, "shares_outstanding": 922,
        "dividends_total": 6100, "tax_rate": 0.20,
        "debt_ratio": 0.55, "debt_ratio_changing": True,
        "cost_of_equity": 0.11, "wacc": 0.085,
        "firm_growth_rate": 0.05, "stable_growth": 0.04,
        "has_competitive_adv": False, "beta": 1.0,
        "risk_free_rate": 0.043, "erp": 0.05,
        "inflation_rate": 0.024, "real_growth_rate": 0.02,
    },
    "ASML": {
        "company": "ASML Holding", "currency": "USD", "unit": "M",
        "net_income": 7800, "ebit": 9500, "revenue": 28000,
        "depreciation": 800, "capex": 1200, "delta_wc": 2000,
        "total_debt": 5000, "cash": 7000, "shares_outstanding": 394,
        "dividends_total": 1700, "tax_rate": 0.15,
        "debt_ratio": 0.10, "debt_ratio_changing": False,
        "cost_of_equity": 0.115, "wacc": 0.105,
        "firm_growth_rate": 0.15, "stable_growth": 0.04,
        "has_competitive_adv": True, "beta": 1.30,
        "risk_free_rate": 0.043, "erp": 0.05,
        "inflation_rate": 0.024, "real_growth_rate": 0.02,
    },
    "INTC": {
        "company": "Intel Corp.", "currency": "USD", "unit": "M",
        "net_income": -1600, "ebit": 500, "revenue": 54200,
        "depreciation": 7500, "capex": 25800, "delta_wc": 1000,
        "total_debt": 48000, "cash": 25000, "shares_outstanding": 4300,
        "dividends_total": 0, "tax_rate": 0.18,
        "debt_ratio": 0.40, "debt_ratio_changing": True,
        "cost_of_equity": 0.13, "wacc": 0.10,
        "firm_growth_rate": 0.06, "stable_growth": 0.04,
        "has_competitive_adv": True, "beta": 1.50,
        "risk_free_rate": 0.043, "erp": 0.05,
        "inflation_rate": 0.024, "real_growth_rate": 0.02,
        "cyclical_negative": True,
    },
    "QCOM": {
        "company": "Qualcomm Inc.", "currency": "USD", "unit": "M",
        "net_income": 8000, "ebit": 10500, "revenue": 38000,
        "depreciation": 1200, "capex": 1800, "delta_wc": 800,
        "total_debt": 15000, "cash": 8000, "shares_outstanding": 1100,
        "dividends_total": 3500, "tax_rate": 0.15,
        "debt_ratio": 0.25, "debt_ratio_changing": False,
        "cost_of_equity": 0.115, "wacc": 0.10,
        "firm_growth_rate": 0.10, "stable_growth": 0.04,
        "has_competitive_adv": True, "beta": 1.30,
        "risk_free_rate": 0.043, "erp": 0.05,
        "inflation_rate": 0.024, "real_growth_rate": 0.02,
    },
    "CRM": {
        "company": "Salesforce Inc.", "currency": "USD", "unit": "M",
        "net_income": 4600, "ebit": 6500, "revenue": 34900,
        "depreciation": 1500, "capex": 1000, "delta_wc": 800,
        "total_debt": 9500, "cash": 12500, "shares_outstanding": 967,
        "dividends_total": 1500, "tax_rate": 0.18,
        "debt_ratio": 0.12, "debt_ratio_changing": False,
        "cost_of_equity": 0.11, "wacc": 0.10,
        "firm_growth_rate": 0.10, "stable_growth": 0.04,
        "has_competitive_adv": True, "beta": 1.20,
        "risk_free_rate": 0.043, "erp": 0.05,
        "inflation_rate": 0.024, "real_growth_rate": 0.02,
    },
    "PLTR": {
        "company": "Palantir Technologies", "currency": "USD", "unit": "M",
        "net_income": 460, "ebit": 550, "revenue": 2600,
        "depreciation": 50, "capex": 80, "delta_wc": 100,
        "total_debt": 200, "cash": 3700, "shares_outstanding": 2350,
        "dividends_total": 0, "tax_rate": 0.15,
        "debt_ratio": 0.03, "debt_ratio_changing": False,
        "cost_of_equity": 0.14, "wacc": 0.135,
        "firm_growth_rate": 0.25, "stable_growth": 0.04,
        "has_competitive_adv": True, "beta": 2.0,
        "risk_free_rate": 0.043, "erp": 0.05,
        "inflation_rate": 0.024, "real_growth_rate": 0.02,
    },
    "CRWD": {
        "company": "CrowdStrike Holdings", "currency": "USD", "unit": "M",
        "net_income": 500, "ebit": 400, "revenue": 3900,
        "depreciation": 250, "capex": 300, "delta_wc": 200,
        "total_debt": 700, "cash": 3200, "shares_outstanding": 245,
        "dividends_total": 0, "tax_rate": 0.15,
        "debt_ratio": 0.05, "debt_ratio_changing": False,
        "cost_of_equity": 0.13, "wacc": 0.125,
        "firm_growth_rate": 0.22, "stable_growth": 0.04,
        "has_competitive_adv": True, "beta": 1.60,
        "risk_free_rate": 0.043, "erp": 0.05,
        "inflation_rate": 0.024, "real_growth_rate": 0.02,
    },

    # ── MEDIA & CONSUMER (US) ──────────────────────────────────────────────
    "WBD": {
        "company": "Warner Bros. Discovery", "currency": "USD", "unit": "M",
        "net_income": -3000, "ebit": 2000, "revenue": 41000,
        "depreciation": 3000, "capex": 2000, "delta_wc": 500,
        "total_debt": 43000, "cash": 3500, "shares_outstanding": 2450,
        "dividends_total": 0, "tax_rate": 0.20,
        "debt_ratio": 0.65, "debt_ratio_changing": True,
        "cost_of_equity": 0.14, "wacc": 0.09,
        "firm_growth_rate": 0.04, "stable_growth": 0.03,
        "has_competitive_adv": False, "beta": 1.60,
        "risk_free_rate": 0.043, "erp": 0.05,
        "inflation_rate": 0.024, "real_growth_rate": 0.02,
        "excess_debt_negative": True, "bankruptcy_likely": False,
    },
    "NFLX": {
        "company": "Netflix Inc.", "currency": "USD", "unit": "M",
        "net_income": 8500, "ebit": 9800, "revenue": 38500,
        "depreciation": 700, "capex": 500, "delta_wc": 1000,
        "total_debt": 14000, "cash": 7600, "shares_outstanding": 430,
        "dividends_total": 0, "tax_rate": 0.18,
        "debt_ratio": 0.15, "debt_ratio_changing": True,
        "cost_of_equity": 0.115, "wacc": 0.10,
        "firm_growth_rate": 0.12, "stable_growth": 0.04,
        "has_competitive_adv": True, "beta": 1.30,
        "risk_free_rate": 0.043, "erp": 0.05,
        "inflation_rate": 0.024, "real_growth_rate": 0.02,
    },
    "DIS": {
        "company": "Walt Disney Co.", "currency": "USD", "unit": "M",
        "net_income": 4400, "ebit": 10000, "revenue": 88900,
        "depreciation": 5000, "capex": 6000, "delta_wc": 1000,
        "total_debt": 42000, "cash": 6000, "shares_outstanding": 1830,
        "dividends_total": 1830, "tax_rate": 0.20,
        "debt_ratio": 0.30, "debt_ratio_changing": True,
        "cost_of_equity": 0.115, "wacc": 0.09,
        "firm_growth_rate": 0.08, "stable_growth": 0.04,
        "has_competitive_adv": True, "beta": 1.30,
        "risk_free_rate": 0.043, "erp": 0.05,
        "inflation_rate": 0.024, "real_growth_rate": 0.02,
    },
    "PARA": {
        "company": "Paramount Global", "currency": "USD", "unit": "M",
        "net_income": -500, "ebit": 800, "revenue": 30000,
        "depreciation": 1200, "capex": 800, "delta_wc": 300,
        "total_debt": 15000, "cash": 2000, "shares_outstanding": 660,
        "dividends_total": 66, "tax_rate": 0.20,
        "debt_ratio": 0.50, "debt_ratio_changing": True,
        "cost_of_equity": 0.14, "wacc": 0.095,
        "firm_growth_rate": 0.03, "stable_growth": 0.03,
        "has_competitive_adv": False, "beta": 1.70,
        "risk_free_rate": 0.043, "erp": 0.05,
        "inflation_rate": 0.024, "real_growth_rate": 0.02,
        "cyclical_negative": True,
    },
    "PG": {
        "company": "Procter & Gamble", "currency": "USD", "unit": "M",
        "net_income": 14800, "ebit": 18500, "revenue": 84000,
        "depreciation": 2800, "capex": 3500, "delta_wc": 1000,
        "total_debt": 28000, "cash": 10000, "shares_outstanding": 2360,
        "dividends_total": 9440, "tax_rate": 0.20,
        "debt_ratio": 0.18, "debt_ratio_changing": False,
        "cost_of_equity": 0.09, "wacc": 0.08,
        "firm_growth_rate": 0.05, "stable_growth": 0.04,
        "has_competitive_adv": True, "beta": 0.60,
        "risk_free_rate": 0.043, "erp": 0.05,
        "inflation_rate": 0.024, "real_growth_rate": 0.02,
    },
    "WMT": {
        "company": "Walmart Inc.", "currency": "USD", "unit": "M",
        "net_income": 15500, "ebit": 27000, "revenue": 648100,
        "depreciation": 11000, "capex": 16000, "delta_wc": 2000,
        "total_debt": 37000, "cash": 9000, "shares_outstanding": 8050,
        "dividends_total": 6100, "tax_rate": 0.22,
        "debt_ratio": 0.15, "debt_ratio_changing": False,
        "cost_of_equity": 0.09, "wacc": 0.08,
        "firm_growth_rate": 0.05, "stable_growth": 0.04,
        "has_competitive_adv": True, "beta": 0.55,
        "risk_free_rate": 0.043, "erp": 0.05,
        "inflation_rate": 0.024, "real_growth_rate": 0.02,
    },

    # ── DEFENSE (US) ───────────────────────────────────────────────────────
    "LMT": {
        "company": "Lockheed Martin", "currency": "USD", "unit": "M",
        "net_income": 6900, "ebit": 9000, "revenue": 67600,
        "depreciation": 1300, "capex": 1800, "delta_wc": 500,
        "total_debt": 17000, "cash": 3000, "shares_outstanding": 240,
        "dividends_total": 3120, "tax_rate": 0.18,
        "debt_ratio": 0.40, "debt_ratio_changing": False,
        "cost_of_equity": 0.105, "wacc": 0.085,
        "firm_growth_rate": 0.06, "stable_growth": 0.04,
        "has_competitive_adv": True, "beta": 0.80,
        "risk_free_rate": 0.043, "erp": 0.05,
        "inflation_rate": 0.024, "real_growth_rate": 0.02,
    },
    "GD": {
        "company": "General Dynamics", "currency": "USD", "unit": "M",
        "net_income": 3600, "ebit": 4800, "revenue": 42300,
        "depreciation": 700, "capex": 1100, "delta_wc": 400,
        "total_debt": 10000, "cash": 1500, "shares_outstanding": 270,
        "dividends_total": 1500, "tax_rate": 0.18,
        "debt_ratio": 0.30, "debt_ratio_changing": False,
        "cost_of_equity": 0.10, "wacc": 0.085,
        "firm_growth_rate": 0.06, "stable_growth": 0.04,
        "has_competitive_adv": True, "beta": 0.85,
        "risk_free_rate": 0.043, "erp": 0.05,
        "inflation_rate": 0.024, "real_growth_rate": 0.02,
    },
    "NOC": {
        "company": "Northrop Grumman", "currency": "USD", "unit": "M",
        "net_income": 3900, "ebit": 5200, "revenue": 39300,
        "depreciation": 800, "capex": 1200, "delta_wc": 300,
        "total_debt": 13000, "cash": 3500, "shares_outstanding": 148,
        "dividends_total": 1110, "tax_rate": 0.18,
        "debt_ratio": 0.35, "debt_ratio_changing": False,
        "cost_of_equity": 0.105, "wacc": 0.085,
        "firm_growth_rate": 0.06, "stable_growth": 0.04,
        "has_competitive_adv": True, "beta": 0.85,
        "risk_free_rate": 0.043, "erp": 0.05,
        "inflation_rate": 0.024, "real_growth_rate": 0.02,
    },
    "RTX": {
        "company": "RTX Corporation", "currency": "USD", "unit": "M",
        "net_income": 3500, "ebit": 7000, "revenue": 69000,
        "depreciation": 2500, "capex": 3000, "delta_wc": 800,
        "total_debt": 32000, "cash": 6000, "shares_outstanding": 1330,
        "dividends_total": 3190, "tax_rate": 0.20,
        "debt_ratio": 0.35, "debt_ratio_changing": True,
        "cost_of_equity": 0.11, "wacc": 0.085,
        "firm_growth_rate": 0.06, "stable_growth": 0.04,
        "has_competitive_adv": True, "beta": 0.95,
        "risk_free_rate": 0.043, "erp": 0.05,
        "inflation_rate": 0.024, "real_growth_rate": 0.02,
    },

    # ── AUTO (GLOBAL) ──────────────────────────────────────────────────────
    "TSLA": {
        "company": "Tesla Inc.", "currency": "USD", "unit": "M",
        "net_income": 7900, "ebit": 8900, "revenue": 96800,
        "depreciation": 4700, "capex": 8900, "delta_wc": 2000,
        "total_debt": 5700, "cash": 16400, "shares_outstanding": 3210,
        "dividends_total": 0, "tax_rate": 0.15,
        "debt_ratio": 0.05, "debt_ratio_changing": False,
        "cost_of_equity": 0.14, "wacc": 0.13,
        "firm_growth_rate": 0.20, "stable_growth": 0.04,
        "has_competitive_adv": True, "beta": 2.0,
        "risk_free_rate": 0.043, "erp": 0.05,
        "inflation_rate": 0.024, "real_growth_rate": 0.02,
    },
    "P911.DE": {
        "company": "Porsche AG", "currency": "USD", "unit": "M",
        "net_income": 4400, "ebit": 6500, "revenue": 40000,
        "depreciation": 2500, "capex": 4000, "delta_wc": 1000,
        "total_debt": 8000, "cash": 5000, "shares_outstanding": 911,
        "dividends_total": 2000, "tax_rate": 0.25,
        "debt_ratio": 0.15, "debt_ratio_changing": False,
        "cost_of_equity": 0.11, "wacc": 0.095,
        "firm_growth_rate": 0.06, "stable_growth": 0.03,
        "has_competitive_adv": True, "beta": 1.10,
        "risk_free_rate": 0.043, "erp": 0.05,
        "inflation_rate": 0.024, "real_growth_rate": 0.02,
    },
    "F": {
        "company": "Ford Motor Co.", "currency": "USD", "unit": "M",
        "net_income": 4300, "ebit": 10000, "revenue": 176000,
        "depreciation": 7000, "capex": 8000, "delta_wc": 2000,
        "total_debt": 100000, "cash": 26000, "shares_outstanding": 3980,
        "dividends_total": 2400, "tax_rate": 0.20,
        "debt_ratio": 0.60, "debt_ratio_changing": True,
        "cost_of_equity": 0.13, "wacc": 0.085,
        "firm_growth_rate": 0.04, "stable_growth": 0.03,
        "has_competitive_adv": False, "beta": 1.50,
        "risk_free_rate": 0.043, "erp": 0.05,
        "inflation_rate": 0.024, "real_growth_rate": 0.02,
    },
    "VOW3.DE": {
        "company": "Volkswagen AG", "currency": "USD", "unit": "M",
        "net_income": 12000, "ebit": 20000, "revenue": 280000,
        "depreciation": 15000, "capex": 18000, "delta_wc": 3000,
        "total_debt": 120000, "cash": 40000, "shares_outstanding": 500,
        "dividends_total": 4500, "tax_rate": 0.25,
        "debt_ratio": 0.45, "debt_ratio_changing": True,
        "cost_of_equity": 0.12, "wacc": 0.08,
        "firm_growth_rate": 0.04, "stable_growth": 0.03,
        "has_competitive_adv": False, "beta": 1.30,
        "risk_free_rate": 0.043, "erp": 0.05,
        "inflation_rate": 0.024, "real_growth_rate": 0.02,
    },
    "HYMTF": {
        "company": "Hyundai Motor", "currency": "USD", "unit": "M",
        "net_income": 8500, "ebit": 11000, "revenue": 115000,
        "depreciation": 5000, "capex": 7000, "delta_wc": 2000,
        "total_debt": 20000, "cash": 12000, "shares_outstanding": 224,
        "dividends_total": 1800, "tax_rate": 0.22,
        "debt_ratio": 0.18, "debt_ratio_changing": False,
        "cost_of_equity": 0.11, "wacc": 0.095,
        "firm_growth_rate": 0.06, "stable_growth": 0.03,
        "has_competitive_adv": True, "beta": 1.10,
        "risk_free_rate": 0.043, "erp": 0.05,
        "inflation_rate": 0.024, "real_growth_rate": 0.02,
    },
}


# ═════════════════════════════════════════════════════════════════════════════
#  UPGRADE 1: Live fundamental fetch via yfinance for any ticker
# ═════════════════════════════════════════════════════════════════════════════

def _safe_val(val, default=0.0):
    """Return float val or default if None/NaN."""
    try:
        v = float(val)
        return v if not (v != v) else default  # NaN check
    except (TypeError, ValueError):
        return default


def fetch_live_fundamentals(ticker: str) -> dict:
    """
    Fetches all needed DCF inputs live from yfinance for any US or Indian stock.
    Indian stocks (.NS/.BO): currency=INR, unit=Cr, divisor=1e7
    US stocks: currency=USD, unit=M, divisor=1e6
    """
    import yfinance as yf

    is_indian = ticker.endswith(".NS") or ticker.endswith(".BO")
    currency = "INR" if is_indian else "USD"
    unit = "Cr" if is_indian else "M"
    divisor = 1e7 if is_indian else 1e6  # raw values are in absolute currency units

    # Macroeconomic constants
    if is_indian:
        risk_free_rate = 0.072
        erp = 0.065
        inflation_rate = 0.05
        real_growth_rate = 0.06
    else:
        risk_free_rate = 0.043
        erp = 0.05
        inflation_rate = 0.024
        real_growth_rate = 0.02

    t = yf.Ticker(ticker)
    info = t.info or {}

    # ── Company name ────────────────────────────────────────────────────────
    company = info.get("longName") or info.get("shortName") or ticker

    # ── Income statement ────────────────────────────────────────────────────
    try:
        fin = t.financials  # annual, most recent first
        if fin is not None and not fin.empty:
            revenue_raw = _safe_val(fin.loc["Total Revenue"].iloc[0] if "Total Revenue" in fin.index else None)
            if revenue_raw == 0:
                revenue_raw = _safe_val(fin.loc["Total Revenue"].iloc[0] if "Total Revenue" in fin.index else info.get("totalRevenue", 0))
            ebit_raw = _safe_val(fin.loc["EBIT"].iloc[0] if "EBIT" in fin.index else
                                  (fin.loc["Operating Income"].iloc[0] if "Operating Income" in fin.index else None))
            net_income_raw = _safe_val(fin.loc["Net Income"].iloc[0] if "Net Income" in fin.index else None)
            # Tax paid
            pretax_income = _safe_val(fin.loc["Pretax Income"].iloc[0] if "Pretax Income" in fin.index else None)
            tax_provision = _safe_val(fin.loc["Tax Provision"].iloc[0] if "Tax Provision" in fin.index else None)
            tax_rate = (tax_provision / pretax_income) if pretax_income and pretax_income != 0 else 0.25
            tax_rate = max(0.01, min(tax_rate, 0.45))

            # Depreciation from financials
            dep_raw = _safe_val(fin.loc["Depreciation And Amortization"].iloc[0]
                                 if "Depreciation And Amortization" in fin.index else None)

            # Prior year revenue for growth estimate
            if fin.shape[1] >= 2:
                rev_prev = _safe_val(fin.loc["Total Revenue"].iloc[1] if "Total Revenue" in fin.index else None)
                firm_growth_rate = ((revenue_raw - rev_prev) / abs(rev_prev)) if rev_prev != 0 else 0.08
                firm_growth_rate = max(-0.30, min(firm_growth_rate, 0.60))
            else:
                firm_growth_rate = info.get("revenueGrowth", 0.08) or 0.08
        else:
            revenue_raw = _safe_val(info.get("totalRevenue", 0))
            ebit_raw = _safe_val(info.get("ebitda", 0)) * 0.8  # rough estimate
            net_income_raw = _safe_val(info.get("netIncomeToCommon", 0))
            tax_rate = 0.25
            dep_raw = 0.0
            firm_growth_rate = _safe_val(info.get("revenueGrowth", 0.08))
    except Exception:
        revenue_raw = _safe_val(info.get("totalRevenue", 0))
        ebit_raw = _safe_val(info.get("ebitda", 0)) * 0.8
        net_income_raw = _safe_val(info.get("netIncomeToCommon", 0))
        tax_rate = 0.25
        dep_raw = 0.0
        firm_growth_rate = _safe_val(info.get("revenueGrowth", 0.08))

    # ── Balance sheet ───────────────────────────────────────────────────────
    try:
        bs = t.balance_sheet
        if bs is not None and not bs.empty:
            total_debt_raw = _safe_val(bs.loc["Total Debt"].iloc[0] if "Total Debt" in bs.index else
                                        bs.loc["Long Term Debt"].iloc[0] if "Long Term Debt" in bs.index else None)
            cash_raw = _safe_val(bs.loc["Cash And Cash Equivalents"].iloc[0]
                                   if "Cash And Cash Equivalents" in bs.index else
                                   bs.loc["Cash"].iloc[0] if "Cash" in bs.index else None)
            # Working capital change
            if bs.shape[1] >= 2:
                curr_assets = _safe_val(bs.loc["Current Assets"].iloc[0] if "Current Assets" in bs.index else None)
                prev_assets = _safe_val(bs.loc["Current Assets"].iloc[1] if "Current Assets" in bs.index else None)
                curr_liab = _safe_val(bs.loc["Current Liabilities"].iloc[0] if "Current Liabilities" in bs.index else None)
                prev_liab = _safe_val(bs.loc["Current Liabilities"].iloc[1] if "Current Liabilities" in bs.index else None)
                # Exclude cash from working capital
                curr_cash = cash_raw
                wc_curr = (curr_assets - curr_cash) - curr_liab
                wc_prev = (prev_assets - _safe_val(bs.loc["Cash And Cash Equivalents"].iloc[1]
                                                    if "Cash And Cash Equivalents" in bs.index else None)) - prev_liab
                delta_wc_raw = wc_curr - wc_prev
            else:
                delta_wc_raw = revenue_raw * 0.01  # rough estimate
        else:
            total_debt_raw = _safe_val(info.get("totalDebt", 0))
            cash_raw = _safe_val(info.get("totalCash", 0))
            delta_wc_raw = 0.0
    except Exception:
        total_debt_raw = _safe_val(info.get("totalDebt", 0))
        cash_raw = _safe_val(info.get("totalCash", 0))
        delta_wc_raw = 0.0

    # ── Cash flow statement ─────────────────────────────────────────────────
    try:
        cf = t.cashflow
        if cf is not None and not cf.empty:
            capex_raw = abs(_safe_val(cf.loc["Capital Expenditure"].iloc[0]
                                       if "Capital Expenditure" in cf.index else
                                       cf.loc["Purchase Of Property Plant And Equipment"].iloc[0]
                                       if "Purchase Of Property Plant And Equipment" in cf.index else None))
            dep_cf = _safe_val(cf.loc["Depreciation And Amortization"].iloc[0]
                                if "Depreciation And Amortization" in cf.index else None)
            if dep_raw == 0:
                dep_raw = dep_cf
            dividends_raw = abs(_safe_val(cf.loc["Cash Dividends Paid"].iloc[0]
                                           if "Cash Dividends Paid" in cf.index else
                                           cf.loc["Common Stock Dividend Paid"].iloc[0]
                                           if "Common Stock Dividend Paid" in cf.index else None))
        else:
            capex_raw = _safe_val(info.get("capitalExpenditures", 0))
            dividends_raw = 0.0
    except Exception:
        capex_raw = _safe_val(info.get("capitalExpenditures", 0))
        dividends_raw = 0.0

    # ── Shares outstanding ──────────────────────────────────────────────────
    shares_raw = _safe_val(info.get("sharesOutstanding", info.get("impliedSharesOutstanding", 0)))

    # ── Beta ────────────────────────────────────────────────────────────────
    beta = _safe_val(info.get("beta", 1.0))
    if beta == 0.0:
        beta = 1.0

    # ── CAPM cost of equity ─────────────────────────────────────────────────
    cost_of_equity = risk_free_rate + beta * erp

    # ── Debt ratio & cost of debt ───────────────────────────────────────────
    market_cap_raw = _safe_val(info.get("marketCap", 0))
    if market_cap_raw > 0 and total_debt_raw > 0:
        total_capital = market_cap_raw + total_debt_raw
        debt_ratio = total_debt_raw / total_capital
    elif total_debt_raw > 0 and net_income_raw != 0:
        equity_book = abs(net_income_raw) * 10  # rough proxy
        debt_ratio = total_debt_raw / (total_debt_raw + equity_book)
    else:
        debt_ratio = total_debt_raw / max(total_debt_raw + abs(net_income_raw) * 10, 1)
    debt_ratio = max(0.0, min(debt_ratio, 0.95))

    cost_of_debt = _safe_val(info.get("debtInterestExpense", 0))
    if cost_of_debt == 0 or total_debt_raw == 0:
        cost_of_debt = risk_free_rate + 0.02  # default spread
    else:
        cost_of_debt = cost_of_debt / total_debt_raw
    cost_of_debt = max(0.02, min(cost_of_debt, 0.20))

    wacc = cost_of_equity * (1 - debt_ratio) + cost_of_debt * (1 - tax_rate) * debt_ratio

    # ── Debt ratio changing? ────────────────────────────────────────────────
    try:
        bs2 = t.balance_sheet
        if bs2 is not None and bs2.shape[1] >= 2:
            td_prev = _safe_val(bs2.loc["Total Debt"].iloc[1] if "Total Debt" in bs2.index else None)
            if td_prev > 0 and total_debt_raw > 0:
                debt_change_pct = abs(total_debt_raw - td_prev) / td_prev
                debt_ratio_changing = debt_change_pct > 0.05
            else:
                debt_ratio_changing = False
        else:
            debt_ratio_changing = False
    except Exception:
        debt_ratio_changing = False

    # ── has_competitive_adv ─────────────────────────────────────────────────
    roe = _safe_val(info.get("returnOnEquity", 0))
    operating_margin = _safe_val(info.get("operatingMargins", 0))
    has_competitive_adv = (roe > 0.15 and operating_margin > 0.20)

    # ── stable_growth ───────────────────────────────────────────────────────
    stable_growth = inflation_rate + real_growth_rate

    # ── Convert all monetary values to unit (Cr for India, M for US) ────────
    def to_unit(raw_val):
        return raw_val / divisor if raw_val != 0 else 0.0

    revenue = to_unit(revenue_raw)
    ebit = to_unit(ebit_raw)
    net_income = to_unit(net_income_raw)
    depreciation = to_unit(dep_raw)
    capex = to_unit(capex_raw)
    delta_wc = to_unit(delta_wc_raw)
    total_debt = to_unit(total_debt_raw)
    cash = to_unit(cash_raw)
    dividends_total = to_unit(dividends_raw)

    # Shares in crores (India) or millions (US)
    shares_outstanding = to_unit(shares_raw)

    return {
        "company": company,
        "currency": currency,
        "unit": unit,
        "net_income": net_income,
        "ebit": ebit,
        "revenue": revenue,
        "depreciation": depreciation,
        "capex": capex,
        "delta_wc": delta_wc,
        "total_debt": total_debt,
        "cash": cash,
        "shares_outstanding": shares_outstanding,
        "dividends_total": dividends_total,
        "tax_rate": tax_rate,
        "debt_ratio": debt_ratio,
        "debt_ratio_changing": debt_ratio_changing,
        "cost_of_equity": cost_of_equity,
        "wacc": wacc,
        "firm_growth_rate": firm_growth_rate,
        "stable_growth": stable_growth,
        "has_competitive_adv": has_competitive_adv,
        "beta": beta,
        "risk_free_rate": risk_free_rate,
        "erp": erp,
        "inflation_rate": inflation_rate,
        "real_growth_rate": real_growth_rate,
        "_live_fetched": True,
    }


def get_fundamental_data(ticker: str) -> dict:
    """
    Returns fundamental data dict for a given ticker.
    1. First checks FUNDAMENTAL_DATA cache (hardcoded, reliable)
    2. If not found, calls fetch_live_fundamentals() via yfinance
    3. If that also fails, raises ValueError
    """
    if ticker in FUNDAMENTAL_DATA:
        return FUNDAMENTAL_DATA[ticker]

    try:
        return fetch_live_fundamentals(ticker)
    except Exception as e:
        raise ValueError(
            f"No fundamental data available for '{ticker}'. "
            f"Live fetch also failed: {e}"
        )
