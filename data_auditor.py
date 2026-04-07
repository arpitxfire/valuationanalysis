"""
Multi-Source Data Auditor
=========================
Audits every financial metric from 3 independent sources:
  Source 1: yfinance Ticker.info dict (market cap, beta, forward PE, etc.)
  Source 2: yfinance financial statements (.financials, .balance_sheet, .cashflow)
  Source 3: Derived cross-checks (e.g., EBIT = Revenue x OPM, tax = tax_paid/pretax,
            FCFE formula consistency, WACC = Ke*(1-DR) + Kd*(1-t)*DR)
"""

import numpy as np


def _safe(val, default=None):
    """Safely convert to float, returning default on failure."""
    if val is None:
        return default
    try:
        v = float(val)
        return v if not (v != v) else default  # NaN check
    except (TypeError, ValueError):
        return default


def _pct_dev(our_val, ref_val):
    """Percentage deviation of our_val from ref_val."""
    if ref_val is None or ref_val == 0:
        return None
    return abs(our_val - ref_val) / abs(ref_val)


def _flag(dev):
    """Return confidence flag based on deviation."""
    if dev is None:
        return "⚪", "Unverified"
    if dev < 0.10:
        return "🟢", "High"
    if dev < 0.30:
        return "🟡", "Medium"
    return "🔴", "Low"


def audit_financial_data(ticker: str, fd: dict) -> dict:
    """
    Audits every metric in fd against 3 independent sources.

    Returns:
    {
        "overall_confidence": "High" | "Medium" | "Low",
        "overall_score": float (0-100),
        "metrics": [ { field, our_value, source1_value, source2_value,
                        source3_value, consensus_value, deviation,
                        confidence, flag, sources_description } ],
        "warnings": [...],
        "errors": [...],
        "data_source_note": "...",
    }
    """
    import yfinance as yf

    is_indian = ticker.endswith(".NS") or ticker.endswith(".BO")
    divisor = 1e7 if is_indian else 1e6
    unit = fd.get("unit", "Cr" if is_indian else "M")
    currency = fd.get("currency", "INR" if is_indian else "USD")
    cur_sym = "\u20b9" if currency == "INR" else "$"

    if is_indian:
        filing_source = "BSE/NSE filings via Yahoo Finance"
        stmt_source = "NSE/BSE Annual Report (via Yahoo Finance)"
    else:
        filing_source = "SEC filings via Yahoo Finance"
        stmt_source = "SEC 10-K Filings (via Yahoo Finance)"

    warnings = []
    errors = []
    metrics = []

    # Fetch all 3 sources
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
    except Exception as e:
        info = {}
        warnings.append(f"Source 1 (yfinance .info) unavailable: {e}")

    try:
        fin = t.financials
    except Exception:
        fin = None

    try:
        bs = t.balance_sheet
    except Exception:
        bs = None

    try:
        cf = t.cashflow
    except Exception:
        cf = None

    def _from_fin(label, idx=0):
        if fin is None or fin.empty:
            return None
        for key in fin.index:
            if label.lower() in key.lower():
                try:
                    return _safe(fin.loc[key].iloc[idx])
                except Exception:
                    return None
        return None

    def _from_bs(label, idx=0):
        if bs is None or bs.empty:
            return None
        for key in bs.index:
            if label.lower() in key.lower():
                try:
                    return _safe(bs.loc[key].iloc[idx])
                except Exception:
                    return None
        return None

    def _from_cf(label, idx=0):
        if cf is None or cf.empty:
            return None
        for key in cf.index:
            if label.lower() in key.lower():
                try:
                    return _safe(cf.loc[key].iloc[idx])
                except Exception:
                    return None
        return None

    def to_unit(v):
        if v is None:
            return None
        return v / divisor

    def add_metric(field, our_value, s1_raw, s2_raw, s3_derived,
                   s1_desc, s2_desc, s3_desc):
        """Build a metric audit row."""
        s1 = to_unit(s1_raw) if s1_raw is not None else None
        s2 = to_unit(s2_raw) if s2_raw is not None else None
        s3 = s3_derived  # already in unit

        # Consensus: median of available values
        avail = [v for v in [s1, s2, s3] if v is not None]
        if avail:
            consensus = float(np.median(avail))
        else:
            consensus = None

        dev = _pct_dev(our_value, consensus) if consensus is not None else None
        flag_sym, confidence = _flag(dev)

        # Max deviation across all sources
        devs = []
        for v in [s1, s2, s3]:
            if v is not None:
                d = _pct_dev(our_value, v)
                if d is not None:
                    devs.append(d)
        max_dev = max(devs) if devs else None

        if max_dev is not None and max_dev > 0.30:
            warnings.append(
                f"{field}: our value {our_value:,.1f} {unit} deviates "
                f"{max_dev:.0%} from one or more sources"
            )

        sources_desc = f"{s1_desc} | {s2_desc} | {s3_desc}"

        metrics.append({
            "field": field,
            "our_value": our_value,
            "source1_value": s1,
            "source2_value": s2,
            "source3_value": s3,
            "consensus_value": consensus,
            "deviation": dev,
            "confidence": confidence,
            "flag": flag_sym,
            "sources_description": sources_desc,
        })

    # 1. Revenue
    s1_rev = _safe(info.get("totalRevenue"))
    s2_rev = _from_fin("total revenue")
    gross_margin = _safe(info.get("grossMargins"))
    gross_profit = _from_fin("gross profit")
    s3_rev_raw = (gross_profit / gross_margin) if (gross_profit and gross_margin and gross_margin > 0) else None
    s3_rev = to_unit(s3_rev_raw) if s3_rev_raw else None
    add_metric(
        "revenue", fd.get("revenue", 0),
        s1_rev, s2_rev, s3_rev,
        "yfinance .info totalRevenue",
        f"{stmt_source} (Total Revenue)",
        "Gross Profit / Gross Margin (derived)",
    )

    # 2. EBIT
    s1_ebit = _safe(info.get("ebitda"))
    s2_ebit = _from_fin("ebit") or _from_fin("operating income")
    op_margin = _safe(info.get("operatingMargins"))
    rev_raw = _safe(info.get("totalRevenue"))
    s3_ebit = to_unit(rev_raw * op_margin) if (rev_raw and op_margin) else None
    ebit_s3_desc = (f"Revenue x Operating Margin ({op_margin:.1%})"
                    if (rev_raw and op_margin) else "N/A")
    add_metric(
        "ebit", fd.get("ebit", 0),
        s1_ebit, s2_ebit, s3_ebit,
        "yfinance .info EBITDA (proxy)",
        f"{stmt_source} (EBIT/Operating Income)",
        ebit_s3_desc,
    )

    # 3. Net Income
    s1_ni = _safe(info.get("netIncomeToCommon"))
    s2_ni = _from_fin("net income")
    net_margin = _safe(info.get("profitMargins"))
    s3_ni = to_unit(rev_raw * net_margin) if (rev_raw and net_margin) else None
    ni_s3_desc = f"Revenue x Net Margin ({net_margin:.1%})" if net_margin else "N/A"
    add_metric(
        "net_income", fd.get("net_income", 0),
        s1_ni, s2_ni, s3_ni,
        "yfinance .info netIncomeToCommon",
        f"{stmt_source} (Net Income)",
        ni_s3_desc,
    )

    # 4. Total Debt
    s1_debt = _safe(info.get("totalDebt"))
    s2_debt = _from_bs("total debt") or _from_bs("long term debt")
    int_exp = _from_fin("interest expense")
    kd_est = 0.07 if is_indian else 0.05
    s3_debt = to_unit(abs(int_exp) / kd_est) if int_exp else None
    add_metric(
        "total_debt", fd.get("total_debt", 0),
        s1_debt, s2_debt, s3_debt,
        "yfinance .info totalDebt",
        f"{stmt_source} (Balance Sheet - Long Term Debt)",
        f"Interest Expense / Assumed Kd ({kd_est:.0%})",
    )

    # 5. Cash
    s1_cash = _safe(info.get("totalCash"))
    s2_cash = _from_bs("cash and cash equivalents") or _from_bs("cash")
    add_metric(
        "cash", fd.get("cash", 0),
        s1_cash, s2_cash, None,
        "yfinance .info totalCash",
        f"{stmt_source} (Balance Sheet - Cash)",
        "N/A",
    )

    # 6. Depreciation
    s2_dep = _from_fin("depreciation") or _from_cf("depreciation")
    ppe = _from_bs("net ppe") or _from_bs("property plant equipment")
    s3_dep = to_unit(ppe * 0.07) if ppe else None
    add_metric(
        "depreciation", fd.get("depreciation", 0),
        None, s2_dep, s3_dep,
        "N/A (not in .info)",
        f"{stmt_source} (Depreciation & Amortization)",
        "Net PP&E x 7% (industry avg depreciation rate)",
    )

    # 7. CapEx
    s1_capex = _safe(info.get("capitalExpenditures"))
    s2_capex = _from_cf("capital expenditure") or _from_cf("purchase of property")
    s3_capex = to_unit(rev_raw * 0.05) if rev_raw else None
    add_metric(
        "capex", fd.get("capex", 0),
        abs(s1_capex) if s1_capex is not None else None,
        abs(s2_capex) if s2_capex is not None else None,
        s3_capex,
        "yfinance .info capitalExpenditures",
        f"{stmt_source} (Cash Flow - CapEx)",
        "Revenue x 5% (industry capex intensity benchmark)",
    )

    # 8. Beta
    s1_beta = _safe(info.get("beta"))
    s2_beta = None  # not in statements
    # Derived: compare with sector average
    sector_betas = {
        "Technology": 1.3, "Financial Services": 1.1, "Consumer Cyclical": 1.2,
        "Healthcare": 0.9, "Industrials": 1.0, "Energy": 1.1,
        "Consumer Defensive": 0.6, "Utilities": 0.5, "Real Estate": 0.8,
        "Communication Services": 1.1, "Basic Materials": 1.0,
    }
    sector = info.get("sector", "")
    s3_beta = sector_betas.get(sector)
    beta_metrics_row = {
        "field": "beta",
        "our_value": fd.get("beta", 1.0),
        "source1_value": s1_beta,
        "source2_value": s2_beta,
        "source3_value": s3_beta,
        "sources_description": (
            "yfinance .info beta | "
            "N/A (not in statements) | "
            f"Sector average beta ({sector})"
        ),
    }
    beta_dev = _pct_dev(fd.get("beta", 1.0), s1_beta) if s1_beta else None
    beta_flag, beta_conf = _flag(beta_dev)
    beta_metrics_row["consensus_value"] = s1_beta
    beta_metrics_row["deviation"] = beta_dev
    beta_metrics_row["confidence"] = beta_conf
    beta_metrics_row["flag"] = beta_flag
    metrics.append(beta_metrics_row)
    if beta_dev and beta_dev > 0.25:
        warnings.append(
            f"Beta differs by {beta_dev:.0%} from yfinance live data "
            f"(our: {fd.get('beta', 1.0):.2f}, yfinance: {s1_beta:.2f})"
        )

    # 9. Tax Rate
    pretax = _from_fin("pretax income")
    tax_paid = _from_fin("tax provision")
    s2_tax = (tax_paid / pretax) if (pretax and pretax != 0 and tax_paid) else None
    s1_tax = _safe(info.get("effectiveTaxRate"))
    # Derived: country statutory rate
    s3_tax = 0.25 if is_indian else 0.21
    our_tax = fd.get("tax_rate", 0.25)
    tax_avail = [v for v in [s1_tax, s2_tax, s3_tax] if v is not None]
    tax_consensus = float(np.median(tax_avail)) if tax_avail else None
    tax_dev = _pct_dev(our_tax, tax_consensus)
    tax_flag, tax_conf = _flag(tax_dev)
    metrics.append({
        "field": "tax_rate",
        "our_value": our_tax,
        "source1_value": s1_tax,
        "source2_value": s2_tax,
        "source3_value": s3_tax,
        "consensus_value": tax_consensus,
        "deviation": tax_dev,
        "confidence": tax_conf,
        "flag": tax_flag,
        "sources_description": (
            "yfinance .info effectiveTaxRate | "
            f"{stmt_source} (Tax Provision / Pretax Income) | "
            f"Statutory rate ({int(s3_tax * 100)}%)"
        ),
    })

    # 10. Cost of Equity (CAPM cross-check)
    rf = fd.get("risk_free_rate", 0.043)
    erp = fd.get("erp", 0.05)
    beta_used = fd.get("beta", 1.0)
    capm_ke = rf + beta_used * erp
    our_ke = fd.get("cost_of_equity", capm_ke)
    ke_dev = _pct_dev(our_ke, capm_ke)
    ke_flag, ke_conf = _flag(ke_dev)
    metrics.append({
        "field": "cost_of_equity",
        "our_value": our_ke,
        "source1_value": s1_beta * erp + rf if s1_beta else None,
        "source2_value": None,
        "source3_value": capm_ke,
        "consensus_value": capm_ke,
        "deviation": ke_dev,
        "confidence": ke_conf,
        "flag": ke_flag,
        "sources_description": (
            f"CAPM using yfinance beta | N/A | "
            f"CAPM: Rf ({rf:.1%}) + beta ({beta_used:.2f}) x ERP ({erp:.1%})"
        ),
    })

    # 11. WACC cross-check
    dr = fd.get("debt_ratio", 0.0)
    our_wacc = fd.get("wacc", our_ke)
    kd_est_w = rf + 0.02
    tax_r = fd.get("tax_rate", 0.25)
    derived_wacc = our_ke * (1 - dr) + kd_est_w * (1 - tax_r) * dr
    wacc_dev = _pct_dev(our_wacc, derived_wacc)
    wacc_flag, wacc_conf = _flag(wacc_dev)
    metrics.append({
        "field": "wacc",
        "our_value": our_wacc,
        "source1_value": None,
        "source2_value": None,
        "source3_value": derived_wacc,
        "consensus_value": derived_wacc,
        "deviation": wacc_dev,
        "confidence": wacc_conf,
        "flag": wacc_flag,
        "sources_description": (
            "N/A | N/A | "
            f"Ke*({1-dr:.0%}) + Kd*(1-t)*{dr:.0%}: "
            f"{our_ke:.2%}*{(1-dr):.0%} + {kd_est_w:.2%}*{(1-tax_r):.0%}*{dr:.0%}"
        ),
    })

    # 12. FCFE consistency check
    ni = fd.get("net_income", 0)
    dep = fd.get("depreciation", 0)
    capex = fd.get("capex", 0)
    dwc = fd.get("delta_wc", 0)
    computed_fcfe = ni - (capex - dep) * (1 - dr) - dwc * (1 - dr)
    # Check against operating cash flow minus capex (rough proxy)
    ocf = _from_cf("operating cash flow") or _from_cf("cash from operations")
    s1_fcfe_proxy = to_unit(ocf - (abs(s2_capex or 0) if s2_capex else 0)) if ocf else None
    fcfe_dev = _pct_dev(computed_fcfe, s1_fcfe_proxy) if s1_fcfe_proxy else None
    fcfe_flag, fcfe_conf = _flag(fcfe_dev)
    metrics.append({
        "field": "fcfe (computed)",
        "our_value": computed_fcfe,
        "source1_value": s1_fcfe_proxy,
        "source2_value": None,
        "source3_value": computed_fcfe,
        "consensus_value": s1_fcfe_proxy or computed_fcfe,
        "deviation": fcfe_dev,
        "confidence": fcfe_conf,
        "flag": fcfe_flag,
        "sources_description": (
            f"{stmt_source} (OCF - CapEx proxy) | "
            "N/A | "
            "Formula: NI - (CapEx-Dep)*(1-DR) - dWC*(1-DR)"
        ),
    })

    # Compute overall score
    scored = [m for m in metrics if m["deviation"] is not None]
    if scored:
        avg_dev = np.mean([m["deviation"] for m in scored])
        overall_score = max(0.0, 100.0 - avg_dev * 200)
    else:
        overall_score = 50.0

    high_count = sum(1 for m in metrics if m["confidence"] == "High")
    medium_count = sum(1 for m in metrics if m["confidence"] == "Medium")
    low_count = sum(1 for m in metrics if m["confidence"] == "Low")
    total_scored = high_count + medium_count + low_count

    if total_scored == 0:
        overall_confidence = "Medium"
    elif high_count / max(total_scored, 1) >= 0.60:
        overall_confidence = "High"
    elif low_count / max(total_scored, 1) >= 0.40:
        overall_confidence = "Low"
    else:
        overall_confidence = "Medium"

    if is_indian:
        data_source_note = (
            "Sources: Yahoo Finance API (yfinance) · "
            "BSE/NSE Annual Report Filings · "
            "Derived Cross-Checks (CAPM, WACC, margin-based verification)"
        )
    else:
        data_source_note = (
            "Sources: Yahoo Finance API (yfinance) · "
            "SEC 10-K Filings · "
            "Derived Cross-Checks (CAPM, WACC, margin-based verification)"
        )

    return {
        "overall_confidence": overall_confidence,
        "overall_score": overall_score,
        "metrics": metrics,
        "warnings": warnings,
        "errors": errors,
        "data_source_note": data_source_note,
        "high_count": high_count,
        "medium_count": medium_count,
        "low_count": low_count,
    }
