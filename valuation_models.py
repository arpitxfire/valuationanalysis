"""
Damodaran DCF Valuation Models
==============================
Implements:
  - Model Selector (model1.xls logic) with full Q&A trail and RICH 3-5 sentence rationale
  - DDM   (Stable, 2-stage, 3-stage) with year-by-year tables
  - FCFE  (Stable, 2-stage, 3-stage) with year-by-year tables
  - FCFF  (Stable, 2-stage, 3-stage) with year-by-year tables

Every model returns:
  - intrinsic value
  - year_by_year: list of dicts (one per year) for tabular display
  - summary: key aggregates
"""

import numpy as np


# ═════════════════════════════════════════════════════════════════════════════
#  MODEL SELECTOR with full decision trail and rich Damodaran rationale
# ═════════════════════════════════════════════════════════════════════════════

def choose_valuation_model(inputs: dict) -> dict:
    """
    Replicates Damodaran's 'Choosing the Right Valuation Model' (model1.xls).
    Returns the model choice AND the full Q&A decision trail with rich rationale.
    """

    economy_growth = inputs["inflation_rate"] + inputs["real_growth_rate"]
    firm_g = inputs["firm_growth_rate"]
    dr = inputs["debt_ratio"]
    cur = inputs.get("currency", "$")
    unit = inputs.get("unit", "")

    # Compute FCFE
    fcfe = (
        inputs["net_income"]
        - (inputs["capex"] - inputs["depreciation"]) * (1 - dr)
        - inputs["delta_wc"] * (1 - dr)
    )

    # ── Build Q&A trail ─────────────────────────────────────────────────────
    qa = []
    qa.append({
        "question": "Level of Earnings",
        "answer": f"{cur}{inputs['net_income']:,.0f} {unit}",
        "note": (
            "The starting point of Damodaran's model selection framework (model1.xls) is always "
            "the current level of earnings. This determines whether we can use current earnings as "
            "the base case for DCF projections, or whether we need to first normalize or estimate "
            "future earnings using a revenue-growth path."
        ),
    })
    qa.append({
        "question": "Are your earnings positive?",
        "answer": "Yes" if inputs["earnings_positive"] else "No",
        "note": (
            "Positive earnings are the prerequisite for standard DCF models. When earnings are "
            "positive, we can use them directly as the base for projecting future cash flows. "
            "When negative, Damodaran prescribes diagnosing the *cause* before choosing a model — "
            "cyclical downturns, startup losses, excess leverage, and structural decline each "
            "require a different treatment."
        ),
    })

    if inputs["earnings_positive"]:
        qa.append({
            "question": "What is the expected inflation rate in the economy?",
            "answer": f"{inputs['inflation_rate']:.2%}",
            "note": (
                "Inflation is a key input because it determines the 'floor' for nominal growth "
                "rates. A firm growing below inflation in real terms is actually shrinking. "
                "Damodaran uses the country's expected inflation rate (not historical) as the "
                "base when computing the economy's nominal growth rate. For India, the RBI's "
                "medium-term target of ~5-6% is used; for the US, the Fed's 2% target is standard."
            ),
        })
        qa.append({
            "question": "What is the expected real growth rate in the economy?",
            "answer": f"{inputs['real_growth_rate']:.2%}",
            "note": (
                "Real GDP growth represents the economy's productive capacity expansion. "
                "India's long-run real growth is ~6%, while the US averages ~2%. These rates "
                "are crucial because no firm can grow faster than the overall economy indefinitely "
                "— this is the mathematical constraint that forces all high-growth firms to "
                "eventually converge to the stable growth rate in terminal value calculations."
            ),
        })
        qa.append({
            "question": "Implied nominal growth rate of the economy (inflation + real growth)?",
            "answer": f"{economy_growth:.2%}",
            "note": (
                f"Nominal GDP growth = inflation ({inputs['inflation_rate']:.1%}) + real growth "
                f"({inputs['real_growth_rate']:.1%}) = {economy_growth:.1%}. This is the critical "
                "benchmark Damodaran uses to classify a firm as 'stable', 'moderate growth', or "
                "'high growth'. A firm growing at or below this rate has already reached stable "
                "state. Growing at 1.5-2× this rate implies two-stage modeling. Growing above "
                "2× with competitive advantages implies three-stage modeling with a transition period."
            ),
        })
        qa.append({
            "question": "What is the expected growth rate in earnings for this firm?",
            "answer": f"{firm_g:.2%}",
            "note": (
                f"The firm's estimated growth rate of {firm_g:.1%} is compared against the economy's "
                f"nominal growth of {economy_growth:.1%}. This ratio — firm growth / economy growth = "
                f"{firm_g/economy_growth:.1f}× — drives the key fork between stable, two-stage, and "
                "three-stage models. Growth rates above 2× economy growth are typically unsustainable "
                "beyond 5-10 years because competition, capital constraints, and market saturation "
                "eventually pull all firms toward the economy's growth rate."
            ),
        })
        qa.append({
            "question": "Does this firm have a significant and sustainable competitive advantage?",
            "answer": "Yes" if inputs["has_competitive_adv"] else "No",
            "note": (
                "Differential advantages include: legal monopoly, patented technology, strong brand "
                "name, network effects, or economies of scale — both existing and future. Damodaran "
                "requires competitive advantages as a prerequisite for the three-stage model because "
                "only firms with defensible moats can sustain above-economy growth for 5+ years. "
                "Without such advantages, high growth is assumed to revert quickly to stable growth, "
                "making the two-stage model (or even stable model) more appropriate."
            ),
        })
    else:
        qa.append({
            "question": "Are the earnings negative because the firm is in a cyclical business?",
            "answer": "Yes" if inputs.get("cyclical_negative") else "No",
            "note": (
                "Cyclical firms (autos, steel, chemicals, semiconductors) experience earnings "
                "swings tied to the economic cycle. Damodaran's prescription is to normalize "
                "earnings over a full cycle — typically using the average earnings over 5-10 years "
                "or the mid-cycle earnings. Using trough earnings would vastly undervalue the firm; "
                "using peak earnings would overvalue it. The normalized base then enters the DCF."
            ),
        })
        qa.append({
            "question": "Are the earnings negative because of a one-time or temporary occurrence?",
            "answer": "Yes" if inputs.get("temporary_negative") else "No",
            "note": (
                "One-time charges — restructuring costs, write-downs, litigation settlements, "
                "natural disaster losses — can make an otherwise profitable firm appear loss-making. "
                "Damodaran strips these out and normalizes earnings to what the firm would earn "
                "in a typical year. The key test: will these charges recur? If yes, they are not "
                "truly one-time and must remain in the earnings base."
            ),
        })
        qa.append({
            "question": "Are the earnings negative because the firm has too much debt?",
            "answer": "Yes" if inputs.get("excess_debt_negative") else "No",
            "note": (
                "Excess leverage can make a fundamentally sound firm look loss-making due to heavy "
                "interest charges overwhelming operating profits. Damodaran's approach depends "
                "critically on whether bankruptcy is likely. If the firm can service debt and survive, "
                "we normalize earnings. If bankruptcy is likely, an option pricing model (treating "
                "equity as a call on firm assets) is more appropriate, as standard DCF ignores "
                "the possibility of the firm defaulting before generating positive cash flows."
            ),
        })
        if inputs.get("excess_debt_negative"):
            qa.append({
                "question": "If yes, is there a strong likelihood of bankruptcy?",
                "answer": "Yes" if inputs.get("bankruptcy_likely") else "No",
                "note": (
                    "Bankruptcy risk fundamentally changes the valuation framework. When bankruptcy "
                    "is likely, the Black-Scholes option pricing model treats equity as a call option "
                    "on the firm's assets with a strike price equal to the face value of debt. "
                    "DCF methods undervalue highly leveraged distressed firms because they assume "
                    "cash flows continue indefinitely — but equity holders lose everything if the firm "
                    "cannot meet its debt obligations before the cash flows materialize."
                ),
            })
        qa.append({
            "question": "Are the earnings negative because the firm is just starting up?",
            "answer": "Yes" if inputs.get("startup_negative") else "No",
            "note": (
                "Startup and early-stage firms invest heavily before generating revenues, producing "
                "negative earnings that reflect future potential, not permanent impairment. Damodaran "
                "values these firms using a 'bottom-up' approach: project the path from current "
                "revenue to a target revenue in year 5-10, apply expected margins at maturity, and "
                "discount back. The terminal value dominates the valuation, making assumptions about "
                "addressable market size and long-run margins the most critical inputs."
            ),
        })

    # Financial leverage section
    qa.append({
        "section": "Financial Leverage",
        "question": "What is the current debt ratio (in market value terms)?",
        "answer": f"{dr:.2%}",
        "note": (
            f"Debt ratio = Debt / (Debt + Equity) = {dr:.1%} in market value terms. "
            "Damodaran always uses market value (not book value) debt ratios because book value "
            "reflects historical cost, not current economic reality. The debt ratio determines "
            "whether to value equity directly via FCFE (stable or declining debt) or value the "
            "entire firm via FCFF and subtract debt (when leverage is changing significantly). "
            f"A debt ratio of {dr:.1%} is {'high (>40%)' if dr > 0.40 else 'moderate' if dr > 0.20 else 'low (<20%)'}."
        ),
    })
    qa.append({
        "question": "Is this debt ratio expected to change significantly?",
        "answer": "Yes" if inputs["debt_ratio_changing"] else "No",
        "note": (
            "When the debt ratio is stable, FCFE models work elegantly — we model equity cash "
            "flows directly. But when leverage is changing (firm is deleveraging, taking on more "
            "debt for acquisitions, or doing a leveraged buyout), the year-by-year FCFE calculation "
            "becomes complex because the equity fraction of financing changes each period. FCFF "
            "sidesteps this by valuing the entire operating business first, then subtracting debt "
            "at the end — a much cleaner approach when capital structure is in flux."
        ),
    })

    # Dividend policy section
    qa.append({
        "section": "Dividend Policy",
        "question": "What did the firm pay out as dividends in the current year?",
        "answer": f"{cur}{inputs['dividends']:,.2f} {unit}",
        "note": (
            "Dividends matter in model selection because the DDM (Dividend Discount Model) is only "
            "appropriate when dividends are meaningful and stable relative to earnings. For firms "
            "that pay no or negligible dividends — common among growth companies and buyback-focused "
            "US firms — the DDM would massively undervalue the stock since it ignores the earnings "
            "being retained and reinvested. In such cases, FCFE or FCFF models capture the full "
            "value including reinvested earnings."
        ),
    })
    qa.append({
        "question": "Can you estimate capital expenditures and working capital requirements?",
        "answer": "Yes" if inputs["can_estimate_capex"] else "No",
        "note": (
            "FCFE and FCFF models require explicit estimates of capital expenditures and working "
            "capital changes because these cash outflows reduce the free cash flow available to "
            "equity or firm. For most publicly listed companies, these are directly available from "
            "the cash flow statement and balance sheet. When unavailable (e.g., very new listings), "
            "Damodaran recommends using industry averages for reinvestment rate (capex - depreciation "
            "as % of revenue) and non-cash working capital as % of revenue."
        ),
    })

    # FCFE computation trail
    qa.append({
        "section": "FCFE Computation",
        "question": "Net Income (NI)",
        "answer": f"{cur}{inputs['net_income']:,.2f}",
    })
    qa.append({
        "question": "Depreciation and Amortization",
        "answer": f"{cur}{inputs['depreciation']:,.2f}",
    })
    qa.append({
        "question": "Capital Spending (incl. acquisitions)",
        "answer": f"{cur}{inputs['capex']:,.2f}",
    })
    qa.append({
        "question": "Δ Non-cash Working Capital (ΔWC)",
        "answer": f"{cur}{inputs['delta_wc']:,.2f}",
    })
    qa.append({
        "question": "FCFE = NI - (CapEx - Dep)×(1-DR) - ΔWC×(1-DR)",
        "answer": f"{cur}{fcfe:,.2f}",
        "formula": (f"= {inputs['net_income']:,.2f} "
                    f"- ({inputs['capex']:,.2f} - {inputs['depreciation']:,.2f})×(1-{dr:.4f}) "
                    f"- {inputs['delta_wc']:,.2f}×(1-{dr:.4f})"),
        "note": (
            "FCFE is the cash flow left for equity holders after paying all operating expenses, "
            "taxes, reinvestment needs (CapEx - Depreciation), and the equity portion of working "
            "capital changes. The debt ratio (1-DR) factor captures what fraction of reinvestment "
            "is funded by equity vs. debt. If FCFE significantly exceeds dividends paid, the DDM "
            "would undervalue the firm — FCFE is the better equity valuation metric in that case."
        ),
    })

    # ── Decision logic ──────────────────────────────────────────────────────
    result = {
        "model_type": "",
        "earnings_level": "",
        "cashflow_type": "",
        "growth_period": "",
        "growth_pattern": "",
        "model_code": "",
        "model_description": "",
        "decision_trail": [],
        "qa_inputs": qa,
    }

    trail = []

    # ── Step 1: Earnings check ───────────────────────────────────────────────
    if inputs["earnings_positive"]:
        result["model_type"] = "Discounted CF Model"
        result["earnings_level"] = "Current Earnings"
        trail.append(
            f"✅ Earnings are positive: Net Income = {cur}{inputs['net_income']:,.0f} {unit}. "
            "In Damodaran's model selection framework (model1.xls), the first fork asks whether "
            "the firm has positive earnings. Positive earnings allow us to use a standard "
            "Discounted Cash Flow model with current earnings as the base. Had earnings been "
            "negative, we would need to diagnose the cause — cyclical downturn, excess leverage, "
            "startup phase, or temporary event — each leading to a different normalization approach "
            "or even an option pricing model for distressed firms."
        )
    else:
        if inputs.get("cyclical_negative") or inputs.get("temporary_negative"):
            result["model_type"] = "Discounted CF Model"
            result["earnings_level"] = "Normalized Earnings"
            trail.append(
                f"⚠️ Earnings are negative: Net Income = {cur}{inputs['net_income']:,.0f} {unit}, "
                "but the cause is cyclical/temporary. Per Damodaran, when a fundamentally sound "
                "firm has negative earnings due to the economic cycle or one-time charges, we "
                "normalize earnings over the cycle (average of 5-10 years) and use those as the "
                "DCF base. This avoids the error of extrapolating trough earnings as a permanent "
                "state. The model structure (stable/two/three-stage) is then determined by the "
                "normalized growth outlook, not the current negative earnings."
            )
        elif inputs.get("excess_debt_negative"):
            if inputs.get("bankruptcy_likely"):
                result["model_type"] = "Option Pricing Model"
                result["earnings_level"] = "Current Earnings"
                trail.append(
                    f"🔴 Earnings are negative (Net Income = {cur}{inputs['net_income']:,.0f} {unit}) "
                    "due to excess debt, AND bankruptcy is likely. Damodaran prescribes the "
                    "Black-Scholes option pricing model here: equity is a call option on firm assets "
                    "with the face value of debt as the strike price. Standard DCF fails because it "
                    "assumes the firm generates cash flows indefinitely, ignoring the scenario where "
                    "debt holders foreclose before equity holders receive anything. The equity value "
                    "under option pricing can be positive even when the firm's net asset value is "
                    "negative — reflecting the probability of a turnaround."
                )
            else:
                result["model_type"] = "Discounted CF Model"
                result["earnings_level"] = "Normalized Earnings"
                trail.append(
                    f"⚠️ Earnings are negative (Net Income = {cur}{inputs['net_income']:,.0f} {unit}) "
                    "due to excess debt, but bankruptcy is NOT likely. Damodaran treats this as "
                    "a capital structure problem, not a business problem. We normalize earnings to "
                    "what the firm would earn at an optimal (lower) debt level, then apply DCF. "
                    "The firm's operating assets are fundamentally sound; the negative earnings "
                    "reflect financial leverage, not operational failure. DCF with normalized "
                    "earnings captures this correctly."
                )
        elif inputs.get("startup_negative"):
            result["model_type"] = "Discounted CF Model"
            result["earnings_level"] = "Current Earnings"
            trail.append(
                f"⚠️ Startup / early-stage firm: Net Income = {cur}{inputs['net_income']:,.0f} {unit}. "
                "Negative earnings reflect planned investment in future growth, not structural failure. "
                "Per Damodaran, startup valuation uses a revenue-path DCF: project revenue from "
                "current level to a target market share in year 5-10, apply target operating margins "
                "at maturity, and discount cash flows back. The terminal value dominates valuation, "
                "making assumptions about total addressable market and long-run profitability the "
                "most important inputs to stress-test."
            )
        else:
            result["model_type"] = "Discounted CF Model"
            result["earnings_level"] = "Normalized Earnings"
            trail.append(
                f"⚠️ Earnings are negative (Net Income = {cur}{inputs['net_income']:,.0f} {unit}) "
                "for unspecified reasons. Defaulting to normalized earnings DCF per Damodaran. "
                "It is important to investigate the cause before accepting this default — the "
                "appropriate normalization method differs significantly between cyclical, structural, "
                "and leverage-driven negative earnings scenarios."
            )

    # ── Step 2: Cashflow type ────────────────────────────────────────────────
    if inputs["debt_ratio_changing"]:
        result["cashflow_type"] = "FCFF (Value firm)"
        trail.append(
            f"📊 Debt ratio ({dr:.1%}) is expected to change significantly → Use FCFF to value "
            "the entire firm, then subtract debt. Damodaran's key insight here: when a firm is "
            "actively changing its capital structure (deleveraging after acquisitions, leveraging "
            "up for expansion, or doing a recapitalization), the FCFE calculation becomes unstable "
            "because the equity fraction of reinvestment changes each year. FCFF avoids this by "
            "treating the firm's operating assets as independent of its financing, discounting at "
            f"WACC. Once firm value is computed, we subtract total debt ({cur}{inputs.get('net_income',0)*0:,.0f}) "
            "and add cash to arrive at equity value per share."
        )
    elif inputs["can_estimate_capex"]:
        div_payout = inputs["dividends"]
        if abs(fcfe) > 0 and abs(fcfe - div_payout) / max(abs(fcfe), 1) > 0.20:
            result["cashflow_type"] = "FCFE (Value equity)"
            trail.append(
                f"📊 Debt ratio ({dr:.1%}) is stable AND dividends ({cur}{div_payout:,.0f} {unit}) "
                f"differ significantly from FCFE ({cur}{fcfe:,.2f}). "
                "Damodaran's rule: when dividends ≠ FCFE by >20%, use FCFE rather than DDM. "
                "A firm may pay lower dividends than FCFE because it is retaining cash for "
                "future growth (FCFE > Dividends), or it may pay more than FCFE by drawing down "
                "cash or borrowing (Dividends > FCFE, unsustainable). In both cases, FCFE better "
                "represents what equity holders are truly entitled to, making it the superior "
                "equity valuation metric. DDM would misjudge the firm's true earning power."
            )
        else:
            result["cashflow_type"] = "Dividends"
            trail.append(
                f"📊 Dividends ({cur}{div_payout:,.0f} {unit}) closely match FCFE ({cur}{fcfe:,.2f}). "
                "When a firm consistently pays out close to what it can afford (dividends ≈ FCFE), "
                "the DDM and FCFE model will give similar results. Damodaran's preference in this "
                "case is the DDM because dividends are directly observable and require no "
                "assumptions about what management *could* pay — they reflect actual behavior. "
                "This is most common for mature, stable dividend-paying companies with consistent "
                "payout policies and predictable earnings."
            )
    else:
        result["cashflow_type"] = "FCFF (Value firm)"
        trail.append(
            "📊 Cannot reliably estimate CapEx/Working Capital → defaulting to FCFF. "
            "Per Damodaran, when firm-specific reinvestment data is unavailable, FCFF with "
            "industry-average reinvestment rates is more reliable than FCFE with noisy inputs."
        )

    # ── Step 3: Growth pattern ────────────────────────────────────────────────
    cf = result["cashflow_type"]
    ratio = firm_g / economy_growth if economy_growth > 0 else 1.0

    if ratio <= 1.0:
        result["growth_period"] = "Stable Growth"
        result["growth_pattern"] = "Stable Growth"
        trail.append(
            f"📈 Firm growth ({firm_g:.1%}) ≤ Economy growth ({economy_growth:.1%}): "
            f"Ratio = {ratio:.2f}×. The firm's expected earnings growth rate of {firm_g:.1%} "
            f"is at or below the nominal GDP growth rate of {economy_growth:.1%} "
            f"(inflation {inputs['inflation_rate']:.1%} + real growth {inputs['real_growth_rate']:.1%}). "
            "Per Damodaran, when a firm grows at roughly the economy's rate, it has reached stable "
            "state — there is no excess growth to model separately. This means a single-stage "
            "(Gordon Growth) variant is appropriate, greatly simplifying the model. The terminal "
            "value captures the entire value of the firm as a perpetuity at the stable growth rate. "
            "Had growth been 1.5-2× the economy rate, we'd use two-stage; above 2× with competitive "
            "advantages, three-stage."
        )
    elif ratio < 2.0 or not inputs["has_competitive_adv"]:
        result["growth_period"] = "High Growth (2 stages)"
        result["growth_pattern"] = "Two-stage Growth"
        trail.append(
            f"📈 Firm growth ({firm_g:.1%}) vs Economy growth ({economy_growth:.1%}): "
            f"Ratio = {ratio:.2f}×. "
            f"{'The firm lacks a strong competitive advantage, ' if not inputs['has_competitive_adv'] else ''}"
            f"{'so high growth is unlikely to be sustained for a long transition period. ' if not inputs['has_competitive_adv'] else ''}"
            "Damodaran's two-stage model applies when growth significantly exceeds the economy "
            f"but is not exceptional enough to warrant a transition period. Growth of {firm_g:.1%} "
            "is projected for a finite high-growth phase (typically 7 years), after which the "
            "firm is assumed to converge directly to stable growth. The terminal value at the end "
            "of year 7 captures the stable-growth perpetuity. This is the most common model used "
            "for large-cap growth companies with moderate but not extraordinary competitive advantages."
        )
    else:
        result["growth_period"] = "High Growth (3 stages)"
        result["growth_pattern"] = "Three-stage Growth"
        trail.append(
            f"📈 Firm growth ({firm_g:.1%}) is {ratio:.1f}× economy growth ({economy_growth:.1%}), "
            "AND the firm has significant competitive advantages. "
            "Damodaran's three-stage model is reserved for firms with exceptional growth that "
            "can be defended for an extended period due to strong moats (patents, network effects, "
            f"brand, scale). The three phases: Phase 1 ({firm_g:.1%} for 5 years, high growth), "
            "Phase 2 (5-year transition where growth linearly declines toward stable growth), "
            f"Phase 3 (stable growth at {economy_growth:.1%} in perpetuity). The transition phase "
            "prevents the unrealistic assumption of an abrupt shift from high to stable growth — "
            "in reality, competitive advantages erode gradually as competitors respond."
        )

    # ── Map to model code ────────────────────────────────────────────────────
    pat = result["growth_pattern"]
    model_map = {
        ("Dividends", "Stable Growth"):                 ("ddmst",   "Stable Growth DDM (Gordon Growth Model)"),
        ("Dividends", "Two-stage Growth"):              ("ddm2st",  "Two-Stage Dividend Discount Model"),
        ("Dividends", "Three-stage Growth"):            ("ddm3st",  "Three-Stage Dividend Discount Model"),
        ("FCFE (Value equity)", "Stable Growth"):       ("fcfest",  "Stable Growth FCFE Model"),
        ("FCFE (Value equity)", "Two-stage Growth"):    ("fcfe2st", "Two-Stage FCFE Discount Model"),
        ("FCFE (Value equity)", "Three-stage Growth"):  ("fcfe3st", "Three-Stage FCFE Discount Model"),
        ("FCFF (Value firm)", "Stable Growth"):         ("fcffst",  "Stable Growth FCFF Model"),
        ("FCFF (Value firm)", "Two-stage Growth"):      ("fcff2st", "Two-Stage FCFF Discount Model"),
        ("FCFF (Value firm)", "Three-stage Growth"):    ("fcff3st", "Three-Stage FCFF Discount Model"),
    }

    key = (cf, pat)
    if key in model_map:
        result["model_code"], result["model_description"] = model_map[key]
    else:
        result["model_code"] = "fcff2st"
        result["model_description"] = "Two-Stage FCFF (Default Fallback)"

    trail.append(
        f"✅ SELECTED MODEL: **{result['model_description']}** (`{result['model_code']}.xls`). "
        f"This model was selected because: earnings are {'positive' if inputs['earnings_positive'] else 'negative/normalized'}, "
        f"cash flow type is '{result['cashflow_type']}', and growth pattern is '{result['growth_pattern']}'. "
        "The model code corresponds to Damodaran's spreadsheet convention where 'st' = stable, "
        "'2st' = two-stage, '3st' = three-stage. The selected model optimally balances "
        "model complexity with data reliability — adding more stages only when justified "
        "by the firm's competitive position and growth differential over the economy."
    )

    result["decision_trail"] = trail
    return result


# ═════════════════════════════════════════════════════════════════════════════
#  HELPER: compute FCFE & FCFF
# ═════════════════════════════════════════════════════════════════════════════

def compute_fcfe(net_income, depreciation, capex, delta_wc, debt_ratio):
    return net_income - (capex - depreciation) * (1 - debt_ratio) - delta_wc * (1 - debt_ratio)


def compute_fcff(ebit, tax_rate, depreciation, capex, delta_wc):
    return ebit * (1 - tax_rate) + depreciation - capex - delta_wc


# ═════════════════════════════════════════════════════════════════════════════
#  DDM MODELS — with year-by-year tables
# ═════════════════════════════════════════════════════════════════════════════

def ddm_stable(dps, cost_of_equity, stable_growth):
    """Gordon Growth Model with explicit calculation."""
    ke, g = cost_of_equity, stable_growth
    if ke <= g:
        return {"error": "Cost of equity must exceed stable growth rate"}

    dps1 = dps * (1 + g)
    value = dps1 / (ke - g)

    year_by_year = [{
        "Year": "Terminal (∞)",
        "Dividend": dps1,
        "Growth Rate": g,
        "Discount Rate": ke,
        "PV Factor": "1/(Ke-g)",
        "Present Value": value,
    }]

    return {
        "intrinsic_value": value,
        "model": "Stable DDM (Gordon Growth Model)",
        "formula": f"Value = DPS₁ / (Ke - g) = {dps1:,.2f} / ({ke:.4f} - {g:.4f}) = {value:,.2f}",
        "year_by_year": year_by_year,
        "summary": {
            "Current DPS (D₀)": dps,
            "Next Year DPS (D₁)": dps1,
            "Cost of Equity (Ke)": ke,
            "Stable Growth Rate (g)": g,
            "Intrinsic Value per Share": value,
        },
    }


def ddm_two_stage(dps, cost_of_equity, high_growth, stable_growth,
                   high_growth_years=7):
    ke, hg, sg = cost_of_equity, high_growth, stable_growth
    rows = []
    pv_dividends = 0
    current_dps = dps

    for yr in range(1, high_growth_years + 1):
        current_dps *= (1 + hg)
        pv_factor = 1 / ((1 + ke) ** yr)
        pv = current_dps * pv_factor
        pv_dividends += pv
        rows.append({
            "Year": yr,
            "Expected Growth": f"{hg:.2%}",
            "Dividend (DPS)": current_dps,
            "Cost of Equity": f"{ke:.2%}",
            "PV Factor": pv_factor,
            "PV of Dividend": pv,
        })

    terminal_dps = current_dps * (1 + sg)
    terminal_value = terminal_dps / (ke - sg)
    pv_terminal = terminal_value / ((1 + ke) ** high_growth_years)

    rows.append({
        "Year": f"Terminal (Yr {high_growth_years}+)",
        "Expected Growth": f"{sg:.2%} (stable)",
        "Dividend (DPS)": terminal_dps,
        "Cost of Equity": f"{ke:.2%}",
        "PV Factor": 1 / ((1 + ke) ** high_growth_years),
        "PV of Dividend": pv_terminal,
    })

    intrinsic = pv_dividends + pv_terminal

    return {
        "intrinsic_value": intrinsic,
        "model": "Two-Stage DDM",
        "year_by_year": rows,
        "summary": {
            "Current DPS (D₀)": dps,
            "High Growth Rate": hg,
            "High Growth Period": f"{high_growth_years} years",
            "Stable Growth Rate": sg,
            "Cost of Equity": ke,
            "PV of High-Growth Dividends": pv_dividends,
            "Terminal Value": terminal_value,
            "PV of Terminal Value": pv_terminal,
            "Intrinsic Value per Share": intrinsic,
        },
    }


def ddm_three_stage(dps, cost_of_equity, high_growth, stable_growth,
                     high_years=5, transition_years=5):
    ke, hg, sg = cost_of_equity, high_growth, stable_growth
    rows = []
    pv_total = 0
    current_dps = dps
    year = 0

    # Phase 1
    for yr in range(1, high_years + 1):
        current_dps *= (1 + hg)
        pv_factor = 1 / ((1 + ke) ** yr)
        pv = current_dps * pv_factor
        pv_total += pv
        rows.append({
            "Year": yr, "Phase": "High Growth",
            "Growth Rate": f"{hg:.2%}", "DPS": current_dps,
            "PV Factor": pv_factor, "PV of DPS": pv,
        })
        year = yr

    pv_phase1 = pv_total

    # Phase 2
    pv_phase2 = 0
    for i in range(1, transition_years + 1):
        blended = hg - (hg - sg) * (i / transition_years)
        current_dps *= (1 + blended)
        year += 1
        pv_factor = 1 / ((1 + ke) ** year)
        pv = current_dps * pv_factor
        pv_total += pv
        pv_phase2 += pv
        rows.append({
            "Year": year, "Phase": "Transition",
            "Growth Rate": f"{blended:.2%}", "DPS": current_dps,
            "PV Factor": pv_factor, "PV of DPS": pv,
        })

    # Phase 3
    terminal_dps = current_dps * (1 + sg)
    terminal_value = terminal_dps / (ke - sg)
    pv_terminal = terminal_value / ((1 + ke) ** year)

    rows.append({
        "Year": f"Terminal (Yr {year}+)", "Phase": "Stable",
        "Growth Rate": f"{sg:.2%}", "DPS": terminal_dps,
        "PV Factor": 1 / ((1 + ke) ** year), "PV of DPS": pv_terminal,
    })

    intrinsic = pv_total + pv_terminal

    return {
        "intrinsic_value": intrinsic,
        "model": "Three-Stage DDM",
        "year_by_year": rows,
        "summary": {
            "Current DPS (D₀)": dps,
            "High Growth Rate": hg,
            "High Growth Years": high_years,
            "Transition Years": transition_years,
            "Stable Growth Rate": sg,
            "Cost of Equity": ke,
            "PV Phase 1 (High Growth)": pv_phase1,
            "PV Phase 2 (Transition)": pv_phase2,
            "Terminal Value": terminal_value,
            "PV of Terminal Value": pv_terminal,
            "Intrinsic Value per Share": intrinsic,
        },
    }


# ═════════════════════════════════════════════════════════════════════════════
#  FCFE MODELS — with year-by-year tables
# ═════════════════════════════════════════════════════════════════════════════

def fcfe_stable(fcfe_ps, cost_of_equity, stable_growth):
    ke, g = cost_of_equity, stable_growth
    if ke <= g:
        return {"error": "Cost of equity must exceed stable growth rate"}

    fcfe1 = fcfe_ps * (1 + g)
    value = fcfe1 / (ke - g)

    return {
        "intrinsic_value": value,
        "model": "Stable FCFE Model",
        "formula": f"Value = FCFE₁ / (Ke - g) = {fcfe1:,.2f} / ({ke:.4f} - {g:.4f}) = {value:,.2f}",
        "year_by_year": [{
            "Year": "Terminal (∞)",
            "FCFE": fcfe1, "Growth": g, "Ke": ke, "Value": value,
        }],
        "summary": {
            "Current FCFE/share": fcfe_ps,
            "Next Year FCFE/share": fcfe1,
            "Cost of Equity": ke,
            "Stable Growth": g,
            "Intrinsic Value per Share": value,
        },
    }


def fcfe_two_stage(fcfe_ps, cost_of_equity, high_growth, stable_growth,
                    high_years=7):
    ke, hg, sg = cost_of_equity, high_growth, stable_growth
    rows = []
    pv_fcfe = 0
    current = fcfe_ps

    for yr in range(1, high_years + 1):
        current *= (1 + hg)
        pv_factor = 1 / ((1 + ke) ** yr)
        pv = current * pv_factor
        pv_fcfe += pv
        rows.append({
            "Year": yr, "Growth": f"{hg:.2%}", "FCFE/Share": current,
            "PV Factor": pv_factor, "PV of FCFE": pv,
        })

    terminal = current * (1 + sg)
    tv = terminal / (ke - sg)
    pv_tv = tv / ((1 + ke) ** high_years)

    rows.append({
        "Year": f"Terminal (Yr {high_years}+)", "Growth": f"{sg:.2%} (stable)",
        "FCFE/Share": terminal, "PV Factor": 1 / ((1 + ke) ** high_years),
        "PV of FCFE": pv_tv,
    })

    intrinsic = pv_fcfe + pv_tv

    return {
        "intrinsic_value": intrinsic,
        "model": "Two-Stage FCFE Model",
        "year_by_year": rows,
        "summary": {
            "Current FCFE/share": fcfe_ps,
            "High Growth Rate": hg,
            "High Growth Period": f"{high_years} years",
            "Stable Growth Rate": sg,
            "Cost of Equity": ke,
            "PV of High-Growth FCFE": pv_fcfe,
            "Terminal Value": tv,
            "PV of Terminal Value": pv_tv,
            "Intrinsic Value per Share": intrinsic,
        },
    }


def fcfe_three_stage(fcfe_ps, cost_of_equity, high_growth, stable_growth,
                      high_years=5, transition_years=5):
    ke, hg, sg = cost_of_equity, high_growth, stable_growth
    rows = []
    pv_total = 0
    current = fcfe_ps
    year = 0

    pv_p1 = 0
    for yr in range(1, high_years + 1):
        current *= (1 + hg)
        pv_f = 1 / ((1 + ke) ** yr)
        pv = current * pv_f
        pv_total += pv
        pv_p1 += pv
        rows.append({"Year": yr, "Phase": "High Growth", "Growth": f"{hg:.2%}",
                      "FCFE/Share": current, "PV Factor": pv_f, "PV of FCFE": pv})
        year = yr

    pv_p2 = 0
    for i in range(1, transition_years + 1):
        blended = hg - (hg - sg) * (i / transition_years)
        current *= (1 + blended)
        year += 1
        pv_f = 1 / ((1 + ke) ** year)
        pv = current * pv_f
        pv_total += pv
        pv_p2 += pv
        rows.append({"Year": year, "Phase": "Transition", "Growth": f"{blended:.2%}",
                      "FCFE/Share": current, "PV Factor": pv_f, "PV of FCFE": pv})

    terminal = current * (1 + sg)
    tv = terminal / (ke - sg)
    pv_tv = tv / ((1 + ke) ** year)

    rows.append({"Year": f"Terminal (Yr {year}+)", "Phase": "Stable",
                  "Growth": f"{sg:.2%}", "FCFE/Share": terminal,
                  "PV Factor": 1 / ((1 + ke) ** year), "PV of FCFE": pv_tv})

    intrinsic = pv_total + pv_tv

    return {
        "intrinsic_value": intrinsic,
        "model": "Three-Stage FCFE Model",
        "year_by_year": rows,
        "summary": {
            "Current FCFE/share": fcfe_ps,
            "PV Phase 1 (High Growth)": pv_p1,
            "PV Phase 2 (Transition)": pv_p2,
            "Terminal Value": tv,
            "PV of Terminal Value": pv_tv,
            "Intrinsic Value per Share": intrinsic,
        },
    }


# ═════════════════════════════════════════════════════════════════════════════
#  FCFF MODELS — with year-by-year tables
# ═════════════════════════════════════════════════════════════════════════════

def fcff_stable(fcff, wacc, stable_growth, total_debt=0, cash=0,
                shares_outstanding=1):
    w, g = wacc, stable_growth
    if w <= g:
        return {"error": "WACC must exceed stable growth rate"}

    fcff1 = fcff * (1 + g)
    firm_value = fcff1 / (w - g)
    equity_value = firm_value - total_debt + cash
    per_share = equity_value / shares_outstanding

    return {
        "intrinsic_value_per_share": per_share,
        "firm_value": firm_value,
        "equity_value": equity_value,
        "model": "Stable FCFF Model",
        "formula": f"Firm Value = FCFF₁/(WACC-g) = {fcff1:,.2f}/({w:.4f}-{g:.4f}) = {firm_value:,.2f}",
        "year_by_year": [{
            "Year": "Terminal (∞)", "FCFF": fcff1, "Growth": g,
            "WACC": w, "Firm Value": firm_value,
        }],
        "summary": {
            "Current FCFF": fcff,
            "Next Year FCFF": fcff1,
            "WACC": w,
            "Stable Growth": g,
            "Firm Value": firm_value,
            "(-) Total Debt": total_debt,
            "(+) Cash": cash,
            "Equity Value": equity_value,
            "Shares Outstanding": shares_outstanding,
            "Intrinsic Value per Share": per_share,
        },
    }


def fcff_two_stage(fcff, wacc_high, wacc_stable, high_growth, stable_growth,
                    high_years=7, total_debt=0, cash=0, shares_outstanding=1):
    wh, ws, hg, sg = wacc_high, wacc_stable, high_growth, stable_growth
    rows = []
    pv_fcff = 0
    current = fcff

    for yr in range(1, high_years + 1):
        current *= (1 + hg)
        pv_f = 1 / ((1 + wh) ** yr)
        pv = current * pv_f
        pv_fcff += pv
        rows.append({
            "Year": yr, "Growth": f"{hg:.2%}", "FCFF": current,
            "WACC": f"{wh:.2%}", "PV Factor": pv_f, "PV of FCFF": pv,
        })

    terminal_fcff = current * (1 + sg)
    tv = terminal_fcff / (ws - sg)
    pv_tv = tv / ((1 + wh) ** high_years)

    rows.append({
        "Year": f"Terminal (Yr {high_years}+)", "Growth": f"{sg:.2%} (stable)",
        "FCFF": terminal_fcff, "WACC": f"{ws:.2%}",
        "PV Factor": 1 / ((1 + wh) ** high_years), "PV of FCFF": pv_tv,
    })

    firm_value = pv_fcff + pv_tv
    equity_value = firm_value - total_debt + cash
    per_share = equity_value / shares_outstanding

    return {
        "intrinsic_value_per_share": per_share,
        "firm_value": firm_value,
        "equity_value": equity_value,
        "model": "Two-Stage FCFF Model",
        "year_by_year": rows,
        "summary": {
            "Current FCFF": fcff,
            "High Growth Rate": hg,
            "High Growth Period": f"{high_years} years",
            "WACC (High Growth)": wh,
            "Stable Growth Rate": sg,
            "WACC (Stable)": ws,
            "PV of High-Growth FCFF": pv_fcff,
            "Terminal Value": tv,
            "PV of Terminal Value": pv_tv,
            "Firm Value (Enterprise Value)": firm_value,
            "(-) Total Debt": total_debt,
            "(+) Cash & Equivalents": cash,
            "Equity Value": equity_value,
            "Shares Outstanding": shares_outstanding,
            "Intrinsic Value per Share": per_share,
        },
    }


def fcff_three_stage(fcff, wacc_high, wacc_stable, high_growth, stable_growth,
                      high_years=5, transition_years=5,
                      total_debt=0, cash=0, shares_outstanding=1):
    wh, ws, hg, sg = wacc_high, wacc_stable, high_growth, stable_growth
    rows = []
    pv_total = 0
    current = fcff
    year = 0

    pv_p1 = 0
    for yr in range(1, high_years + 1):
        current *= (1 + hg)
        pv_f = 1 / ((1 + wh) ** yr)
        pv = current * pv_f
        pv_total += pv
        pv_p1 += pv
        rows.append({"Year": yr, "Phase": "High Growth", "Growth": f"{hg:.2%}",
                      "FCFF": current, "WACC": f"{wh:.2%}",
                      "PV Factor": pv_f, "PV of FCFF": pv})
        year = yr

    pv_p2 = 0
    for i in range(1, transition_years + 1):
        bg = hg - (hg - sg) * (i / transition_years)
        bw = wh - (wh - ws) * (i / transition_years)
        current *= (1 + bg)
        year += 1
        pv_f = 1 / ((1 + bw) ** year)
        pv = current * pv_f
        pv_total += pv
        pv_p2 += pv
        rows.append({"Year": year, "Phase": "Transition", "Growth": f"{bg:.2%}",
                      "FCFF": current, "WACC": f"{bw:.2%}",
                      "PV Factor": pv_f, "PV of FCFF": pv})

    terminal = current * (1 + sg)
    tv = terminal / (ws - sg)
    pv_tv = tv / ((1 + ws) ** year)

    rows.append({"Year": f"Terminal (Yr {year}+)", "Phase": "Stable",
                  "Growth": f"{sg:.2%}", "FCFF": terminal, "WACC": f"{ws:.2%}",
                  "PV Factor": 1 / ((1 + ws) ** year), "PV of FCFF": pv_tv})

    firm_value = pv_total + pv_tv
    equity_value = firm_value - total_debt + cash
    per_share = equity_value / shares_outstanding

    return {
        "intrinsic_value_per_share": per_share,
        "firm_value": firm_value,
        "equity_value": equity_value,
        "model": "Three-Stage FCFF Model",
        "year_by_year": rows,
        "summary": {
            "Current FCFF": fcff,
            "PV Phase 1 (High Growth)": pv_p1,
            "PV Phase 2 (Transition)": pv_p2,
            "Terminal Value": tv,
            "PV of Terminal Value": pv_tv,
            "Firm Value (Enterprise Value)": firm_value,
            "(-) Total Debt": total_debt,
            "(+) Cash & Equivalents": cash,
            "Equity Value": equity_value,
            "Shares Outstanding": shares_outstanding,
            "Intrinsic Value per Share": per_share,
        },
    }
