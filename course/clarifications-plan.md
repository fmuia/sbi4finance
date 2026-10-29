# Clarifications plan: Lectures 1, 2, 4, 5 and 7

29 October 2026. Owner: Francesco Muia. [Course index](README.md).

This plan collects the questions raised in a read-through of Lecture 1, mostly around Figures 1.6 and 1.7, and
says where each answer belongs. The rule for placement: the main body keeps what a reader needs to follow the argument,
in a sentence or two; derivations, worked numbers and conventions that only a careful reader will check go to a new
appendix. Nothing here changes a result.

## 1. Recap of the points raised

| # | Question | Short answer | Where | Priority |
| --- | --- | --- | --- | --- |
| 1 | Figure 1.7 is "all out of the money", yet it spans $k < 0$ and $k > 0$. | Out of the money depends on the option type. With $k = \ln(K/F)$, the bars at $k < 0$ are puts and those at $k > 0$ are calls; the same holds for $\delta$. Some texts define moneyness as $\ln(F/K)$ or $S/K$, which flips the sign. | Body: L1 §1.5 (sign convention) and Fig. 1.7 caption | High |
| 2 | What is the orange bar? | It is SPY's design noise, ±0.3 vol points, drawn at an arbitrary 25% (`course/scripts/realignment.py:207`). On a 0–80 axis it reads as a level, not a width. It misleads. | Figure fix (Section 2.1) | High |
| 3 | How are the error bars computed? Are they quoted? | Bid and ask are quoted (Cboe delayed quotes at the close). Each is inverted through Black–Scholes separately: a bar runs from IV(bid) to IV(ask), and from 0 when there is no bid. The dot is IV(mid), not the midpoint of the two IVs. They are not statistical error bars. | Body: Fig. 1.7 caption | High |
| 4 | What is an option chain? How many expiries are typical? | All listed contracts on one underlying: expiries × strikes × {call, put}. SPY: 13,114 options in 35 expirations (dailies, weeklies, monthlies, quarterlies, LEAPs to about 2–3 years), strikes about $1 apart. EWI: 4 expiries to 238 days, a few dozen strikes, 132 contracts traded that day. | Body: two sentences in L1 §1.6. Appendix A.1: expiry types, strike spacing, settlement | Medium |
| 5 | Why only expiries up to one year? | The design grid stops at 252 days. Liquidity is concentrated there; dividend, rate and early-exercise effects the models leave out grow with $T$; the course's products depend mostly on the first year. The long end is treated as an extrapolation choice (L15 §15.6). | Body: one sentence in L1 §1.6, next to the design grid | Medium |
| 6 | How were 63 contracts picked out of 13,114? | They were not picked. The grid (7 maturities × 9 $\delta$) is fixed in advance, and its prices are read off the cleaned chain through a smoothing spline (L5 §5.4). It is fixed because the inference network needs an input of fixed size, the same on simulated and real data. | Body: one sentence in L1 §1.6 (the "fixed because" reason is missing) | Medium |
| 7 | What is a variance-swap strip? | The fair variance-swap strike replicated by OTM options weighted by $1/K^2$, $K_{\text{var}} = (2/T)\sum_i \Delta K_i\, P_i/(D K_i^2)$, summed over the quoted strikes. On bids and on asks it gives a band, not a confidence interval. | Body: Fig. 1.7 caption gets one line, with the forward reference to L5 §5.9 it already has. Appendix A.4: worked example | Medium |
| 8 | Why do American puts carry an early-exercise premium, and calls (almost) not? Exercising either pays its intrinsic value immediately. | Exercising a call *pays* $K$ now; exercising a put *receives* $K$ now. With $r > 0$ one wants to pay late and receive early. Hence $C_{\text{Eu}} \ge S - Ke^{-rT} > S - K$, so a call is worth more alive than exercised, while $P_{\text{Eu}} \ge Ke^{-rT} - S$ can fall below $K - S$ deep in the money. | Body: L1 §1.3 replaces "for calls … little value; for puts it has some" with this reason in two sentences. Appendix A.2: the bounds and the proof | High |
| 9 | What is the dividend argument? | The share drops by $d$ at the ex-date; a shareholder receives $d$, a call holder does not. Exercising just before the ex-date pays when $d$ exceeds the interest on $K$ plus the remaining time value. The bound becomes $C \ge S - \mathrm{PV}(d) - Ke^{-rT}$. Dividends make early exercise of puts less attractive. | Body: one sentence in L1 §1.3. Appendix A.2 | Medium |
| 10 | Why is the premium small for OTM quotes? | An OTM option has no intrinsic value, so exercising it today pays nothing. Its premium comes only from the chance of becoming deep in the money before $T$. EWI's 29-day put at $k = -0.10$: 0.0036 of a half-spread; EWG: at most 0.31. | Body: one sentence in L1 §1.6, where OTM quotes are chosen | High |
| 11 | Is the premium quoted? | No. The screen shows the American price only. The premium is American minus European model price, from a CRR tree with escrowed dividends at the quote's own IV (`lab/checks/thin_chains/thin_chain.py:53`). Half-spread is $(\text{ask} - \text{bid})/2$; cleaning removes crossed quotes, so no absolute value is needed. | Appendix A.2 (method). Body: L2 §2.4 says "computed with a binomial tree", with a link | Low |
| 12 | Why use SPY, which is American, instead of SPX? | Fair objection. The reason given is the data in hand (`v2-plan.md`, decision table). The stronger argument: the thin chains (EWG, EWI) are American ETFs anyway, so the pipeline must handle American exercise regardless, and SPY runs the same pipeline on both. SPX brings its own conventions: AM-settled monthlies vs PM-settled SPXW, an implied dividend yield in the forward, the multiplier. SPX is a planned extension (v2-plan, option E). | Body: short "Why SPY" paragraph in L1 §1.6. Appendix A.5: SPX conventions | High |
| 13 | Why is $C - P = D(F - K)$? | Already derived in L2 §2.4: long call, short put pays $S_T - K$; receiving $S_T$ at $T$ is worth $DF$ and paying $K$ is worth $DK$. | Body: L1 §1.6 cites "L2 §2.4" where it invokes parity. No new text | Low |
| 14 | How does American exercise bias parity? | $C_{\text{Am}} - P_{\text{Am}} = D(F - K) + e_C(K) - e_P(K)$. With $e_C \approx 0$ and $e_P \approx e_0 + e_1 K$, $e_1 > 0$, the regression slope is $-(D + e_1)$, so $\hat D > D$ (up to 1.044 on SPY). With $D$ known, $\hat F = F - e_P/D < F$ (0.15–0.46% on EWG), which moves put vols down and call vols up. | Body: L2 §2.4 gets the three-line derivation (it now states the effect without the algebra) | High |
| 15 | What is the statistical model? Is there an observation error? | $y_c = P_c(\theta; \xi_0) + \varepsilon_c$, $\varepsilon_c \sim \mathcal N(0, \sigma_c^2)$, independent, $\sigma_c$ known. SPY: $\sigma_c = \mathrm{vega}_c(20\%) \times s_{\text{iv}}(\delta_c)$. Thin chain: the half-spread in price, floored at the design noise, tested at scales 0.5 and 2. | Already in L7 §7.1. Body: Fig. 1.7 caption says the half-spread becomes the noise sd in Lecture 16 | Low |
| 16 | What is $\mathrm{vega}_c$, and why this $\sigma_c$? What is the design noise? | $\mathrm{vega}_c = \sqrt{T}\varphi(d_1)$ at $\sigma_{\text{ref}} = 20\%$. Quote uncertainty is roughly constant in vol units, so the delta method maps a vol error $s_{\text{iv}}$ to a price error $\mathrm{vega} \times s_{\text{iv}}$. Design noise: $s_{\text{iv}}(\delta) = 0.3 + 0.35\,\lvert\delta\rvert$ vol points, chosen above SPY's real half-spreads at every grid point. Weak points: vega at 20% rather than the quote's own vol (up to 6× at $\lvert\delta\rvert = 2$), half-spread = 1 sd is a convention, a Gaussian is unbounded while a bid–ask interval is not. | Body: L5 §5.3 states the formula for $s_{\text{iv}}$ and the evidence for 0.3 and 1.0 (Section 2.3). L7 §7.1 lists the vega-at-20% weakness next to its four assumptions | High |
| 17 | Why do the models leave out dividends, rates and early exercise? | They don't leave out dividends and rates: those enter through $F$ and $D$, and the models live in forward units, $X_t = S_t/F_t$. What they leave out is *stochastic* rates and dividends, and early exercise. Reasons: second order up to a year next to the vol effects; only vol parameters to infer; early exercise checked small on OTM quotes; American pricing under rough Bergomi needs regression Monte Carlo, too slow for a likelihood or a training bank. | Body: a short "What the models leave out" paragraph in L1 §1.7 or L2 §2.5. L16 §16.1 condition 3 already states the test | Medium |
| 18 | How does the variance-swap strip work on the thin chain, and what are the approximations? | On EWI's 29-day expiry the bids give 14.2% and the asks 51.9%. Ask-only calls (strikes 57–71) supply 60% of the ask variance; the 71 call is offered at $1.75 with the forward at $55.2, a placeholder. Approximations: bid/ask band; truncation (both bounds biased low); discretization; American as European; continuous paths (jumps); discrete monitoring and day count; forward and $D$. | Appendix A.4: the worked table and the list. Body: L5 §5.9 gets one sentence on truncation and the band | Medium |
| 19 | Why are OTM puts so much more expensive than OTM calls (Figure 1.6)? | Two layers. Black–Scholes alone makes the put *cheaper*: $p(k) = e^{k}c(-k)$, so at $k = \mp 0.4$ the put costs 0.67 of the call; the put is more likely to pay (2.9% against 1.8%) but pays on the scale of a strike of $0.67F$ against $1.49F$. The market's skew reverses it: 28.7% against 14.8% implied volatility multiplies the put by 6.3 and divides the call by 10.9, a factor of 68, giving 45.8. Wing prices fall like $e^{-n^2/2}$ in $n = k/(\sigma\sqrt T)$, the distance in the option's *own* standard deviations: 1.39 for the put, 2.70 for the call. A put price is the area under the distribution function left of its strike, so expensive puts mean a fat left tail. | Body: one sentence in L1 §1.6 (line 158) with the 1.4 and 2.7 standard deviations; a link from L4 §4.8. Appendix A.6: the symmetry proof, the decomposition, the tail asymptotic, the area formula, the economic reasons | High |
| 20 | What is marking for? A desk values a vanilla at the close; then what? | The mark feeds everything that follows. The day's change in the marks is the desk's P&L, against which traders are paid and limits are set; the Greeks and value-at-risk are computed from the same marked positions; collateral with counterparties is exchanged on them; and they enter the bank's books at fair value, ranked by IFRS 13 into Level 1 (quoted), 2 (inputs observable) and 3 (unobservable inputs). Product control checks the marks independently. A mark-to-model position has no price to check against, so the bank holds a reserve for the uncertainty of its mark (L6 §6.6). | Body: two or three sentences after "Marking a book" in L1 §1.7 | High |

## 2. Changes, by file

### 2.1 Figure 1.7 (`course/scripts/realignment.py`, `thin_chain`)

1. Remove the orange bar and its legend entry.
2. Put the comparison in each panel's title, e.g. "median near-money width 36 pts (SPY design noise: 0.6 pts)".
3. Optionally, overlay SPY's actual bid–ask bars at the matching maturity, for a visual comparison at the same scale.
4. Mark puts and calls distinctly (e.g. a thin vertical line at $k = 0$ labelled "puts | calls"), so item 1 is visible in the figure itself.
5. Consider dropping the variance-swap strip from the titles here, since it is defined only in Lecture 5. If kept, the caption says it is a band from all-bids to all-asks over the quoted strikes, biased low by truncation.

### 2.2 Lecture 1 (`01-options-quotes-and-volatility.md`)

- **§1.3, "European and American exercise"** (line 98): the pay-$K$ / receive-$K$ reason for the call–put asymmetry, and the dividend exception, in two or three sentences (items 8, 9). Link to Appendix A.2.
- **§1.5, moneyness** (line 136): state that out of the money means $k < 0$ for puts and $k > 0$ for calls under $k = \ln(K/F)$, and that some texts use the opposite sign (item 1).
- **§1.6, "What a market quotes"**:
  - Two sentences on chain anatomy (item 4).
  - At the OTM paragraph (line 150): early-exercise value is concentrated in the money, which is one more reason to keep OTM quotes (item 10). Cite L2 §2.4 for parity (item 13).
  - A short "Why SPY and not SPX" paragraph (item 12).
  - At the design grid (line 152): fixed in advance, read off rather than selected, because the network needs a fixed input (item 6); and why it stops at one year (item 5).
  - Figure 1.7 caption (line 164): what the bars are (item 3); puts left, calls right (item 1); the strip as a band (item 7); the half-spread as noise sd in Lecture 16 (item 15); the orange bar removed (item 2).
- **§1.6, the asymmetry paragraph** (line 158): add that in the market's own volatilities the $\delta = -2$ put is 1.4 standard deviations from the forward and the $\delta = +2$ call 2.7, that wing prices fall like a Gaussian tail in that number, and link Appendix A.6 (item 19).
- **§1.7**: after "Marking a book", say what the marks are used for: P&L, risk, collateral, fair-value accounting and its three levels, independent price verification (item 20). A short "What the models leave out" paragraph (item 17), unless it goes in L2 §2.5.

### 2.3 Lecture 5 (`05-implied-volatility-and-the-smile.md`)

- **§5.3** (line 57): write $s_{\text{iv}}(\delta) = 0.3 + 0.35\,\lvert\delta\rvert$ explicitly, and give the evidence for 0.3 and 1.0: SPY's median half-spreads in vol points at the money and at $\lvert\delta\rvert = 2$ on 21 October 2026, from study S8 (item 16). *Open:* these numbers must be pulled from the S8 surface; the lecture now says only "as the bid-ask spreads show".
- **§5.9**: one sentence that a strip over quoted strikes is truncated and biased low, and that bids and asks give a band (item 18). Link to Appendix A.4.

### 2.4 Lecture 2 (`02-no-arbitrage-without-a-model.md`)

- **§2.4, "A limit: American exercise"** (line 83): add the decomposition $C_{\text{Am}} - P_{\text{Am}} = D(F - K) + e_C - e_P$ and the two consequences, $\hat D > D$ and $\hat F < F$ (item 14). Say the premium is computed with a binomial tree, linking Appendix A.2 (item 11).
- **§2.6, bounds**: optionally add the spot-based bounds $C \ge S - Ke^{-rT}$ and $P \ge Ke^{-rT} - S$, which are what the early-exercise argument uses. Otherwise they live in Appendix A.2.

### 2.5 Lecture 4 (`04-black-scholes.md`)

- **§4.8, interpretation**: link Appendix A.6.3 for why the Black–Scholes put is cheaper although it is more likely to pay (item 19).

### 2.6 Lecture 7 (`07-calibration-as-estimation.md`)

- **§7.1** (line 21): name the vega-at-20% choice as a fifth assumption, or as a remark under assumption 2, with the factor of up to 6× from Fig. 7.3 (item 16).

## 3. New appendix: [`appendix-a-reading-a-real-chain.md`](appendix-a-reading-a-real-chain.md)

One appendix for the course, rather than one per lecture, because the material is shared by Lectures 1, 2, 4, 5 and 16. **Drafted** on 29 October 2026, with every derivation written out; the lectures do not link to it yet.

- **A.0 Notation**, and the replication argument every bound uses.
- **A.1 Anatomy of a chain.** Expiries, strikes, what a quote contains, liquid against thin, where the quotes come from.
- **A.2 Early exercise.** The replication bounds with and without dividends; the proof that a call on a non-dividend asset is never exercised early, by selling and by a short-share strategy; when a dividend makes call exercise possible, with EWI's numbers; why a deep in-the-money put is exercised; the bound $e_P \le K(1 - D)\,\mathbb{Q}(\text{early exercise})$ and why out-of-the-money premia are small, with a table; the binomial tree that computes the premium.
- **A.3 Parity with American options.** The decomposition; the least-squares bias $\hat D = D + \beta > D$; the forward bias $\hat F_i = F - e_P(K_i)/D$; the effect on implied volatilities through $\partial P/\partial F$ and $\partial C/\partial F$.
- **A.4 The variance-swap strip on a thin chain.** The change of variable from $k$ to $K$; the discrete strip; EWI's 29-day table; seven approximations, with the derivation of the jump term $J^3/3 + J^4/12 + \dots$.
- **A.5 SPY and SPX.** The conventions side by side, and why the course uses SPY.
- **A.6 Why out-of-the-money puts cost more than calls** (item 19). The symmetry $p(k) = e^k c(-k)$, proved line by line; why Black–Scholes puts are cheaper; the market's decomposition; the elasticity and the tail asymptotic $c \approx e^{-n^2/2}\times(\dots)$; prices as areas under the distribution function; the economic reasons.

## 4. Order of work

1. Figure 1.7 and its caption (items 1, 2, 3, 7): the figure is the source of most of the questions.
2. Link the lectures to the drafted appendix, and review it.
3. Lecture 1 body edits (items 4–6, 8–10, 12, 19, 20).
4. Lecture 2 §2.4 derivation (item 14).
5. Lecture 5 §5.3 noise evidence (item 16; needs the S8 half-spreads) and Lecture 7 §7.1.
6. The "What the models leave out" paragraph (item 17).
