# Appendix A: Reading a real chain

[Course index](README.md) · Draft of 29 October 2026, from the [clarifications plan](clarifications-plan.md).

**What this appendix is for.** Lectures 1, 2, 5 and 16 read real option chains: SPY, a liquid one, and EWI and EWG, two thin ones. Reading them raises questions the lectures answer in a sentence: what a chain contains, why American exercise matters for some quotes and not for others, how it biases put–call parity, what the variance-swap strip of a thin chain does and does not measure, why the course uses SPY rather than SPX, and why out-of-the-money puts cost so much more than out-of-the-money calls. This appendix gives the full arguments and derivations, step by step. Nothing in it changes a result of the course.

**What it assumes.** Lecture 1 (options, moneyness), Lecture 2 (replication, the forward, put–call parity, forward units, the pricing measure $\mathbb{Q}$) and Lecture 4 (the Black–Scholes formula in forward units). Section A.4 also uses Lecture 5, Section 5.9.

## A.0 Notation

| Symbol | Meaning |
| --- | --- |
| $t$, $T$ | Today and the option's maturity; $\tau = T - t$ is the time left |
| $S_t$ | Spot price of the underlying at $t$ |
| $K$ | Strike |
| $r$ | Continuously compounded interest rate, taken constant here |
| $D(t, T) = e^{-r(T - t)}$ | Discount factor: the price at $t$ of one unit paid at $T$. Written $D$ when the dates are clear. With $r > 0$, $D < 1$ |
| $F$ | Forward price for delivery at $T$ (Lecture 2, Section 2.3) |
| $C$, $P$ | Call and put prices in currency. Subscripts Eu and Am mark European and American exercise |
| $k = \ln(K/F)$ | Log-moneyness. Out of the money means $k < 0$ for a put and $k > 0$ for a call |
| $c$, $p$ | Prices in forward units, $c = C/(DF)$, $p = P/(DF)$ (Lecture 2, Section 2.5) |
| $X_T = S_T/F$ | Terminal price in forward units; $\mathbb{E}^{\mathbb{Q}}[X_T] = 1$ |
| $w = \sigma^2 T$ | Total variance of a Black–Scholes model with volatility $\sigma$ |
| $N$, $\varphi$ | Standard normal distribution function and density |
| $d_{1,2} = (-k \pm \tfrac12 w)/\sqrt w$ | The Black–Scholes arguments in forward units (Lecture 4, Section 4.7) |

A **replication bound** below is always proved the same way, by the law of one price of Lecture 2, Section 2.2. If portfolio A is worth at least as much as portfolio B at $T$ in every state of the world, then A costs at least as much as B today. Otherwise one would buy A, sell B, keep the difference today, and owe nothing at $T$: an arbitrage.

## A.1 Anatomy of a chain

An **option chain** is the list of all option contracts listed on one underlying. It is organised in three layers:

1. **Expiries.** The dates on which contracts mature. A large index ETF such as SPY lists many kinds: daily expiries for the next few weeks, weeklies, the standard monthly expiry (the third Friday of the month), end-of-month and quarterly expiries, and long-dated options called LEAPs, which reach two to three years. On 21 October 2026 SPY listed 35 expirations. A thin chain lists a few monthly and quarterly expiries: EWI listed four within a year, at 29, 57, 148 and 238 calendar days. When the third Friday is an exchange holiday, the expiry moves to the Thursday before it; EWI's June 2027 expiry is Thursday 17 June for that reason.
2. **Strikes.** For each expiry, a list of strikes, usually evenly spaced near the money and wider apart far from it. EWI's strikes are one dollar apart around a price of 55 dollars, about 1.8% of the price.
3. **Calls and puts.** At each strike there is one call and one put.

**What is quoted for each contract.** A **bid** (the highest price someone is willing to pay), an **ask** (the lowest price someone is willing to accept), the sizes behind them, the last trade, the day's **volume** (contracts traded that day) and the **open interest** (contracts outstanding). A missing bid is recorded as zero: nobody is willing to buy. Each contract is on 100 shares, the **multiplier**; quoted prices are per share.

**Liquid against thin.** SPY's file of 21 October 2026 holds 13,114 options; 32 expirations keep at least 8 out-of-the-money quotes after cleaning (Lecture 1, Section 1.6). EWI's file of 22 October holds 188 options; 94 are out of the money, 45 of them two-sided (with a bid as well as an ask), and the whole chain traded 132 contracts that day. The width of a quote, measured in implied volatility, is the most useful single summary: at EWI's four expiries the median width of the two-sided quotes within 0.1 of the money is 36, 23, 24 and 20 volatility points, against SPY's design noise of $\pm 0.3$ points at the money (Lecture 5, Section 5.3).

**Where the quotes come from.** The course uses Cboe's delayed quotes at the close. A data terminal shows the same bid and ask; nothing in a quote is computed by the course. Implied volatilities, mids and everything else are derived from these two numbers.

## A.2 Early exercise of American options

An **American** option can be exercised at any time up to $T$; a **European** option only at $T$. SPY, EWI and EWG options are American. The question of this section is when the right to exercise early is worth something, and how much.

**Step 0: an American option is worth at least the European one.** The holder of an American option can always choose to wait until $T$, and then holds exactly the European option. A right that can be ignored cannot have negative value, so $C_{\text{Am}} \ge C_{\text{Eu}}$ and $P_{\text{Am}} \ge P_{\text{Eu}}$. The difference is the **early-exercise premium**,

$$
e_C = C_{\text{Am}} - C_{\text{Eu}} \ge 0, \qquad e_P = P_{\text{Am}} - P_{\text{Eu}} \ge 0 .
$$

### A.2.1 Two replication bounds

Take first an underlying that pays no dividend before $T$.

**Call.** Portfolio A is one European call and $K$ zero-coupon bonds paying 1 at $T$, which cost $KD$ today. Portfolio B is one share. At $T$:

$$
A_T = (S_T - K)^+ + K = \max(S_T, K) \ \ge\ S_T = B_T .
$$

A is worth at least B in every state, so it costs at least as much today: $C_{\text{Eu}} + KD \ge S_t$, that is

$$
C_{\text{Eu}} \ \ge\ S_t - K D .
\tag{A.1}
$$

**Put.** Portfolio A is one European put and one share. Portfolio B is $K$ zero-coupon bonds. At $T$:

$$
A_T = (K - S_T)^+ + S_T = \max(K, S_T) \ \ge\ K = B_T ,
$$

so $P_{\text{Eu}} + S_t \ge KD$, that is

$$
P_{\text{Eu}} \ \ge\ K D - S_t .
\tag{A.2}
$$

**With dividends.** Let the share pay cash dividends $d_j$ at dates $t_j$ in $(t, T]$, with present value $\mathrm{PV} = \sum_j d_j\, D(t, t_j)$. The holder of the share receives them; reinvest each in bonds until $T$, where it is worth $d_j / D(t_j, T)$. Add to the bond side of each comparison bonds that pay the same amounts at $T$; they cost $\mathrm{PV}$ today. Repeating the two arguments with these extra cash flows on both sides gives

$$
C_{\text{Eu}} \ \ge\ S_t - \mathrm{PV} - KD, \qquad P_{\text{Eu}} \ \ge\ KD + \mathrm{PV} - S_t .
\tag{A.3}
$$

The dividends lower the call's bound and raise the put's, because the call holder does not receive them and the share price falls when they are paid.

### A.2.2 A call on a non-dividend asset is never exercised early

At any time $t$ before $T$, exercising the call pays $S_t - K$. By Step 0 and (A.1), keeping the call is worth

$$
C_{\text{Am}} \ \ge\ C_{\text{Eu}} \ \ge\ S_t - KD \ >\ S_t - K ,
$$

where the last inequality holds because $D < 1$ when $r > 0$. The call alive is worth strictly more than the call exercised. A holder who wants to leave the position should sell the call, not exercise it.

**If the call cannot be sold.** The conclusion does not depend on a buyer being available. Compare two strategies at $t$, for a call in the money ($S_t > K$):

- *Exercise.* Pay $K$, receive the share, sell it, and invest the proceeds $S_t - K$ in bonds. At $T$ this is worth $(S_t - K)/D$.
- *Keep the call and sell one share short.* Invest the short-sale proceeds $S_t$ in bonds, worth $S_t/D$ at $T$. At $T$ the share must be returned. If $S_T > K$, exercise the call and return the share received, paying $K$; if $S_T \le K$, buy the share in the market for $S_T$. Either way the cost at $T$ is $\min(S_T, K)$, and the strategy is worth $S_t/D - \min(S_T, K)$.

The second minus the first is

$$
\frac{S_t}{D} - \min(S_T, K) - \frac{S_t - K}{D} \;=\; \frac{K}{D} - \min(S_T, K) \;\ge\; \frac{K}{D} - K \;>\; 0 ,
$$

in every state. Keeping the call beats exercising it for two reasons, visible in the two terms. The strike is paid later, so it earns interest ($K/D$ against $K$); and if the price ends below the strike, the holder who waited pays only $S_T$ instead of $K$, which is the insurance an exercised call gives up.

**Consequence.** If early exercise is never optimal, the American call is worth what it pays when held to $T$: $C_{\text{Am}} = C_{\text{Eu}}$ and $e_C = 0$.

**Why the put differs.** The two payoffs look symmetric, $S_t - K$ and $K - S_t$, but the cash flows are not. Exercising a call means *paying* $K$; exercising a put means *receiving* $K$. With a positive rate one prefers to pay late and receive early. The interest argument above therefore always favours waiting with a call, and can favour exercising a put (Section A.2.4).

### A.2.3 Dividends: when a call may be exercised early

Let a single dividend $d$ be paid at $t_d$, between $t$ and $T$. On the ex-dividend date the share price drops by $d$: before it, the share carries the right to the dividend; after it, it does not. (Taxes and other frictions are ignored.)

**Claim 1: if a call is exercised early, it is just before $t_d$.** At any time $s$ before $t_d$, the holder can wait until just before $t_d$, a moment written $t_d^-$, and exercise then. No dividend is paid between $s$ and $t_d^-$, so Section A.2.2 applies with $t_d^-$ in place of $T$: waiting is worth at least $S_s - K\,D(s, t_d) > S_s - K$. Exercising at $s$ is dominated. After $t_d$ there is no further dividend, and Section A.2.2 applies up to $T$. The only candidate is $t_d^-$.

**Claim 2: a necessary condition.** At $t_d^-$ the share is worth $S$ (with the dividend). Exercising pays $S - K$. Not exercising, the holder keeps a call on a share that drops to $S - d$ an instant later and pays no further dividend, so by (A.1) the call is then worth at least $S - d - K\,D(t_d, T)$. Waiting is better whenever

$$
S - d - K D(t_d, T) \ >\ S - K
\quad\Longleftrightarrow\quad
d \ <\ K\,\big(1 - D(t_d, T)\big) .
$$

Early exercise can therefore pay only if the dividend exceeds the interest the strike would earn between the ex-date and maturity, $K(1 - D(t_d, T)) \approx K r (T - t_d)$. The condition is necessary, not sufficient: exercising also gives up the call's insurance, so it pays only deep in the money.

**On EWI.** The course assumes a dividend of 0.70 dollars on 15 December 2026, three days before the December expiry. For the call struck at 57, $K(1 - D) = 57\,(1 - e^{-0.0422 \times 3/365}) = 0.020$ dollars, far below 0.70. The condition holds, and the binomial tree of Section A.2.6 gives this call an early-exercise premium of 0.18 dollars, more than its half-spread of 0.125 (Lecture 2, Section 2.4). EWG pays no dividend before June 2027, so its calls carry none.

**Dividends and puts.** A put holder who exercises before the ex-date gives up the price drop the dividend causes, which would raise the put's payoff. Dividends therefore make early exercise of a put *less* attractive.

### A.2.4 Puts: why early exercise can pay

Exercising the put at $t$ pays $K - S_t$. The bound (A.2), $P_{\text{Eu}} \ge KD - S_t$, lies below this by $K(1 - D)$, so unlike the call's, it does not show that the European put is worth more than immediate exercise. In fact, deep in the money, it is worth less.

**Proof.** The put's payoff is at most $K$, so $K$ bonds dominate it and $P_{\text{Eu}} \le KD$. If $S_t < K(1 - D)$, then

$$
K - S_t \ >\ K - K(1 - D) = KD \ \ge\ P_{\text{Eu}} .
$$

The American put is worth at least its exercise value, so $P_{\text{Am}} \ge K - S_t > P_{\text{Eu}}$, and the early-exercise premium is strictly positive.

**Size.** The bound $S_t < K(1 - D)$ is crude. Under Black–Scholes with $K = 100$, $T = 1$, $\sigma = 20\%$ and $r = 4.22\%$, it gives $S_t < 4.13$, but the European put is already below its intrinsic value for every $S_t$ below 88.5.

**Interpretation.** Exercising the put brings forward the receipt of $K$, which is worth the interest $K(1 - D)$. It gives up the insurance: the chance that the price rises back above the strike, in which case waiting would have cost less. Deep in the money that chance is small, the interest dominates, and the put is exercised. Near the money the insurance dominates, and it is not. So the intuition "an in-the-money option should always be exercised, to invest the money" is right for a put deep in the money, wrong for a put near the money, and wrong for every call on a non-dividend asset.

There is a critical price $S^*(t) < K$, the **exercise boundary**, below which the American put is exercised. It has no closed form and is computed numerically, as in Section A.2.6.

### A.2.5 Why the premium is small for out-of-the-money options

**A bound on the premium.** Let $\tau^*$ be the time at which the American put is optimally exercised, with $\tau^* = T$ if it is never exercised early. Two facts:

1. The American price is the discounted expected payoff under the optimal policy: $P_{\text{Am}} = \mathbb{E}^{\mathbb{Q}}\big[D(t, \tau^*)\,(K - S_{\tau^*})^+\big]$.
2. The discounted European price is a $\mathbb{Q}$-martingale (Lecture 2, Section 2.5: every price is a discounted expectation), and a martingale stopped at a bounded random time keeps its expectation (optional stopping). So $P_{\text{Eu}} = \mathbb{E}^{\mathbb{Q}}\big[D(t, \tau^*)\,P_{\text{Eu}}(\tau^*)\big]$, where $P_{\text{Eu}}(\tau^*)$ is the European put's price at $\tau^*$.

Subtracting,

$$
e_P = \mathbb{E}^{\mathbb{Q}}\Big[D(t, \tau^*)\,\big(K - S_{\tau^*} - P_{\text{Eu}}(\tau^*)\big)\,\mathbf{1}\lbrace\tau^* < T\rbrace\Big] ,
$$

because at $\tau^* = T$ the two payoffs coincide. By (A.2) at time $\tau^*$, $P_{\text{Eu}}(\tau^*) \ge K D(\tau^*, T) - S_{\tau^*}$, so the bracket is at most $K\,(1 - D(\tau^*, T)) \le K\,(1 - D(t, T))$. With $D(t, \tau^*) \le 1$:

$$
e_P \ \le\ K\,\big(1 - D(t, T)\big)\ \mathbb{Q}\big(\tau^* < T\big) .
\tag{A.4}
$$

**Reading the bound.** The premium is at most the interest on the strike over the option's life, times the probability that the put is exercised early. An out-of-the-money put ($K < F$) is exercised early only if the price first falls below the strike and then further below the boundary $S^*(t) < K$ before maturity: a large fall, with a small probability. Its premium is therefore small. An option that is out of the money also has no intrinsic value today, so exercising it now would pay nothing at all.

**Numbers.** Puts with $S = 100$, $T = 0.5$, $\sigma = 20\%$, $r = 4.22\%$, priced on a 2,000-step tree (Section A.2.6):

| Strike | $k = \ln(K/F)$ | European | American | Premium | Premium / European |
| --- | --- | --- | --- | --- | --- |
| 80 | −0.244 | 0.2138 | 0.2182 | 0.0044 | 2.1% |
| 90 | −0.126 | 1.3457 | 1.3853 | 0.0395 | 2.9% |
| 100 | −0.021 | 4.5956 | 4.7887 | 0.1931 | 4.2% |
| 110 | +0.074 | 10.4918 | 11.1175 | 0.6258 | 6.0% |
| 120 | +0.161 | 18.4640 | 20.0024 | 1.5383 | 8.3% |

The premium rises with the strike, in dollars and as a share of the price. The course keeps only out-of-the-money quotes (the first two rows and their call counterparts), where it is smallest. On the real chains, the premium is 0.0036 of a half-spread for EWI's 29-day put at $k = -0.10$, and at most 0.31 of a half-spread for EWG's puts (Lecture 16, Section 16.2).

### A.2.6 How the premium is computed: the binomial tree

The premium is not quoted. The screen shows the American option's bid and ask; the premium is a model number, the American price minus the European price under the same model. The course computes it with a **Cox–Ross–Rubinstein (CRR) binomial tree** (`lab/checks/thin_chains/thin_chain.py`, method `tree`).

1. **Time steps.** Split $[t, T]$ into $n$ steps of length $\Delta t = \tau/n$ ($n = 600$ in the code).
2. **Moves.** In each step the price moves up by the factor $u = e^{\sigma\sqrt{\Delta t}}$ or down by $1/u$. The log-price then moves by $\pm\sigma\sqrt{\Delta t}$, so its variance per step is $\sigma^2\Delta t$, as in Black–Scholes.
3. **Pricing probability.** Under $\mathbb{Q}$ the expected price after one step must equal the price carried at the rate $r$: $p\,u + (1 - p)/u = e^{r\Delta t}$, which gives $p = (e^{r\Delta t} - 1/u)/(u - 1/u)$.
4. **At maturity.** At each terminal node the option is worth its payoff, $(K - S)^+$ for a put.
5. **Backwards.** At each earlier node the value of waiting is the discounted expectation of the two next values, $e^{-r\Delta t}\,[p\,V_{\text{up}} + (1 - p)\,V_{\text{down}}]$. For the European option this is the node's value. For the American option the node's value is the larger of waiting and exercising now: $\max\big(\text{waiting},\ K - S\big)$.
6. **Dividends.** With cash dividends, the tree is built on the share price minus the present value of the dividends still to be paid before $T$, the part of the price that moves randomly ("escrowed" dividends). At each node, the exercise value uses the full price, the tree value plus the present value of the dividends still to come.
7. **The premium.** Run the tree twice, with and without step 5's maximum, at the quote's own implied volatility (25% when the quote has no bid), and subtract. Both runs use the same tree, so most of its discretization error cancels. The code reports the premium divided by the quote's **half-spread**, $(\text{ask} - \text{bid})/2$, the natural scale of the quote's uncertainty.

## A.3 Put–call parity with American options

### A.3.1 The decomposition

European parity, $C_{\text{Eu}} - P_{\text{Eu}} = D(F - K)$, was proved in Lecture 2, Section 2.4: long one call and short one put pay $S_T - K$ at $T$ in every state, which is worth $D(F - K)$ today. Writing each American price as its European price plus its premium, $C_{\text{Am}} = C_{\text{Eu}} + e_C$ and $P_{\text{Am}} = P_{\text{Eu}} + e_P$, and subtracting:

$$
C_{\text{Am}} - P_{\text{Am}} \;=\; D\,(F - K) \;+\; e_C(K) \;-\; e_P(K) .
\tag{A.5}
$$

Parity on American quotes is European parity plus an error term. With no dividend before $T$, $e_C = 0$ (Section A.2.2), and the error is $-e_P(K)$, which grows with the strike (Section A.2.5).

### A.3.2 The regression for the discount factor is biased upwards

**The method.** Take $m$ strikes $K_1, \dots, K_m$ that have both a call and a put, and set $y_i = C_i - P_i$. Under European parity, $y_i = DF - D K_i$: a straight line in $K$ with intercept $DF$ and slope $-D$. Fitting $y = a + bK$ by ordinary least squares gives $\hat D = -\hat b$ and $\hat F = \hat a/\hat D$.

**The bias.** The least-squares slope is

$$
\hat b = \frac{\sum_i (K_i - \bar K)(y_i - \bar y)}{\sum_i (K_i - \bar K)^2} .
$$

By (A.5) with $e_C = 0$, $y_i = DF - DK_i - e_{P,i}$, where $e_{P,i} = e_P(K_i)$. Substituting, and using $\sum_i (K_i - \bar K) = 0$ to drop the constant $DF$:

$$
\hat b \;=\; -D \;-\; \frac{\sum_i (K_i - \bar K)(e_{P,i} - \bar e_P)}{\sum_i (K_i - \bar K)^2} \;=\; -D - \beta ,
$$

where $\beta$ is the least-squares slope of the premia against the strikes. Since $e_P$ increases with $K$, $\beta > 0$. To see it, let $K_c$ be a cut such that $e_{P,i} \ge \bar e_P$ exactly when $K_i \ge K_c$; it exists because $e_P$ is increasing. Because $\sum_i (e_{P,i} - \bar e_P) = 0$, the constant $\bar K$ in the numerator can be replaced by $K_c$: $\sum_i (K_i - \bar K)(e_{P,i} - \bar e_P) = \sum_i (K_i - K_c)(e_{P,i} - \bar e_P)$. In the second sum each term is a product of two numbers of the same sign, so it is non-negative, and the sum is positive unless all premia are equal. Hence

$$
\hat D = D + \beta \ >\ D .
$$

**On SPY.** The regression returns a discount factor above 1, which is impossible with a positive rate, for 30 of the 32 expirations, up to 1.044 at 562 days (Lecture 2, Section 2.4 and Figure 2.1). The course therefore takes $D$ from Treasury yields.

### A.3.3 The forward from parity is biased downwards

With $D$ known, each pair gives a forward, $\hat F_i = K_i + (C_i - P_i)/D$. By (A.5) with $e_C = 0$,

$$
\hat F_i = K_i + \frac{D(F - K_i) - e_{P,i}}{D} = F - \frac{e_P(K_i)}{D} \ <\ F .
$$

Every pair gives a forward that is too low, by its put's premium. The course takes the median over the six pairs nearest the money, where the premia are smallest. On EWG this forward is still 0.15% to 0.46% below the carry forward $S_0 e^{rT}$, so Lecture 16 uses the carry forward.

### A.3.4 What a low forward does to implied volatilities

Implied volatility inverts the Black–Scholes price at a given forward. In currency, $P = D\,(K N(-d_2) - F N(-d_1))$ and $C = D\,(F N(d_1) - K N(d_2))$. Differentiating with respect to $F$, and using $F\varphi(d_1) = K\varphi(d_2)$ (Lecture 4, Section 4.10) to cancel the terms from $\partial d_{1,2}/\partial F$:

$$
\frac{\partial P}{\partial F} = -D\,N(-d_1) < 0, \qquad \frac{\partial C}{\partial F} = D\,N(d_1) > 0 .
$$

Suppose the forward used is too low. At a fixed volatility, the model put price is then too high, so to match the observed put price the implied volatility must be lower: vega is positive (Lecture 4, Section 4.10). The model call price is too low, so the call's implied volatility must be higher. A forward that is too low moves put volatilities down and call volatilities up. On EWG the shift reaches 1.24 half-spreads (Lecture 16, Section 16.2).

## A.4 The variance-swap strip on a thin chain

### A.4.1 From forward units to strikes

Lecture 5, Section 5.9 shows that, for any model with continuous paths, the fair variance-swap strike $K_{\text{var}}(T)$ satisfies

$$
K_{\text{var}}(T)\,T = w_{\text{var}}(T) = 2\int_{-\infty}^{0} p(k)\,e^{-k}\,dk + 2\int_{0}^{\infty} c(k)\,e^{-k}\,dk ,
$$

with prices in forward units. Write $O(K)$ for the out-of-the-money price in currency at strike $K$: the put for $K \le F$, the call for $K > F$. Change variable from $k$ to $K = F e^k$:

- $dk = dK/K$,
- $e^{-k} = F/K$,
- the forward-unit price is $O(K)/(DF)$.

Each integrand becomes

$$
\frac{O(K)}{DF}\cdot\frac{F}{K}\cdot\frac{dK}{K} = \frac{O(K)}{D\,K^2}\,dK ,
$$

and the two integrals join into one over all strikes:

$$
K_{\text{var}}(T) = \frac{2}{T\,D}\int_0^\infty \frac{O(K)}{K^2}\,dK .
$$

Each option enters with weight $1/K^2$: low strikes weigh more. The **variance-swap volatility** is $\sigma_{\text{var}} = \sqrt{K_{\text{var}}}$.

### A.4.2 The discrete strip on bids and on asks

A chain quotes finitely many strikes $K_1 < \dots < K_m$. The code replaces the integral by a sum over them (`thin_chain.py`, function `strip_vols`):

$$
K_{\text{var}} \approx \frac{2}{T\,D}\sum_{i=1}^{m} \Delta K_i\,\frac{O_i}{K_i^2},
\qquad
\Delta K_i = \tfrac12\,(K_{i+1} - K_{i-1}) ,
$$

with one-sided differences at the two ends, $\Delta K_1 = K_2 - K_1$ and $\Delta K_m = K_m - K_{m-1}$. It is evaluated twice: once with every $O_i$ set to its bid (zero when there is none), once with every $O_i$ set to its ask. Every weight is positive, so the bid strip is below the ask strip.

**Meaning.** The sum is the price of a portfolio: $2\Delta K_i/(TDK_i^2)$ units of each option. Buying that portfolio at the quoted prices costs the ask strip; selling it raises the bid strip. The two numbers bound what a trader could pay or receive for the strip on that day. They are not a confidence interval for anything.

### A.4.3 Worked example: EWI, 20 November 2026

On 22 October 2026, EWI's first expiry is 29 calendar days away: $T = 29/365 = 0.0795$, $F = 55.205$ dollars, $D = 0.99665$. All strikes are one dollar apart, so every $\Delta K_i = 1$. The contribution of strike $i$ is $2\,\Delta K_i\,O_i/(T D K_i^2)$.

| Type | Strike | $k$ | Bid | Ask | Contribution on bids | Contribution on asks |
| --- | --- | --- | --- | --- | --- | --- |
| Put | 50 | −0.099 | 0.20 | 1.85 | 0.00202 | 0.01869 |
| Put | 51 | −0.079 | 0.05 | 1.70 | 0.00049 | 0.01651 |
| Put | 52 | −0.060 | 0.40 | 0.65 | 0.00374 | 0.00607 |
| Put | 53 | −0.041 | 0.10 | 2.20 | 0.00090 | 0.01978 |
| Put | 54 | −0.022 | 0.25 | 2.35 | 0.00217 | 0.02035 |
| Put | 55 | −0.004 | 1.00 | 1.60 | 0.00835 | 0.01336 |
| Call | 56 | +0.014 | 0.30 | 1.60 | 0.00242 | 0.01289 |
| Calls, ask only | 57 to 71 | +0.032 to +0.252 | none | 1.00 to 2.15 | 0 | 0.16163 |
| **Sum** | | | | | **0.0201** | **0.2693** |
| $\sigma_{\text{var}} = \sqrt{\text{sum}}$ | | | | | **14.2%** | **51.9%** |

**What the table shows.**

1. **The band is wide.** 14.2% to 51.9%. The implied volatilities of the seven two-sided mids range from 21% to 50% (Figure 1.7), and the strip is barely constrained.
2. **The ask strip is mostly placeholder offers.** The fifteen calls from 57 to 71 have no bid and asks between 1.00 and 2.15 dollars. The call struck at 71, 29% above the forward with a month to go, is offered at 1.75 dollars; at 40% volatility it is worth 3 cents, and even at 60% about 32 cents. These standing offers supply 0.162 of the ask strip's 0.269, 60% of it.
3. **The bid strip is missing a wing.** Every call above 56 contributes zero on the bid side, and nothing below 50 is quoted.

### A.4.4 The approximations

1. **Bid and ask.** The band says what the strip costs to trade, not where its fair value lies (Section A.4.2).
2. **Truncation.** The integrand $O(K)/K^2$ is positive, so every strike left out removes a positive amount. A strip over the quoted strikes alone is biased low, on bids and on asks. On SPY at one year the puts beyond the design grid carry about a quarter of the variance (Lecture 5, Section 5.9).
3. **Discretization.** The sum is a midpoint rule; for a smooth integrand its error shrinks as $\Delta K^2$. With strikes 1.8% of the forward apart, it is small against the band. The integrand has a kink at $K = F$, where the put is replaced by the call, which adds an error of the same order at that one strike.
4. **American prices treated as European.** The early-exercise premia of out-of-the-money quotes are small (Section A.2.5): 0.0036 of a half-spread for the put at 50.
5. **Continuous paths.** The formula of Section A.4.1 assumes the price has no jumps. With jumps, the strip and the variance swap differ. To see by how much, cut $[0, T]$ into small steps with returns $R_i = X_{i+1}/X_i - 1$, and use $\ln(1 + R) = R - \big(R - \ln(1 + R)\big)$:

   $$
   \ln X_T = \sum_i R_i - \sum_i \big(R_i - \ln(1 + R_i)\big) .
   $$

   For a small diffusive step, $R - \ln(1 + R) \approx \tfrac12 R^2$, and these terms add up to $\tfrac12\int_0^T V_t\,dt$ (Lecture 5, Section 5.9). For a jump of size $J$ in the log-price, $R = e^J - 1$ and $R - \ln(1 + R) = e^J - 1 - J$ exactly. The sum $\sum_i R_i$ has mean zero under $\mathbb{Q}$, because $X$ is a martingale. Hence

   $$
   \mathbb{E}^{\mathbb{Q}}\big[-2\ln X_T\big] = \mathbb{E}^{\mathbb{Q}}\Big[\int_0^T V_t\,dt\Big] + \mathbb{E}^{\mathbb{Q}}\Big[\sum_{\text{jumps}} 2\,(e^J - 1 - J)\Big] .
   $$

   The left side is what the strip prices. The variance swap pays the realized variance, the sum of squared log-returns, which is $\int_0^T V_t\,dt + \sum_{\text{jumps}} J^2$. The difference is

   $$
   \text{strip} - \text{variance swap} = \mathbb{E}^{\mathbb{Q}}\Big[\sum_{\text{jumps}} \big(2(e^J - 1 - J) - J^2\big)\Big],
   \qquad 2(e^J - 1 - J) - J^2 = \frac{J^3}{3} + \frac{J^4}{12} + \dots
   $$

   For a downward jump ($J < 0$) the leading term is negative: with crashes, the strip understates the fair variance-swap strike. For a 10% fall, $J^3/3 = -3.3 \times 10^{-4}$ against $J^4/12 = 8 \times 10^{-6}$.
6. **Discrete monitoring and the clock.** A variance swap pays $\frac{252}{n}\sum r_i^2$ over $n$ daily closing returns. The strip measures the continuous total variance divided by the model's $T$. The two differ by the day count (here $T$ is calendar days over 365; on a trading-day clock $T$ changes and $\sigma_{\text{var}}$ scales by $\sqrt{T_{\text{old}}/T_{\text{new}}}$) and by sampling once a day instead of continuously.
7. **The forward and the discount factor.** $F$ decides at which strike the strip switches from puts to calls, and $D$ converts prices to forward units. On a thin chain both come from outside the chain (Section A.3).

This is why Lecture 6, step 3, reads the forward-variance curve from the strip only at maturities whose strikes support it, and why Lecture 16's thin network infers the curve instead.

## A.5 SPY and SPX

The S&P 500 has two main option chains: options on SPY, an exchange-traded fund that holds the index, and options on the index itself, SPX.

| | SPY options | SPX options |
| --- | --- | --- |
| Exercise | American | European |
| Settlement | Physical: 100 shares of SPY change hands | Cash |
| Settlement price | The close of the last trading day | Standard monthlies (SPX): an opening price on expiry morning, "AM-settled". Weeklies and dailies (SPXW): the close, "PM-settled" |
| Size | SPY trades near a tenth of the index | Ten times larger per contract |
| Dividends | SPY pays quarterly cash dividends; calls can carry early-exercise value before an ex-date | The index pays nothing to the option holder, but its forward still falls by the expected dividends |

**What SPX would remove.** With European options, parity holds exactly: the regression of Section A.3.2 gives an unbiased discount factor and forward, and no tree check is needed.

**What SPX would add.** A choice between AM- and PM-settled expiries, kept consistent across the surface; for AM-settled expiries the option's life ends at the open, not the close; the forward must carry the index's expected dividends, read from parity; and a separate data source.

**Why the course uses SPY.** The SPY chain is the data in hand (v2 plan, decisions table). The stronger reason is that the course's thin chains, EWG and EWI, are American ETF options with no European counterpart, so the pipeline must handle American exercise in any case: the carry forward, Treasury discount factors, and a check that early-exercise premia are small against the spreads. Using SPY runs the same pipeline on the liquid and the thin chain. A second liquid smile on SPX is a planned extension (v2 plan, option E).

## A.6 Why out-of-the-money puts cost more than out-of-the-money calls

Lecture 1, Figure 1.6 shows SPY's out-of-the-money prices on the design grid, in forward units, on a log scale. At every maturity the left side (puts) is far above the right side (calls). This section explains the asymmetry in two layers. The first, mechanical, is present even in Black–Scholes, and makes puts slightly *cheaper*. The second, the market's skew, reverses it by a large factor.

### A.6.1 The set-up

Take the one-year points at $\delta = \pm 2$. With $\delta = k/(\sigma_{\text{ref}}\sqrt T)$, $\sigma_{\text{ref}} = 20\%$ and $T = 1$, they sit at $k = \pm 2 \times 0.2 \times 1 = \pm 0.4$: a put struck at $e^{-0.4}F = 0.67F$ and a call struck at $e^{0.4}F = 1.49F$. Both strikes are the same distance from the forward in log terms. In SPY's data the put costs 0.0086 of the forward and the call 0.00019, a ratio of about 45 (Lecture 1, Section 1.6).

### A.6.2 Put–call symmetry in Black–Scholes

**The put formula.** By parity in forward units, $p = c - 1 + e^k$ (Lecture 2, Section 2.5). Substituting $c = N(d_1) - e^k N(d_2)$ and using $1 - N(x) = N(-x)$:

$$
p(k) = N(d_1) - e^k N(d_2) - 1 + e^k = e^k\big(1 - N(d_2)\big) - \big(1 - N(d_1)\big) = e^k N(-d_2) - N(-d_1) .
$$

**The arguments under $k \to -k$.** From $d_1(k) = (-k + \tfrac12 w)/\sqrt w$ and $d_2(k) = (-k - \tfrac12 w)/\sqrt w$:

$$
d_1(-k) = \frac{k + \tfrac12 w}{\sqrt w} = -d_2(k), \qquad d_2(-k) = \frac{k - \tfrac12 w}{\sqrt w} = -d_1(k) .
$$

**The symmetry.** The call at $-k$ is therefore

$$
c(-k) = N\big(d_1(-k)\big) - e^{-k} N\big(d_2(-k)\big) = N\big(-d_2(k)\big) - e^{-k} N\big(-d_1(k)\big) .
$$

Multiplying by $e^k$:

$$
e^k\,c(-k) = e^k N(-d_2) - N(-d_1) = p(k) .
\tag{A.6}
$$

Lecture 4, Section 4.8 gives a second proof, by a change of measure, which shows where the symmetry comes from: under Black–Scholes, the law of the price in units of cash and the law of cash in units of the asset are the same log-normal.

### A.6.3 What the symmetry says: Black–Scholes puts are cheaper

At $k = -0.4$, (A.6) gives $p(-0.4) = e^{-0.4}\,c(0.4) = 0.670\,c(0.4)$. With $\sigma = 20\%$: $c(0.4) = 0.00207$ and $p(-0.4) = 0.00138$. In a symmetric model the put is *cheaper* than the call.

**It is not because the put is less likely to pay.** Under Black–Scholes, $\ln X_T \sim \mathcal{N}(-\tfrac12 w, w)$ (Lecture 4, Section 4.6), with $w = 0.04$ here. The put pays if $\ln X_T < -0.4$, the call if $\ln X_T > 0.4$:

$$
\mathbb{Q}(\ln X_T < -0.4) = N\Big(\frac{-0.4 + 0.02}{0.2}\Big) = N(-1.9) = 2.9\%, \qquad
\mathbb{Q}(\ln X_T > 0.4) = N\Big(\frac{-0.4 - 0.02}{0.2}\Big) = N(-2.1) = 1.8\% .
$$

The put is the *more* likely to pay, because the distribution of $\ln X_T$ is centred slightly below zero, at $-\tfrac12 w$.

**It is because the put pays less when it pays.** A log-move of size $m$ beyond the strike pays the put $e^{-0.4}(1 - e^{-m})$ and the call $e^{0.4}(e^m - 1)$. For $m = 0.1$, that is $0.67 \times 0.095 = 0.064$ for the put against $1.49 \times 0.105 = 0.157$ for the call. The put's payoff is on the scale of its strike, $0.67F$, and can never exceed it; the call's is on the scale of $1.49F$ and is unbounded. Formally, $(e^{-a} - X_T)^+ = e^{-a}\,(1 - e^{a}X_T)^+$: the factor $e^{-a}$ comes out of the payoff, and (A.6) says that what remains is worth exactly the call.

### A.6.4 The market's prices

The market does not use one volatility. At one year on 21 October 2026, SPY's implied volatility is 28.7% at $\delta = -2$ and 14.8% at $\delta = +2$ (Lecture 5, Section 5.5). Both points are still at $k = \pm 0.4$, because $\delta$ uses the fixed $\sigma_{\text{ref}} = 20\%$. Pricing each option at its own implied volatility:

| | Black–Scholes at 20% | Market implied volatility | Market price | Market / Black–Scholes |
| --- | --- | --- | --- | --- |
| Put, $k = -0.4$ | 0.00138 | 28.7% | 0.00867 | × 6.3 |
| Call, $k = +0.4$ | 0.00207 | 14.8% | 0.000190 | ÷ 10.9 |
| Put / call | 0.670 | | 45.8 | × 68 |

The market prices reproduce Figure 1.6. The factor of 68 between the market's ratio and Black–Scholes' splits into the put's volatility being raised (× 6.3) and the call's lowered (÷ 10.9).

### A.6.5 Why a few volatility points move wing prices so much

**The elasticity.** The relative change in a price per unit of volatility is $\partial \ln c/\partial\sigma = \mathrm{vega}/c = \sqrt T\varphi(d_1)/c$ (vega from Lecture 4, Section 4.10). At the money, $c \approx 0.4\,\sigma\sqrt T$ (Lecture 4, Section 4.9), so $\partial\ln c/\partial\sigma \approx 1/\sigma$: at 14.8%, one volatility point changes the price by about 7%. For the call at $k = 0.4$, $T = 1$ and $\sigma = 14.8\%$, the same formula gives 66 per unit of volatility, 0.66 per volatility point: one point nearly doubles the price. Far out of the money, prices are about ten times more sensitive to volatility, in relative terms, than at the money.

**Where the sensitivity comes from.** For a call far out of the money, $d_1$ and $d_2$ are large and negative. Write $x_1 = -d_1 > 0$ and $x_2 = -d_2 = x_1 + \sqrt w$. The normal tail satisfies, for large $x > 0$,

$$
N(-x) = \int_x^\infty \varphi(z)\,dz = \frac{\varphi(x)}{x} - \int_x^\infty \frac{\varphi(z)}{z^2}\,dz = \frac{\varphi(x)}{x}\Big(1 + O\big(x^{-2}\big)\Big),
$$

where the second equality integrates by parts with $\varphi'(z) = -z\varphi(z)$. Applying it to both terms of $c = N(-x_1) - e^k N(-x_2)$, and using $\varphi(d_1) = e^k\varphi(d_2)$ (Lecture 4, Section 4.10):

$$
c \approx \frac{\varphi(x_1)}{x_1} - e^k\frac{\varphi(x_2)}{x_2} = e^k\varphi(d_2)\Big(\frac{1}{x_1} - \frac{1}{x_2}\Big) = e^k\varphi(d_2)\,\frac{\sqrt w}{x_1 x_2} .
$$

Expanding $d_2^2 = (k/\sqrt w + \tfrac12\sqrt w)^2 = k^2/w + k + w/4$ gives $e^k\varphi(d_2) = (2\pi)^{-1/2}\exp\big(-k^2/(2w) + k/2 - w/8\big)$, so

$$
c \;\approx\; \frac{\sqrt w}{\sqrt{2\pi}\,x_1 x_2}\,\exp\Big(-\frac{n^2}{2} + \frac{k}{2} - \frac{w}{8}\Big),
\qquad n = \frac{k}{\sigma\sqrt T} .
$$

The dominant factor is $e^{-n^2/2}$, where $n$ is the distance from the forward in units of the option's *own* standard deviation $\sigma\sqrt T$. A change of volatility changes $n$, and the price moves as a Gaussian tail in $n$. By (A.6) the same holds for a put at $-k$.

**The two SPY options.** In the market's own volatilities:

- the put: $n = 0.4/0.287 = 1.39$ standard deviations from the forward;
- the call: $n = 0.4/0.148 = 2.70$ standard deviations.

The factors $e^{-n^2/2}$ are 0.38 and 0.026, a ratio of 15 before the slower terms. In general $n = \lvert\delta\rvert\,\sigma_{\text{ref}}/\sigma$: where the market's volatility is above 20%, the design's $\delta$ overstates the distance, and where it is below, it understates it. This is why, in Figure 1.6, the put side falls slowly away from the money and the call side falls fast.

**Accuracy.** The approximation is the first term of an expansion in $1/x$. At $n = 2.70$ it gives 0.00026 against the exact 0.00019; at $n = 1.39$ it is poor. It explains the mechanism; the numbers in the table come from the exact formula.

### A.6.6 What the prices say about the distribution

**A put is the area under the distribution function.** For $X_T \ge 0$ and a strike $x > 0$,

$$
(x - X_T)^+ = \int_0^x \mathbf{1}\lbrace X_T \le u\rbrace\,du .
$$

Check: if $X_T \ge x$ the indicator is zero for every $u < x$, and the integral is 0; if $X_T < x$ the indicator is one for $u$ from $X_T$ to $x$, and the integral is $x - X_T$. Take expectations and exchange expectation and integral (allowed because the integrand is non-negative):

$$
p(x) = \mathbb{E}^{\mathbb{Q}}\big[(x - X_T)^+\big] = \int_0^x \mathbb{Q}\big(X_T \le u\big)\,du .
$$

In the same way, $c(x) = \int_x^\infty \mathbb{Q}(X_T > u)\,du$. An out-of-the-money put price is the area under the distribution function to the left of its strike; an out-of-the-money call price is the area under the tail probability to the right of its strike.

**Consequences.** Differentiating once, $p'(x) = \mathbb{Q}(X_T \le x)$; twice, $p''(x) = q(x)$, the density (Breeden and Litzenberger 1978; Lecture 2, Section 2.7). Puts far above their Black–Scholes prices mean that the market's distribution puts far more probability on large falls than the log-normal does. Calls below their Black–Scholes prices mean less probability on large rises. On SPY at three months, the market puts 3.6% probability on a fall of about 16% or more, against 0.6% for the log-normal (Lecture 2, Figure 2.3).

### A.6.7 Why the market's distribution is skewed

The arithmetic above says *what* the market's prices imply. Three reasons are usually given for *why*:

1. **Demand for insurance.** Investors who hold the index buy puts below the current price to limit their losses (Lecture 1, Section 1.4). Dealers who sell those puts bear the crash risk and charge for it.
2. **The leverage effect.** Volatility tends to rise when prices fall (Lecture 3, Section 3.7; Black 1976). A fall is then followed by larger moves, which fattens the left tail of the distribution of $X_T$. In Heston and rough Bergomi this is the negative correlation $\rho$ between price and volatility, and the skew identifies the product $\rho\eta$ (Lecture 7, Section 7.7).
3. **Jumps.** Large falls happen suddenly and large rises rarely do. Downward jumps add probability to the left tail, most visibly at short maturities, where a diffusion has little time to move far.

Lecture 5 measures the asymmetry as the skew of the implied-volatility smile, and Parts III and IV build the models that reproduce it.

## References

- Breeden, D. T., and Litzenberger, R. H. (1978). Prices of state-contingent claims implicit in option prices. *Journal of Business* 51(4), 621–651.
- Black, F. (1976). Studies of stock price volatility changes. *Proceedings of the 1976 Meetings of the American Statistical Association, Business and Economic Statistics Section*, 177–181.
- Cox, J. C., Ross, S. A., and Rubinstein, M. (1979). Option pricing: a simplified approach. *Journal of Financial Economics* 7(3), 229–263.
- Demeterfi, K., Derman, E., Kamal, M., and Zou, J. (1999). A guide to volatility and variance swaps. *Journal of Derivatives* 6(4), 9–32.
- Hull, J. C. *Options, Futures, and Other Derivatives*. Pearson. Chapters on properties of stock options (bounds, early exercise) and binomial trees.
- Merton, R. C. (1973). Theory of rational option pricing. *Bell Journal of Economics and Management Science* 4(1), 141–183.
