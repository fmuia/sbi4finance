# Note on this result

`audit.json` was computed on 22 October 2026 with an earlier audit likelihood that used the
variance σ_c² + s_c² at the Monte Carlo prices. That counts the Monte Carlo variance twice: averaged
over the Monte Carlo error it is the likelihood of a noise variance σ_c² + 2 s_c². The audit now uses
σ_c² − s_c², an unbiased estimate of the exact likelihood (`src/sbi4finance/calibration/correction.py`,
`tutorial_final/studies/audit.py`; design document, Appendix G.3). With a = s_c/σ_c the old version
widens the audited posterior by about a² in standard deviation. The study is to be rerun with the
current code before its numbers are used as final.
