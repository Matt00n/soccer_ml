# Surprise premium: robustness

Slope of realized minus predicted value on rarity within strata (goal-probability units per unit rarity).

## premium_specifications (20 rows)

| sample | spec | residual | slope | low | high | p | decisions | mean_q | slope_relative_to_mean_q |
|---|---|---|---|---|---|---|---|---|---|
| cross-fitted | Q deciles | res_td | 0.0016 | 0.0014 | 0.0018 | 0.0000 | 469733 | 0.0097 | 0.1626 |
| cross-fitted | Q deciles | res_mc | 0.0018 | 0.0012 | 0.0024 | 0.0000 | 469733 | 0.0097 | 0.1883 |
| cross-fitted | Q deciles x action type x third | res_td | 0.0019 | 0.0017 | 0.0021 | 0.0000 | 469733 | 0.0097 | 0.1913 |
| cross-fitted | Q deciles x action type x third | res_mc | 0.0020 | 0.0014 | 0.0026 | 0.0000 | 469733 | 0.0097 | 0.2033 |
| cross-fitted | passes only, Q deciles | res_td | 0.0024 | 0.0022 | 0.0026 | 0.0000 | 306312 | 0.0066 | 0.3617 |
| cross-fitted | passes only, Q deciles | res_mc | 0.0020 | 0.0012 | 0.0027 | 0.0000 | 306312 | 0.0066 | 0.2995 |
| cross-fitted | carries only, Q deciles | res_td | -0.0016 | -0.0019 | -0.0012 | 0.0000 | 126803 | 0.0105 | -0.1485 |
| cross-fitted | carries only, Q deciles | res_mc | -0.0004 | -0.0015 | 0.0008 | 0.5560 | 126803 | 0.0105 | -0.0340 |
| cross-fitted | targets in view, fine strata | res_td | 0.0012 | 0.0010 | 0.0014 | 0.0000 | 399435 | 0.0100 | 0.1188 |
| cross-fitted | targets in view, fine strata | res_mc | 0.0015 | 0.0008 | 0.0021 | 0.0000 | 399435 | 0.0100 | 0.1450 |
| reference | Q deciles | res_td | 0.0014 | 0.0011 | 0.0017 | 0.0000 | 166217 | 0.0100 | 0.1396 |
| reference | Q deciles | res_mc | 0.0029 | 0.0018 | 0.0040 | 0.0000 | 166217 | 0.0100 | 0.2916 |
| reference | Q deciles x action type x third | res_td | 0.0024 | 0.0021 | 0.0028 | 0.0000 | 166217 | 0.0100 | 0.2409 |
| reference | Q deciles x action type x third | res_mc | 0.0037 | 0.0026 | 0.0048 | 0.0000 | 166217 | 0.0100 | 0.3669 |
| reference | passes only, Q deciles | res_td | 0.0029 | 0.0026 | 0.0033 | 0.0000 | 111492 | 0.0070 | 0.4199 |
| reference | passes only, Q deciles | res_mc | 0.0036 | 0.0023 | 0.0048 | 0.0000 | 111492 | 0.0070 | 0.5101 |
| reference | carries only, Q deciles | res_td | -0.0021 | -0.0028 | -0.0015 | 0.0000 | 43887 | 0.0112 | -0.1889 |
| reference | carries only, Q deciles | res_mc | 0.0010 | -0.0011 | 0.0031 | 0.3860 | 43887 | 0.0112 | 0.0898 |
| reference | targets in view, fine strata | res_td | 0.0020 | 0.0016 | 0.0024 | 0.0000 | 146875 | 0.0103 | 0.1946 |
| reference | targets in view, fine strata | res_mc | 0.0033 | 0.0021 | 0.0046 | 0.0000 | 146875 | 0.0103 | 0.3193 |

