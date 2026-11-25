# Feasible EE QP regression

`feasible_ee_qp.npz` stores the 14-variable primary problem (`H,g,C,l,u`) from
the common-input box timing probe at frame2256 (75.2s), with full seed42 prior.
The captured v4 path SHA256 is
`9fea6765451a924e1726d6c27df55ff29663ac9f582fe7841ff5362a27bb6999`.

ProxQP0.7.3 at `eps_abs=1e-8,max_iter=50,max_iter_in=50` returned
`PROXQP_MAX_ITER_REACHED` with zero primal residual and dual residual1.04083e-8.
Independent HiGHS linear feasibility passed. The regression preserves the
external2e-8 constraint check while testing a bounded DAQP fallback with the
same QP. Internal ProxQP `eps_abs` stays1e-8; DAQP must report optimal success.
