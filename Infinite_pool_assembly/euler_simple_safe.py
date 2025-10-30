#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Explicit Euler with        NO internal history
and fully working state-events for PSD2 / IBM models.

    • keeps only k communication points (ncp_list)
    • calls problem.handle_event(...) exactly like Assimulo
    • compatible with Assimulo ≥ 3.2
"""

import numpy as np
from assimulo.explicit_ode import Explicit_ODE

class EulerSimpleSafe(Explicit_ODE):
    def __init__(self, problem):
        super().__init__(problem)
        self.options["inith"]    = 1.0
        self.options["maxsteps"] = 10_000_000          # hard safety cap
        self.supports["report_continuously"] = False   # => no hidden list
        self.supports["state_events"]        = True

    # ------------------------------------------------------------------
    # light wrappers
    # ------------------------------------------------------------------
    def _f(self, t, y):
        return self.problem.rhs(t, y, self.sw)

    def _check_events(self, t, y):
        """Detect roots and fire problem.handle_event if any changed sign."""
        g = np.asarray(self.problem.state_events(t, y, self.sw), float)

        # first call → just cache and return
        if not hasattr(self, "_g_old"):
            self._g_old = g
            return

        changed = (g > 0) != (self._g_old > 0)
        if not changed.any():                   # nothing crossed zero
            self._g_old = g
            return

        rootsfound = np.where(changed, np.where(g > 0, 1, -1), 0)
        self._g_old = g

        # --- call the model’s discrete-event handler -------------------
        # signature:   handle_event(solver, (rootsfound, g))
        self.problem.handle_event(self, (rootsfound, g))

    # ------------------------------------------------------------------
    # fixed-step loop (no history)
    # ------------------------------------------------------------------
    def simulate(self, tfinal, ncp_list):
        h        = float(self.options["inith"])
        maxsteps = int(self.options["maxsteps"])
        t, y     = self.t0, self.y0.copy()

        cps      = np.sort(np.unique(np.append(ncp_list, tfinal)))
        T        = np.empty(len(cps));          T[0] = t
        Y        = np.empty((len(cps), len(y))); Y[0] = y
        nxt      = 1                                         # next cp slot

        for _ in range(maxsteps):
            if t >= tfinal:
                break

            ydot  = self._f(t, y)
            t_new = t + h
            y_new = y + h * ydot

            if self.problem_info["state_events"]:
                self.sw = self.sw or []          # ensure sw exists
                self._check_events(t_new, y_new) # may update solver.y etc.

            # --- linear output interpolation ---------------------------
            while nxt < len(cps) and t_new >= cps[nxt]:
                θ      = (cps[nxt] - t) / h
                T[nxt] = cps[nxt]
                Y[nxt] = y + θ * (y_new - y)
                nxt   += 1

            t, y = t_new, y_new                # advance step

        return T[:nxt], Y[:nxt]
