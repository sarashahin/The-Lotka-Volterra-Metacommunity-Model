############################################
# euler_simple_safe.py
############################################
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Explicit Euler with        NO internal history
and fully working state-events for PSD2 / IBM models.

    • keeps only k communication points (ncp_list)
    • calls problem.handle_event(...) exactly like Assimulo
    • compatible with Assimulo ≥ 3.2
"""

from accelerator import np
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
        
        # Initialize solver attributes
        self.t = self.t0
        self.y = self.y0.copy() 

        # 0. Initialise event checking
        if self.problem_info["state_events"]:
            self.sw = self.sw or []          
            self._check_events(self.t, self.y) 

        cps      = np.sort(np.unique(np.append(ncp_list, tfinal)))
        T        = np.empty(len(cps));          T[0] = self.t
        Y        = np.empty((len(cps), len(self.y))); Y[0] = self.y
        nxt      = 1                                         # next cp slot

        for _ in range(maxsteps):
            if self.t >= tfinal:
                break

            # 1. Keep reference to start of step (needed for interpolation)
            y_start = self.y
            t_start = self.t

            # 2. Calculate next step directly into self.y
            ydot   = self._f(t_start, y_start)
            self.t = t_start + h
            self.y = y_start + h * ydot

            # 3. Check events using the UPDATED self.y
            #    If the handler runs, it modifies self.y in place.
            if self.problem_info["state_events"]:
                self.sw = self.sw or []          
                self._check_events(self.t, self.y) 

            # 4. Interpolate output using y_start (old) and self.y (new)
            while nxt < len(cps) and self.t >= cps[nxt]:
                θ      = (cps[nxt] - t_start) / h
                T[nxt] = cps[nxt]
                Y[nxt] = y_start + θ * (self.y - y_start)
                nxt   += 1

        return T[:nxt], Y[:nxt]
