############################################
# euler_simple_safe.py
############################################
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Explicit Euler with NO internal history.
Strict per-step event checking (high sync latency on GPU).
Re-implemented to remove Assimulo dependency for Accelerator compatibility.
"""

from accelerator import np
import numpy as _cpu_numpy 

# Standalone replacement for Assimulo's Problem class
class Explicit_Problem:
    def __init__(self, rhs, y0, t0=0.0, sw0=None, state_events=None, handle_event=None):
        self.rhs = rhs
        self.y = np.array(y0, copy=True) 
        self.t = float(t0)               
        self.sw = sw0
        self.state_events = state_events
        self.handle_event = handle_event
        self.name = "Explicit_Problem"

class EulerSimpleSafe:
    def __init__(self, problem):
        # super().__init__(problem)  <-- Removed Assimulo inheritance
        self.problem = problem
        self.t0 = float(problem.t)
        self.y0 = problem.y
        self.sw = problem.sw
        
        self.options = {
            "inith": 1.0,
            "maxsteps": 10_000_000         # hard safety cap
        }
        self.supports = {
            "report_continuously": False,  # => no hidden list
            "state_events": True
        }
        self.problem_info = {
            "state_events": (problem.state_events is not None)
        }

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
        
        if not np.any(changed): # nothing crossed zero
            self._g_old = g
            return

        rootsfound = np.where(changed, np.where(g > 0, 1, -1), 0)
        self._g_old = g
        
        # --- call the model’s discrete-event handler -------------------
        self.problem.handle_event(self, (rootsfound, g))

    # ------------------------------------------------------------------
    # fixed-step loop (no history)
    # ------------------------------------------------------------------
    def simulate(self, tfinal, ncp_list=None):
        # Optimization: Ensure control flow variables are on CPU
        if ncp_list is None:
            ncp_list = _cpu_numpy.array([tfinal], dtype=float)
        else:
            if hasattr(ncp_list, 'cpu'):
                ncp_list = ncp_list.detach().cpu().numpy()
            elif isinstance(ncp_list, (list, tuple)):
                ncp_list = _cpu_numpy.array(ncp_list, dtype=float)
            elif hasattr(ncp_list, 'numpy'): 
                 ncp_list = ncp_list.numpy()
            else:
                 ncp_list = _cpu_numpy.asarray(ncp_list, dtype=float)

        tfinal = float(tfinal)
        # Use CPU numpy for the sorting/unique operations to avoid GPU sync
        cps = _cpu_numpy.sort(_cpu_numpy.unique(_cpu_numpy.append(ncp_list, tfinal)))
        
        h        = float(self.options["inith"])
        maxsteps = int(self.options["maxsteps"])
        
        # Initialize solver attributes
        self.t = self.t0
        self.y = self.y0.copy() 

        # 0. Initialise event checking
        has_events = self.problem_info["state_events"]
        if has_events:
            if self.sw is None: self.sw = []
            self._check_events(self.t, self.y) 

        # Output arrays
        # T matches cps length (on CPU)
        T = _cpu_numpy.empty(len(cps))
        T[0] = self.t
        
        # Y must match tensor backend
        y_shape = self.y.shape
        Y = np.empty((len(cps), *y_shape)) 
        Y[0] = self.y
        
        nxt = 1 # next cp slot

        for _ in range(maxsteps):
            if self.t >= tfinal:
                break

            # 1. Keep reference to start of step (needed for interpolation)
            y_start = self.y.clone() if hasattr(self.y, 'clone') else self.y.copy()
            t_start = self.t

            # 2. Calculate next step directly into self.y
            ydot   = self._f(t_start, y_start)
            self.t = t_start + h
            self.y = y_start + h * ydot

            # 3. Check events using the UPDATED self.y
            if has_events:
                self._check_events(self.t, self.y) 

            # 4. Interpolate output using y_start (old) and self.y (new)
            #    We use CPU comparison for self.t >= cps[nxt] to allow fast looping
            while nxt < len(cps) and self.t >= cps[nxt]:
                # Double check events before recording? (kept consistent with new logic)
                if has_events: self._check_events(self.t, self.y)
                
                target_t = cps[nxt]
                theta  = (target_t - t_start) / h
                
                T[nxt] = target_t
                Y[nxt] = y_start + theta * (self.y - y_start)
                nxt   += 1

        return T[:nxt], Y[:nxt]
