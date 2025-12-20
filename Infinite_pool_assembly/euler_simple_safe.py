############################################
# euler_simple_safe.py
############################################
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Explicit Euler with NO internal history.
Strict per-step event checking (high sync latency on GPU).
"""

from accelerator import np
import numpy as _cpu_numpy 

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
        self.problem = problem
        self.t0 = float(problem.t)
        self.y0 = problem.y
        self.sw = problem.sw
        
        self.options = {
            "inith": 1.0,
            "maxsteps": 10_000_000
        }
        self.supports = {
            "report_continuously": False,
            "state_events": True
        }
        self.problem_info = {
            "state_events": (problem.state_events is not None)
        }

    def _f(self, t, y):
        return self.problem.rhs(t, y, self.sw)

    def _check_events(self, t, y):
        g = np.asarray(self.problem.state_events(t, y, self.sw), float)

        if not hasattr(self, "_g_old"):
            self._g_old = g
            return

        changed = (g > 0) != (self._g_old > 0)
        
        if not np.any(changed):  
            self._g_old = g
            return

        rootsfound = np.where(changed, np.where(g > 0, 1, -1), 0)
        self._g_old = g
        self.problem.handle_event(self, (rootsfound, g))

    def simulate(self, tfinal, ncp_list=None):
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
        cps = _cpu_numpy.sort(_cpu_numpy.unique(_cpu_numpy.append(ncp_list, tfinal)))
        
        h        = float(self.options["inith"])
        maxsteps = int(self.options["maxsteps"])
        
        self.t = self.t0
        self.y = self.y0.copy() 

        has_events = self.problem_info["state_events"]
        if has_events:
            if self.sw is None: self.sw = []
            self._check_events(self.t, self.y) 

        T = _cpu_numpy.empty(len(cps))
        T[0] = self.t
        
        y_shape = self.y.shape
        Y = np.empty((len(cps), *y_shape)) 
        Y[0] = self.y
        
        nxt = 1

        for _ in range(maxsteps):
            if self.t >= tfinal:
                break

            y_start = self.y.clone() if hasattr(self.y, 'clone') else self.y.copy()
            t_start = self.t

            ydot   = self._f(t_start, y_start)
            self.t = t_start + h
            self.y = y_start + h * ydot

            # Strict Event Checking: Every Step
            if has_events:
                self._check_events(self.t, self.y) 

            while nxt < len(cps) and self.t >= cps[nxt]:
                if has_events: self._check_events(self.t, self.y)
                
                target_t = cps[nxt]
                theta  = (target_t - t_start) / h
                
                T[nxt] = target_t
                Y[nxt] = y_start + theta * (self.y - y_start)
                nxt   += 1

        return T[:nxt], Y[:nxt]
