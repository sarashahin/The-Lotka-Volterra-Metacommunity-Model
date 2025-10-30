#!/usr/bin/env python 
# -*- coding: utf-8 -*-

# Copyright (C) 2010 Modelon AB
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

import numpy as np

from assimulo.ode import ID_PY_EVENT, ID_PY_COMPLETE, NORMAL, ID_PY_OK
from assimulo.explicit_ode import Explicit_ODE
from assimulo.exception import Explicit_ODE_Exception, AssimuloException


class EulerSimple(Explicit_ODE):
    """
    Fixed step Euler scheme with event handling at end of steps.
    """
    def __init__(self, problem):
        """
        Initiates the solver.
        
            Parameters::
            
                problem     
                            - The problem to be solved. Should be an instance
                              of the 'Explicit_Problem' class.                       
        """
        Explicit_ODE.__init__(self, problem) #Calls the base class
        
        #Solver options
        self.options["inith"] = 1
        self.options["maxsteps"] = 10000

        #Internal temporary result vector
        self.Y1 = np.array([0.0]*len(self.y0))
        
        #Solver support
        self.supports["report_continuously"] = True
        self.supports["interpolated_output"] = True
        self.supports["state_events"] = True
    
    def initialize(self):
        #Reset statistics
        self.statistics.reset()
            
    def set_problem_data(self): 
        def f(t, y): 
            return self.problem.rhs(t, y, self.sw)
        self.f = f
        if self.problem_info["state_events"]: 
            def event_func(t, y):
                try:
                    res = self.problem.state_events(t, y, self.sw)
                except BaseException as E:
                    self._py_err = E
                    return -1, None # non-recoverable
                return 0, res ## OK
            self.event_func = event_func
            self._event_info = [0] * self.problem_info["dimRoot"] 
            ret, self.g_old = self.event_func(self.t, self.y)
            if ret < 0:
                raise self._py_err
            self.statistics["nstatefcns"] += 1
    
    def _set_initial_step(self, initstep):
        try:
            initstep = float(initstep)
        except (ValueError, TypeError):
            raise Explicit_ODE_Exception('The initial step must be an integer or float.')
        
        self.options["inith"] = initstep
        
    def _get_initial_step(self):
        """
        This determines the initial step-size to be used in the integration.
        
            Parameters::
            
                inith    
                            - Default '1'.
                            
                            - Should be float.
                            
                                Example:
                                    inith = 1
        """
        return self.options["inith"]
        
    inith = property(_get_initial_step,_set_initial_step)
    
    def _get_maxsteps(self):
        """
        The maximum number of steps allowed to be taken to reach the
        final time.
        
            Parameters::
            
                maxsteps
                            - Default 10000
                            
                            - Should be a positive integer
        """
        return self.options["maxsteps"]
    
    def _set_maxsteps(self, max_steps):
        try:
            max_steps = int(max_steps)
        except (TypeError, ValueError):
            raise Explicit_ODE_Exception("Maximum number of steps must be a positive integer.")
        self.options["maxsteps"] = max_steps
    
    maxsteps = property(_get_maxsteps, _set_maxsteps)
    
    def step(self, t, y, tf, opts):
        initialize = opts["initialize"]
        
        if initialize:
            self.solver_iterator = self._iter(t,y,tf,opts)

        return self.solver_iterator.next()
    
    def integrate(self, t, y, tf, opts):
        """
        Integrates (t,y) values until t > tf
        """
        [flags, tlist, ylist] = zip(*list(self._iter(t, y, tf,opts)))
        
        return flags[-1], tlist, ylist

    def _simple_event_locator(self,t,y):
        n_g = self.problem_info["dimRoot"]
        ret, g_new = self.event_func(t, y)
        self.statistics["nstatefcns"] += 1
        if ret < 0:
            raise self._py_err
        event_info = np.zeros(n_g, dtype = int)
        ## faster, slightly less safe replacement for code below
        w = np.where((g_new > 0) != (self.g_old > 0))[0]
        if w.size == 0:
            flag = ID_PY_OK
        else:
            flag = ID_PY_EVENT
            for i in w:
                event_info[i] = 1 if g_new[i] > 0 else -1
        # flag = ID_PY_OK
        # for i in range(n_g):
        #     if (g_new[i] > 0) != (self.g_old[i] > 0):
        #         event_info[i] = 1 if g_new[i] > 0 else -1
        #         flag = ID_PY_EVENT
        self.set_event_info(event_info)
        self.statistics["nstateevents"] += 1
        self.g_old = g_new
        return flag
    
    def _iter(self,t,y,tf,opts):
        if opts["initialize"]:
            self.set_problem_data()
        maxsteps = self.options["maxsteps"]
        h = self.options["inith"]
        flag = ID_PY_OK
        
        for i in range(maxsteps):
            if t+h < tf and flag == ID_PY_OK:
                t, y = self._step(t, y, h)
                self.statistics["nsteps"] += 1
                if self.problem_info["state_events"]: 
                    flag = self._simple_event_locator(t, y) 
                
                if opts["report_continuously"]:
                    initialize_flag = self.report_solution(t, y, opts)
                    if initialize_flag: flag = ID_PY_EVENT
                    yield flag, t,y
                elif opts["output_list"] is None:
                    yield flag, t, y
                else:
                    output_list = opts["output_list"]
                    output_index = opts["output_index"]
                    try:
                        while output_list[output_index] <= t:
                            yield flag, output_list[output_index], self.interpolate(output_list[output_index])
                            output_index = output_index + 1
                    except IndexError:
                        pass
                    opts["output_index"] = output_index
            else:
                yield flag, t, y
                break
        else:
            raise Explicit_ODE_Exception('Final time not reached within maximum number of steps')
        
        #If no event has been detected, do the last step.
        if flag == ID_PY_OK:
            t, y = self._step(t, y, h)
            self.statistics["nsteps"] += 1
            if self.problem_info["state_events"]: 
                flag = self._simple_event_locator(t, y)
                if flag == ID_PY_OK: flag = ID_PY_COMPLETE
            if opts["report_continuously"]:
                initialize_flag = self.report_solution(t, y, opts)
                if initialize_flag: flag = ID_PY_EVENT
                else:               flag = ID_PY_COMPLETE
                yield flag, t,y
            elif opts["output_list"] is None:
                yield flag, t,y
            else:
                output_list = opts["output_list"]
                output_index = opts["output_index"]
                try:
                    while output_list[output_index] <= t:
                        yield flag, output_list[output_index], self.interpolate(output_list[output_index])
                        output_index = output_index + 1
                except IndexError:
                    pass
                    opts["output_index"] = output_index
    
    def _step(self, t, y, h):
        """
        This calculates the next step in the integration.
        """
        self.statistics["nfcns"] += 1
        

        t_next = t + h
        y_next = y + h*self.f(t, y)

        def interpolate(time):
            print(f"INTERPOLATING {time}")
            thetha = (time - t) / (t_next - t)
            return (1 - thetha) * y + thetha * y_next 
        self.interpolate = interpolate

        return t_next, y_next
        
    def state_event_info(self): 
        return self._event_info
        
    def set_event_info(self, event_info):
        self._event_info = event_info
    
    def print_statistics(self, verbose):
        """
        Should print the statistics.
        """
        Explicit_ODE.print_statistics(self, verbose) #Calls the base class
        
        log_message_verbose = lambda msg: self.log_message(msg, verbose)
        log_message_verbose('\nSolver options:\n')
        log_message_verbose(' Solver                  : SimpleEuler')
        log_message_verbose(' Solver type             : Dumb')
        log_message_verbose('')      