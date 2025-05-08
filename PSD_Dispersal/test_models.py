############################################
# test_models.py
############################################
"""
Simple tests/demonstrations of the model classes with minimal parameter sets.
"""
# import numpy as np
# from models_ibm import IBMModel
# from models_psd import PSDModel
# from models_psd2 import PSD2Model
# from models_ode import ODEModel
# from config import BODY_MASS

# def test_ibm():
#     # 2-species example
#     r = np.array([1.0, 0.9])
#     C = np.array([[1.0, 0.4],[0.4,1.0]])
#     model = IBMModel(r, C, nsteps=10000, record_step=1000)
#     traj = model.run()
#     print("IBM test completed. Final biomass snapshot:", traj[-1,:])

# def test_psd():
#     r = np.array([1.0, 0.9])
#     C = np.array([[1.0, 0.4],[0.4,1.0]])
#     model = PSDModel(r, C, nsteps=10000, record_step=1000)
#     traj_log, waiting = model.run()
#     # exponentiate for final snapshot
#     B = np.exp(traj_log[-1,:]) * (~waiting[-1,:])
#     print("PSD test completed. Final snapshot biomass:", B)

# def test_psd2():
#     r = np.array([1.0, 0.9])
#     C = np.array([[1.0, 0.4],[0.4,1.0]])
#     model = PSD2Model(r, C, tmax=10000, record_step=2000)
#     # Unpack all 7 outputs returned by PSD2Model.run()
#     t, traj, waiting, pc, gr, ir, ep = model.run()
#     print("PSD2 test completed. Final snapshot biomass:", traj[-1,:])

# def test_ode():
#     r = np.array([1.0, 0.9])
#     C = np.array([[1.0, 0.4],[0.4,1.0]])
#     model = ODEModel(r, C, tmax=10000, record_step=2000)
#     t, traj = model.run()
#     print("ODE test completed. Final snapshot biomass:", traj[-1,:])

# if __name__ == "__main__":
#     test_ibm()
#     test_psd()
#     test_psd2()
#     test_ode()


from models_psd2 import PSD2Model
import numpy as np, config

r = np.ones(3); C = np.eye(3); C[C==0] = 0.4
m = PSD2Model(r, C, tmax=3_000, record_step=50, seed=0)
for step, (t,B) in enumerate(zip(*m.run()[:2])):
    big = B.max()
    if big > 2*config.NUM_PATCHES_X*config.NUM_PATCHES_Y*config.BODY_MASS:
        print(f"⚠️  biomass explosion at t={t:.0f}, max={big:.2e}")
        break

