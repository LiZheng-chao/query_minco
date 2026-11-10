"""Map-independent B0 physical initializations for the manual Portal counterexample."""
import torch
from experiments.query_minco_p1a2_falsification_v1 import falsification as f
FAMILIES=("zero_lateral","random","symmetric_plus","symmetric_minus")
SEEDS=(None,20261109,20261110,20261111,20261112,20261113)
AMPLITUDE_M=1.0
Y_INDEX=(1,4,7,10,13)
def initial(state,family,seed):
    if family not in FAMILIES:raise ValueError("FAMILY")
    theta=f.actual_theta(torch.zeros(15),state,"B0").detach().double()
    eps=torch.zeros(15,dtype=torch.float64) if seed is None else .01*torch.tensor(f.p1.D_VALUES,dtype=torch.float64)*torch.randn(15,generator=torch.Generator().manual_seed(seed),dtype=torch.float64)
    if family!="random":eps[list(Y_INDEX)]=0.
    theta=theta+eps
    if family in ("symmetric_plus","symmetric_minus"):
        sign=1 if family=="symmetric_plus" else -1
        # Generic sextic reference y=sign*a*64*z^3*(1-z)^3, z=t/T.
        # It has zero lateral PVA at both ends; sample only inner positions.
        for index,z in ((1,1/3),(4,2/3)):theta[index]+=sign*AMPLITUDE_M*64*z**3*(1-z)**3
    if not bool(torch.isfinite(theta).all()) or bool((theta.abs()>=theta.new_tensor(f.p1.D_VALUES)).any()):raise ValueError("B0_OPEN_SUPPORT")
    return f.inverse_B(theta,state,"B0")
