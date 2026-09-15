import os
import numpy as np

_DATA_FEW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_few")

# Import relevant EMRI packages
import few
from few.trajectory.ode.base import ODEBase
from few.utils.geodesic import get_fundamental_frequencies, ELQ_to_pex

from typing import Union
from multispline.spline import CubicSpline

print("backend=",few.get_backend("cuda12x"))


# define trajectory RHS class
class SchwarzCirc1PASpinScalar(ODEBase):
    """
    Schwarzschild circular flux ODE.
    Args:
        use_ELQ: If True, the ODE will output derivatives of the orbital elements of (E, L, Q). Defaults to False.
    """

    def __init__(self, *args, use_ELQ: bool = False, **kwargs):
        super().__init__(*args, use_ELQ=use_ELQ, **kwargs)

        # construct the CubicSpline object from the expected file

        data = np.loadtxt(os.path.join(_DATA_FEW_DIR, "QuasiCircularData_new.dat"))
        data_scal = np.loadtxt(os.path.join(_DATA_FEW_DIR, "pdotCircScal_y.dat"))
        data_spin = np.loadtxt(os.path.join(_DATA_FEW_DIR, "QuasiCircularDataSpin_restructured.dat"))

        x = np.unique(data[:, 0])
        x_scal = np.unique(data_scal[:,0])

        # Building the flux interpolants here so they can be reused via self.<name> in evaluate_rhs.
        self.pdot_interp_0pa =  CubicSpline(x, data[:, 1])
        self.pdot_interp_1pa =  CubicSpline(x, data[:, 2])
        self.pdot_interp_spin =  CubicSpline(x, data_spin[:, 1])
        self.pdot_interp_scal =  CubicSpline(x_scal, data_scal[:, 1])

        self._min_p = 6.0 # Setting this to a lower value
        self._max_p = 30.0

    @property
    def equatorial(self):
        return True

    @property
    def background(self):
        return "Schwarzschild"

    @property
    def separatrix_buffer_dist(self):
        # 0.25 (not FEW's default 0.05/0.1) so that forward integration - which stops at
        # get_separatrix(...) + separatrix_buffer_dist, see FEW's stop_integrate_check, NOT
        # min_p()/max_p() below (those only validate the initial p0) - terminates exactly at
        # p=6.25 (6.0 Schwarzschild ISCO + 0.25), matching the flux grid's own lower validity
        # limit instead of overshooting into extrapolated territory.
        return 0.25

    @property
    def supports_ELQ(self):
        return False

    def min_p(self, e=0.0, x=1.0, a=0.0):
        return self._min_p

    def max_p(self, e=0.0, x=1.0, a=0.0):
        return self._max_p       

    #@property
    #def flux_output_convention(self):
    #    return "pex"

    def distance_to_outer_boundary(self, y):
        p, e, x = self.get_pex(y)
        dist_p = 30.0 - p
        return dist_p  

    def interpolate_flux_grids(
              self, p: float
              ) -> tuple[float]:

        #u = 1/Sqrt[scalar[[All,1]]-9/10 KerrGeoISCO[0,1]]
        u_scal = 1./np.sqrt(p-0.9*6)
        #y1 =  np.log((p - 2. * e - 2.1))

        # This explicit bounds check is intentionally NOT active: min_p()/max_p() (=6.0/30.0)
        # plus separatrix_buffer_dist and distance_to_outer_boundary already make FEW's own
        # integrator terminate forward integration at p=6.25 and backward integration at p=30
        # (see separatrix_buffer_dist docstring above), so p should never actually reach here
        # out of [6.25, 30]. Do not re-enable without re-checking that mechanism still holds.
        #if p < 6.25 or p > 30.:  # bounds described in 2104.04582
        #    raise ValueError("Interpolation: p out of bounds.")

        #yPN = Omega_phi**(2/3)


        f0PA0PN = - 64./(5*(p**3))
        f1PA0PN = 176./(5*(p**4))

        pdot_0pa = (self.pdot_interp_0pa(p)/(p**4))+f0PA0PN
        #if self.additional_args[2] != 1 and self.additional_args[2] != 0 :  #setting 1pa or 0pa
        #    raise ValueError("args for 1PA must be either 0 (for 0PA) or 1 (for 1PA)")
        #pdot_1pa = self.additional_args[2]*(((self.pdot_interp_1pa(p))/(p**5))+f1PA0PN)
        pdot_1pa = ((self.pdot_interp_1pa(p))/(p**5))+f1PA0PN
        #minus sign cause EdotGW = -Edot 
        pdot_scal = -(self.additional_args[0])*(self.pdot_interp_scal(u_scal))/4.
        pdot_spin = (self.additional_args[1])*(((self.pdot_interp_spin(p))/(p**5))+f1PA0PN)
        return [pdot_0pa, pdot_1pa, pdot_spin, pdot_scal]

    def evaluate_rhs(
        self, y: Union[list[float], np.ndarray]
    ) -> list[Union[float, np.ndarray]]:
        if self.use_ELQ:
            E, L, Q = y[:3]
            p, e, x = ELQ_to_pex(self.a, E, L, Q)

        else:
            p, e, x = y[:3]
        #eps=mu/M
        nu = self.massratio
        #nu = eps/((1+eps)**2)
        Omega_phi, Omega_theta, Omega_r = get_fundamental_frequencies(self.a, p, e, x)
        pdot_0pa, pdot_1pa, pdot_spin, pdot_scal = self.interpolate_flux_grids(p)
        pdot  = pdot_0pa+(nu*pdot_1pa)+(nu*pdot_spin)+pdot_scal
        return [pdot, 0., 0., Omega_phi, Omega_theta, Omega_r]



# define trajectory RHS class
class SchwarzCirc1PAScalar(ODEBase):
    """
    Schwarzschild circular flux ODE.
    Args:
        use_ELQ: If True, the ODE will output derivatives of the orbital elements of (E, L, Q). Defaults to False.
    """

    def __init__(self, *args, use_ELQ: bool = False, **kwargs):
        super().__init__(*args, use_ELQ=use_ELQ, **kwargs)

        # construct the CubicSpline object from the expected file

        data = np.loadtxt(os.path.join(_DATA_FEW_DIR, "QuasiCircularData_new.dat"))
        data_scal = np.loadtxt(os.path.join(_DATA_FEW_DIR, "pdotCircScal_y.dat"))

        x = np.unique(data[:, 0])
        x_scal = np.unique(data_scal[:,0])

        # Building the flux interpolants here so they can be reused via self.<name> in evaluate_rhs.
        self.pdot_interp_0pa =  CubicSpline(x, data[:, 1])
        self.pdot_interp_1pa =  CubicSpline(x, data[:, 2])
        self.pdot_interp_scal =  CubicSpline(x_scal, data_scal[:, 1])

        self._min_p = 6.0 # Setting this to a lower value
        self._max_p = 30.0

    @property
    def equatorial(self):
        return True

    @property
    def background(self):
        return "Schwarzschild"

    @property
    def supports_ELQ(self):
        return False

    @property
    def separatrix_buffer_dist(self):
        # 0.25 (not FEW's default 0.05/0.1) so that forward integration - which stops at
        # get_separatrix(...) + separatrix_buffer_dist, see FEW's stop_integrate_check, NOT
        # min_p()/max_p() below (those only validate the initial p0) - terminates exactly at
        # p=6.25 (6.0 Schwarzschild ISCO + 0.25), matching the flux grid's own lower validity
        # limit instead of overshooting into extrapolated territory.
        return 0.25

    def min_p(self, e=0.0, x=1.0, a=0.0):
        return self._min_p

    def max_p(self, e=0.0, x=1.0, a=0.0):
        return self._max_p    

    #@property
    #def flux_output_convention(self):
    #    return "pex"

    def distance_to_outer_boundary(self, y):
        p, e, x = self.get_pex(y)
        dist_p = 30.0 - p
        return dist_p

    def interpolate_flux_grids(
              self, p: float
              ) -> tuple[float]:

        #u = 1/Sqrt[scalar[[All,1]]-9/10 KerrGeoISCO[0,1]]
        u_scal = 1./np.sqrt(p-0.9*6)
        #y1 =  np.log((p - 2. * e - 2.1))

        # This explicit bounds check is intentionally NOT active: min_p()/max_p() (=6.0/30.0)
        # plus separatrix_buffer_dist and distance_to_outer_boundary already make FEW's own
        # integrator terminate forward integration at p=6.25 and backward integration at p=30
        # (see separatrix_buffer_dist docstring above), so p should never actually reach here
        # out of [6.25, 30]. Do not re-enable without re-checking that mechanism still holds.
        #if p < 6.25 or p > 30.:  # bounds described in 2104.04582
        #    raise ValueError("Interpolation: p out of bounds.")

        #yPN = Omega_phi**(2/3)

        #if p < 6.25 or p > 30  or np.isnan(p):
        #    raise ValueError("Interpolation: p out of bounds.")

        f0PA0PN = - 64./(5*(p**3))
        f1PA0PN = 176./(5*(p**4))

        pdot_0pa = (self.pdot_interp_0pa(p)/(p**4))+f0PA0PN
        pdot_1pa = ((self.pdot_interp_1pa(p))/(p**5))+f1PA0PN
        #minus sign cause EdotGW = -Edot 
        pdot_scal = -(self.additional_args[0])*(self.pdot_interp_scal(u_scal))/4.
        return [pdot_0pa, pdot_1pa, pdot_scal]

    def evaluate_rhs(
        self, y: Union[list[float], np.ndarray]
    ) -> list[Union[float, np.ndarray]]:
        if self.use_ELQ:
            E, L, Q = y[:3]
            p, e, x = ELQ_to_pex(self.a, E, L, Q)

        else:
            p, e, x = y[:3]
        #eps=mu/M
        nu = self.massratio
        #nu = eps/((1+eps)**2)
        Omega_phi, Omega_theta, Omega_r = get_fundamental_frequencies(self.a, p, e, x)
        pdot_0pa, pdot_1pa, pdot_scal = self.interpolate_flux_grids(p)
         
        #pdot  = (nu/eps)*(pdot_0pa+(nu*pdot_1pa)+pdot_scal)
        #print("pdot_0pa",pdot_0pa) 
        #print("pdot_0pa+(nu*pdot_1pa)=",pdot_0pa+(nu*pdot_1pa)) 
        pdot  = pdot_0pa+(nu*pdot_1pa)+pdot_scal
        return [pdot, 0., 0., Omega_phi, Omega_theta, Omega_r]


# define trajectory RHS class
class SchwarzCirc0PAScalarS(ODEBase):
    """
    Schwarzschild circular flux ODE.
    Args:
        use_ELQ: If True, the ODE will output derivatives of the orbital elements of (E, L, Q). Defaults to False.
    """

    def __init__(self, *args, use_ELQ: bool = False, **kwargs):
        super().__init__(*args, use_ELQ=use_ELQ, **kwargs)

        # construct the CubicSpline object from the expected file
        data = np.loadtxt(os.path.join(_DATA_FEW_DIR, "QuasiCircularData_new.dat"))
        data_scal = np.loadtxt(os.path.join(_DATA_FEW_DIR, "pdotCircScal_y.dat"))

        x = np.unique(data[:, 0])
        x_scal = np.unique(data_scal[:,0])

        # Building the flux interpolants here so they can be reused via self.<name> in evaluate_rhs.
        self.pdot_interp_0pa =  CubicSpline(x, data[:, 1])
        self.pdot_interp_1pa =  CubicSpline(x, data[:, 2])
        self.pdot_interp_scal = CubicSpline(x_scal, data_scal[:, 1])

        self._min_p = 6.0 # Setting this to a lower value
        self._max_p = 30.0

    @property
    def equatorial(self):
        return True

    @property
    def background(self):
        return "Schwarzschild"

    @property
    def separatrix_buffer_dist(self):
        # 0.25 (not FEW's default 0.05/0.1) so that forward integration - which stops at
        # get_separatrix(...) + separatrix_buffer_dist, see FEW's stop_integrate_check, NOT
        # min_p()/max_p() below (those only validate the initial p0) - terminates exactly at
        # p=6.25 (6.0 Schwarzschild ISCO + 0.25), matching the flux grid's own lower validity
        # limit instead of overshooting into extrapolated territory.
        return 0.25

    @property
    def supports_ELQ(self):
        return False

    def min_p(self, e=0.0, x=1.0, a=0.0):
        return self._min_p

    def max_p(self, e=0.0, x=1.0, a=0.0):
        return self._max_p    

    #@property
    #def flux_output_convention(self):
    #    return "pex"

    def distance_to_outer_boundary(self, y):
        p, e, x = self.get_pex(y)
        dist_p = 30.0 - p
        return dist_p

    def interpolate_flux_grids(
              self, p: float
              ) -> tuple[float]:

        #u = 1/Sqrt[scalar[[All,1]]-9/10 KerrGeoISCO[0,1]]
        u_scal = 1./np.sqrt(p-0.9*6)
        #y1 =  np.log((p - 2. * e - 2.1))

        # This explicit bounds check is intentionally NOT active: min_p()/max_p() (=6.0/30.0)
        # plus separatrix_buffer_dist and distance_to_outer_boundary already make FEW's own
        # integrator terminate forward integration at p=6.25 and backward integration at p=30
        # (see separatrix_buffer_dist docstring above), so p should never actually reach here
        # out of [6.25, 30]. Do not re-enable without re-checking that mechanism still holds.
        #if p < 6.25 or p > 30.:  # bounds described in 2104.04582
        #    raise ValueError("Interpolation: p out of bounds.")

        #yPN = Omega_phi**(2/3)


        f0PA0PN = - 64./(5*(p**3))
        f1PA0PN = 176./(5*(p**4))

        pdot_0pa = (self.pdot_interp_0pa(p)/(p**4))+f0PA0PN
        #if self.additional_args[2] != 1 and self.additional_args[2] != 0 :  #setting 1pa or 0pa
        #    raise ValueError("args for 1PA must be either 0 (for 0PA) or 1 (for 1PA)")
        #pdot_1pa = self.additional_args[2]*(((self.pdot_interp_1pa(p))/(p**5))+f1PA0PN)
        #PAorder = 0.
        #pdot_1pa = PAorder*(((self.pdot_interp_1pa(p))/(p**5))+f1PA0PN)
        #minus sign cause EdotGW = -Edot 
        pdot_scal = -(self.additional_args[0])*(self.pdot_interp_scal(u_scal))/4.
        return [pdot_0pa, pdot_scal]

    def evaluate_rhs(
        self, y: Union[list[float], np.ndarray]
    ) -> list[Union[float, np.ndarray]]:
        if self.use_ELQ:
            E, L, Q = y[:3]
            p, e, x = ELQ_to_pex(self.a, E, L, Q)

        else:
            p, e, x = y[:3]
        #eps=mu/M
        #eps = self.epsilon
        #nu = eps/((1+eps)**2)
        Omega_phi, Omega_theta, Omega_r = get_fundamental_frequencies(self.a, p, e, x)
        pdot_0pa, pdot_scal = self.interpolate_flux_grids(p)
         
        pdot  = (pdot_0pa+pdot_scal)
        return [pdot, 0., 0., Omega_phi, Omega_theta, Omega_r]



class SchwarzCirc0PASpinScalar(ODEBase):
    """
    Schwarzschild circular flux ODE.
    Args:
        use_ELQ: If True, the ODE will output derivatives of the orbital elements of (E, L, Q). Defaults to False.
    """

    def __init__(self, *args, use_ELQ: bool = False, **kwargs):
        super().__init__(*args, use_ELQ=use_ELQ, **kwargs)

        # construct the CubicSpline object from the expected file

        data = np.loadtxt(os.path.join(_DATA_FEW_DIR, "QuasiCircularData_new.dat"))
        data_scal = np.loadtxt(os.path.join(_DATA_FEW_DIR, "pdotCircScal_y.dat"))
        data_spin = np.loadtxt(os.path.join(_DATA_FEW_DIR, "QuasiCircularDataSpin_restructured.dat"))

        x = np.unique(data[:, 0])
        x_scal = np.unique(data_scal[:,0])

        # Building the flux interpolants here so they can be reused via self.<name> in evaluate_rhs.
        self.pdot_interp_0pa =  CubicSpline(x, data[:, 1])
        self.pdot_interp_spin =  CubicSpline(x, data_spin[:, 1])
        self.pdot_interp_scal =  CubicSpline(x_scal, data_scal[:, 1])

        self._min_p = 6.0
        self._max_p = 30.0

    @property
    def equatorial(self):
        return True

    @property
    def background(self):
        return "Schwarzschild"

    @property
    def separatrix_buffer_dist(self):
        # 0.25 (not FEW's default 0.05/0.1) so that forward integration - which stops at
        # get_separatrix(...) + separatrix_buffer_dist, see FEW's stop_integrate_check, NOT
        # min_p()/max_p() below (those only validate the initial p0) - terminates exactly at
        # p=6.25 (6.0 Schwarzschild ISCO + 0.25), matching the flux grid's own lower validity
        # limit instead of overshooting into extrapolated territory.
        return 0.25

    @property
    def supports_ELQ(self):
        return False

    def min_p(self, e=0.0, x=1.0, a=0.0):
        return self._min_p

    def max_p(self, e=0.0, x=1.0, a=0.0):
        return self._max_p

    #@property
    #def flux_output_convention(self):
    #    return "pex"

    def distance_to_outer_boundary(self, y):
        p, e, x = self.get_pex(y)
        dist_p = 30.0 - p
        return dist_p

    def interpolate_flux_grids(
              self, p: float
              ) -> tuple[float]:

        #u = 1/Sqrt[scalar[[All,1]]-9/10 KerrGeoISCO[0,1]]
        u_scal = 1./np.sqrt(p-0.9*6)
        #y1 =  np.log((p - 2. * e - 2.1))

        # This explicit bounds check is intentionally NOT active: min_p()/max_p() (=6.0/30.0)
        # plus separatrix_buffer_dist and distance_to_outer_boundary already make FEW's own
        # integrator terminate forward integration at p=6.25 and backward integration at p=30
        # (see separatrix_buffer_dist docstring above), so p should never actually reach here
        # out of [6.25, 30]. Do not re-enable without re-checking that mechanism still holds.
        #if p < 6.25 or p > 30.:  # bounds described in 2104.04582
        #    raise ValueError("Interpolation: p out of bounds.")

        #yPN = Omega_phi**(2/3)


        f0PA0PN = - 64./(5*(p**3))
        f1PA0PN = 176./(5*(p**4))

        pdot_0pa = (self.pdot_interp_0pa(p)/(p**4))+f0PA0PN
        #if self.additional_args[2] != 1 and self.additional_args[2] != 0 :  #setting 1pa or 0pa
        #    raise ValueError("args for 1PA must be either 0 (for 0PA) or 1 (for 1PA)")
        #pdot_1pa = self.additional_args[2]*(((self.pdot_interp_1pa(p))/(p**5))+f1PA0PN)
        #pdot_1pa = PAorder*(((self.pdot_interp_1pa(p))/(p**5))+f1PA0PN)
        #minus sign cause EdotGW = -Edot 
        pdot_scal = -(self.additional_args[0])*(self.pdot_interp_scal(u_scal))/4.
        pdot_spin = (self.additional_args[1])*(((self.pdot_interp_spin(p))/(p**5))+f1PA0PN)
        return [pdot_0pa, pdot_spin, pdot_scal]

    def evaluate_rhs(
        self, y: Union[list[float], np.ndarray]
    ) -> list[Union[float, np.ndarray]]:
        if self.use_ELQ:
            E, L, Q = y[:3]
            p, e, x = ELQ_to_pex(self.a, E, L, Q)

        else:
            p, e, x = y[:3]
        #eps=mu/M
        nu = self.massratio
        #nu = eps/((1+eps)**2)
        Omega_phi, Omega_theta, Omega_r = get_fundamental_frequencies(self.a, p, e, x)
        pdot_0pa, pdot_spin, pdot_scal = self.interpolate_flux_grids(p)
         
        pdot  = pdot_0pa+(nu*pdot_spin)+pdot_scal
        return [pdot, 0., 0., Omega_phi, Omega_theta, Omega_r]        

# define trajectory RHS class
class SchwarzCircPNScalar(ODEBase):
    """
    Schwarzschild circular flux ODE.
    Args:
        use_ELQ: If True, the ODE will output derivatives of the orbital elements of (E, L, Q). Defaults to False.
    """

    def __init__(self, *args, use_ELQ: bool = False, **kwargs):
        super().__init__(*args, use_ELQ=use_ELQ, **kwargs)

        # construct the CubicSpline object from the expected file
        data = np.loadtxt(os.path.join(_DATA_FEW_DIR, "QuasiCircularData_new.dat"))

        x = np.unique(data[:, 0])

        # Building the flux interpolants here so they can be reused via self.<name> in evaluate_rhs.
        self.pdot_interp_0pa =  CubicSpline(x, data[:, 1])
        self.pdot_interp_1pa =  CubicSpline(x, data[:, 2])

        self._min_p = 6.0
        self._max_p = 30.0

    @property
    def equatorial(self):
        return True

    @property
    def background(self):
        return "Schwarzschild"

    @property
    def separatrix_buffer_dist(self):
        # 0.25 (not FEW's default 0.05/0.1) so that forward integration - which stops at
        # get_separatrix(...) + separatrix_buffer_dist, see FEW's stop_integrate_check, NOT
        # min_p()/max_p() below (those only validate the initial p0) - terminates exactly at
        # p=6.25 (6.0 Schwarzschild ISCO + 0.25), matching the flux grid's own lower validity
        # limit instead of overshooting into extrapolated territory. Note: this class's pdot_0pa
        # still uses the same numerical grid as the SchwarzCirc* classes, hence the same fix.
        return 0.25

    @property
    def supports_ELQ(self):
        return False

    def min_p(self, e=0.0, x=1.0, a=0.0):
        return self._min_p

    def max_p(self, e=0.0, x=1.0, a=0.0):
        return self._max_p

    #@property
    #def flux_output_convention(self):
    #    return "pex"

    def distance_to_outer_boundary(self, y):
        p, e, x = self.get_pex(y)
        dist_p = 30.0 - p
        return dist_p

    def interpolate_flux_grids(
              self, p: float
              ) -> tuple[float]:

        #u = 1/Sqrt[scalar[[All,1]]-9/10 KerrGeoISCO[0,1]]
        #y1 =  np.log((p - 2. * e - 2.1))

        # This explicit bounds check is intentionally NOT active: min_p()/max_p() (=6.0/30.0)
        # plus separatrix_buffer_dist and distance_to_outer_boundary already make FEW's own
        # integrator terminate forward integration at p=6.25 and backward integration at p=30
        # (see separatrix_buffer_dist docstring above), so p should never actually reach here
        # out of [6.25, 30]. Do not re-enable without re-checking that mechanism still holds.
        #if p < 6.25 or p > 30.:  # bounds described in 2104.04582
        #    raise ValueError("Interpolation: p out of bounds.")

        #yPN = Omega_phi**(2/3)


        f0PA0PN = - 64./(5*(p**3))
        f1PA0PN = 176./(5*(p**4))

        pdot_0pa = (self.pdot_interp_0pa(p)/(p**4))+f0PA0PN
        pdot_scal = - self.additional_args[0]*((p-3.)**(3./2)/(6.*(p-6.)*(p**(5./2)))) #Edot = A/(p^4) , pdot = (2 A (-3 + p)^(3/2))/( (-6 + p) p^(5/2)), A= Lambda/12
        return [pdot_0pa, pdot_scal]

    def evaluate_rhs(
        self, y: Union[list[float], np.ndarray]
    ) -> list[Union[float, np.ndarray]]:
        if self.use_ELQ:
            E, L, Q = y[:3]
            p, e, x = ELQ_to_pex(self.a, E, L, Q)

        else:
            p, e, x = y[:3]
        #eps=mu/M
        #eps = self.epsilon
        #nu = eps/((1+eps)**2)
        Omega_phi, Omega_theta, Omega_r = get_fundamental_frequencies(self.a, p, e, x)
        pdot_0pa, pdot_scal = self.interpolate_flux_grids(p)
         
        pdot  = (pdot_0pa+pdot_scal)
        return [pdot, 0., 0., Omega_phi, Omega_theta, Omega_r]        

class HybridSFPN_1PAScalar_DenominatorExpanded(ODEBase):
    """
    Schwarzschild circular flux ODE.
    Args:
        use_ELQ: If True, the ODE will output derivatives of the orbital elements of (E, L, Q). Defaults to False.
    """

    def __init__(self, *args, use_ELQ: bool = False, **kwargs):
        super().__init__(*args, use_ELQ=use_ELQ, **kwargs)

        #construct the CubicSpline object from the expected file
        data = np.loadtxt(os.path.join(_DATA_FEW_DIR, "Numerical_F1_F2_EFLx.dat"))
        datagrav_0pa = np.loadtxt(os.path.join(_DATA_FEW_DIR, "Numerical_Grav0pa_to50.dat"))
        data_scal = np.loadtxt(os.path.join(_DATA_FEW_DIR, "pdotCircScal_y_ext.dat"))
        
        x_scal = np.unique(data_scal[:,0]) 
        x = np.unique(data[:, 0])

        self.F1_interp =  CubicSpline(np.unique(datagrav_0pa[:, 0]), datagrav_0pa[:, 1])
        self.F2_interp =  CubicSpline(x, data[:, 2])
        self.EFLx_prime_interp =  CubicSpline(x, data[:, 3])
        self.pdot_interp_scal =  CubicSpline(x_scal, data_scal[:, 1])

        self._min_p = 6.0 # Setting this to a lower value
        self._max_p = 50.0

    @property
    def equatorial(self):
        return True

    @property
    def background(self):
        return "Schwarzschild"

    @property
    def supports_ELQ(self):
        return False

    @property
    def separatrix_buffer_dist(self):
        return 0.251

    def min_p(self, e=0.0, x=1.0, a=0.0):
        return self._min_p

    def max_p(self, e=0.0, x=1.0, a=0.0):
        return self._max_p     

    def distance_to_outer_boundary(self, y):
        p, e, x = self.get_pex(y)
        dist_p = 50.0 - p
        return dist_p       

    def interpolate_flux_grids(
              self, p: float
              ) -> tuple[float,float,float]:

        #if p < 6.25 or p > 30.: 
        #    raise ValueError("Interpolation: p out of bounds.")

        f0PA0PN = - 64./(5*(p**3))
        f1PA0PN = 176./(5*(p**4))

        F1 = (self.F1_interp(p)/p**4)+f0PA0PN
        F2 = (self.F2_interp(p)/p**5)+f1PA0PN
        EFLx_prime = self.EFLx_prime_interp(p)/p**5

        return [F1, F2, EFLx_prime]

    def interpolate_flux_grav0pa(
              self, p: float
              ) -> float:     

        #if p < 6.25 or p > 60.: 
        #    raise ValueError("Interpolation: p scalar out of bounds.")

        f0PA0PN = - 64./(5*(p**3))
        F1 = (self.F1_interp(p)/p**4)+f0PA0PN

        return F1


    def interpolate_flux_scalar(
              self, p: float
              ) -> float:     

        #if p < 6.25 or p > 60.: 
        #    raise ValueError("Interpolation: p scalar out of bounds.")

        u_scal = 1./np.sqrt(p-0.9*6)
        pdot_scal = - (self.additional_args[0])*(self.pdot_interp_scal(u_scal))/4.

        return pdot_scal  

    def F2PN(self, r0):
        pi = np.pi
        EulerGamma = np.euler_gamma

        x = 1.0 / r0

        x2 = x * x
        x4 = x2 * x2
        x5 = x4 * x
        x6 = x5 * x
        x7 = x6 * x
        x8 = x7 * x
        x9 = x8 * x

        x15_2 = x7 * np.sqrt(x)
        x17_2 = x15_2 * x
        x19_2 = x17_2 * x

        log_r0 = -np.log(r0)
        log2 = np.log(2.0)
        log3 = np.log(3.0)

        term = (
            - (56.0 / 3.0) * x6
            + (37084.0 / 315.0) * x7
            + (
                -134543.0 / 1215.0
                + (82.0 / 15.0) * pi**2
            ) * x8
            + (
                -1452202403629.0 / 229209750.0
                + (1327296.0 / 1225.0) * EulerGamma
                - (267127.0 / 720.0) * pi**2
                + (15329984.0 / 11025.0) * log2
                + (37908.0 / 49.0) * log3
                + (663648.0 / 1225.0) * log_r0
            ) * x9
            - (2332.0 / 15.0) * pi * x15_2
            + (42949.0 / 54.0) * pi * x17_2
            + (
                2062241.0 / 3465.0
                + (328.0 / 15.0) * pi**2
            ) * pi * x19_2
            )

        return term


    #def DeltaEPN_prime(self, nu, r0):
    #    numerator = nu**2 * ( -852390 * np.pi**2 + 3888 * r0**2 + 120 * r0 * (1674 + 7 * nu) - 35 * (-640863 + 774 * nu + 11 * nu**2))
    #    denominator = 62208 * r0**4
    #    return numerator / denominator
   
    def EPN_prime(self, nu, r0):
        pi = np.pi
        x0 = 1.0/r0
        x0_2 = x0*x0
        x0_3 = x0_2*x0
        x0_4 = x0_2*x0_2
        EulerGamma = 0.57721566490153286060
        
        c1 = 103680.0 * (9.0 + nu)

        c2 = 77760.0  * ((nu - 57.0) * nu + 81)

        c3 = 2400.0*(
            ((7.0*nu+1674.0)*nu
                + 9 * (-6889.0 + 246.0 * pi**2) 
            )* nu
                +10935.0 
            )

        c4 = (
            (((-7700.0 * nu - 541800.0) * nu
            + (448604100.0 - 17047800.0 * pi**2)) * nu
            + (48202884.0
             - 185794560.0 * EulerGamma
             - 18299925.0 * pi**2
              - 92897280.0 * (np.log(16.0) - np.log(r0))
             )
                ) * nu
            + 96446700.0
            )
        
        numerator = (
            - 622080.0 
            + c1 * x0
            + c2 * x0_2
            + c3 * x0_3 
            + c4 * x0_4
        )    

        denominator = 1244160 

        return numerator / denominator


    def E1PN_prime(self, r0):
        pi = np.pi
        EulerGamma = 0.57721566490153286060
        
        x0 = 1.0 / r0
        x0_2 = x0 * x0
        x0_3 = x0_2 * x0
        x0_4 = x0_2 * x0_2

        log_term = np.log(16.0) - np.log(r0)

        c1 = 3840.0
        c2 = -(164160.0)
        c3 = -(5511200.0 - 196800.0 * pi**2)
        c4 = 1785292.0 - 6881280.0 * EulerGamma - 677775.0 * pi**2 - 3440640.0 * log_term

        expr = (
            c1 * x0
            + c2 * x0_2
            + c3 * x0_3
            + c4 * x0_4
            ) / 46080.0

        return expr       

    def F0HybridExtended(self, nu, r0):
        # NOTE ON NAMING: the local F1/F2 below are the *numerical grid* columns
        # (interpolate_flux_grids' raw script-flux data, i.e. the paper's calligraphic
        # F0^GSF/F1^GSF in Eq. 18), unrelated to this *method's* name F0HybridExtended or to
        # F1HybridExtended just below - three different things share similar names here.
        # DENOMINATOR EXPANSION (paper Eq. 18/30, hence the class name): a(r_hat) = term1 =
        # (dE0/dr_hat)^-1 for circular Schwarzschild. F0HybridExtended = num*a(r_hat) is the
        # O(nu^0) piece; the O(nu) correction to the denominator (term2, from dE1/dr_hat) is
        # deliberately NOT included here - it is instead folded into F1HybridExtended via
        # -F1_flux*term2/term1**2 (the first-order Taylor expansion of 1/(term1+nu*term2)),
        # so FHybridExtended = F0 + nu*F1 already carries it at the correct order.

        if 6.25 <= r0 <= 30.:
            F1, F2, EFLx_prime = self.interpolate_flux_grids(r0)
            num = - F1
            term1 = -( (1 - 6 / r0) / (3 * (1 - 3 / r0)**(3/2) * np.sqrt(1 / r0)) )
            term2 = nu * (2/3) * r0**(1/2) * EFLx_prime
            #term3 = (2/3) * r0**(1/2) * self.DeltaEPN_prime(nu, r0)
            return num/(term1)  # O(nu) denominator term folded into F1HybridExtended instead, see note above

        elif r0 > 30.:
            F1 = self.interpolate_flux_grav0pa(r0) 
            num = - F1
            term1 = -( (1 - 6 / r0) / (3 * (1 - 3 / r0)**(3/2) * np.sqrt(1 / r0)) )
            term2 = nu * (2/3) * r0**(1/2) * self.E1PN_prime(r0)
            #term3 = (2/3) * r0**(1/2) * self.DeltaEPN_prime(nu, r0)
            return num/(term1)  # O(nu) denominator term folded into F1HybridExtended instead, see note above

        else:
            raise ValueError("Interpolation: p out of bounds.")

    def F1HybridExtended(self, nu, r0):
        #x = 1 / r0 
    
        if 6.25 <= r0 <= 30.:
            F1, F2, EFLx_prime = self.interpolate_flux_grids(r0)
            num = - F2
            term1 = -( (1 - 6 / r0) / (3 * (1 - 3 / r0)**(3/2) * np.sqrt(1 / r0)) )
            term2 = nu * (2/3) * r0**(1/2) * EFLx_prime
            #term3 = (2/3) * r0**(1/2) * self.DeltaEPN_prime(nu, r0)
            return num/(term1) - F1*term2/(term1**2)  # denominator-expansion cross term, see note in F0HybridExtended above

        elif r0 > 30.:
            F1 = self.interpolate_flux_grav0pa(r0) 
            num = - self.F2PN(r0)
            term1 = -( (1 - 6 / r0) / (3 * (1 - 3 / r0)**(3/2) * np.sqrt(1 / r0)) )
            term2 = nu * (2/3) * r0**(1/2) * self.E1PN_prime(r0)
            #term3 = (2/3) * r0**(1/2) * self.DeltaEPN_prime(nu, r0)
            return num/(term1) - F1*term2/(term1**2)  # denominator-expansion cross term, see note in F0HybridExtended above
            
        else:
            raise ValueError("Interpolation: p out of bounds.")

    def FHybridExtended(self,nu, r0):
        return  self.F0HybridExtended(nu, r0) + nu * self.F1HybridExtended(nu, r0)

    def evaluate_rhs(
        self, y: Union[list[float], np.ndarray]
    ) -> list[float]:
        if self.use_ELQ:
            E, L, Q = y[:3]
            p, e, x = ELQ_to_pex(self.a, E, L, Q)

        else:
            p, e, x = y[:3]
        #eps=mu/M
        nu = self.massratio #here eps = new_mu/new_M = \nu! 
        #nu = eps/((1+eps)**2) #if input M, m
        Omega_phi, Omega_theta, Omega_r = get_fundamental_frequencies(self.a, p, e, x)
        omegadot = self.FHybridExtended(nu, p)
        domegadr = -(3/(2*p**(5/2)))
        pdot  =  omegadot/domegadr
        #pdot = pdot/(eps) #if input M, m
        pdot = pdot #if input new_M=m+M, new_m = mu*M/(mu+M)
        pdot_scal = self.interpolate_flux_scalar(p)
        return [pdot+pdot_scal, 0., 0., Omega_phi, Omega_theta, Omega_r]



class HybridSFPN_0PAScalar_DenominatorExpanded(ODEBase):
    """
    Schwarzschild circular flux ODE.
    Args:
        use_ELQ: If True, the ODE will output derivatives of the orbital elements of (E, L, Q). Defaults to False.
    """

    def __init__(self, *args, use_ELQ: bool = False, **kwargs):
        super().__init__(*args, use_ELQ=use_ELQ, **kwargs)

        #construct the CubicSpline object from the expected file
        data = np.loadtxt(os.path.join(_DATA_FEW_DIR, "Numerical_F1_F2_EFLx.dat"))
        datagrav_0pa = np.loadtxt(os.path.join(_DATA_FEW_DIR, "Numerical_Grav0pa_to50.dat"))
        data_scal = np.loadtxt(os.path.join(_DATA_FEW_DIR, "pdotCircScal_y_ext.dat"))
        
        x_scal = np.unique(data_scal[:,0]) 
        x = np.unique(data[:, 0])

        self.F1_interp =  CubicSpline(np.unique(datagrav_0pa[:, 0]), datagrav_0pa[:, 1])
        self.F2_interp =  CubicSpline(x, data[:, 2])
        self.EFLx_prime_interp =  CubicSpline(x, data[:, 3])
        self.pdot_interp_scal =  CubicSpline(x_scal, data_scal[:, 1])

        self._min_p = 6.0 # Setting this to a lower value
        self._max_p = 50.0

    @property
    def equatorial(self):
        return True

    @property
    def background(self):
        return "Schwarzschild"

    @property
    def separatrix_buffer_dist(self):
        return 0.251

    @property
    def supports_ELQ(self):
        return False

    def min_p(self, e=0.0, x=1.0, a=0.0):
        return self._min_p

    def max_p(self, e=0.0, x=1.0, a=0.0):
        return self._max_p     

    def distance_to_outer_boundary(self, y):
        p, e, x = self.get_pex(y)
        dist_p = 50.0 - p
        return dist_p          

    def interpolate_flux_grids(
              self, p: float
              ) -> tuple[float,float,float]:

        #if p < 6.25 or p > 60.: 
        #    raise ValueError("Interpolation: p out of bounds.")

        f0PA0PN = - 64./(5*(p**3))
        f1PA0PN = 176./(5*(p**4))

        F1 = (self.F1_interp(p)/p**4)+f0PA0PN
        F2 = (self.F2_interp(p)/p**5)+f1PA0PN
        EFLx_prime = self.EFLx_prime_interp(p)/p**5

        return [F1, F2, EFLx_prime]

    def interpolate_flux_grav0pa(
              self, p: float
              ) -> float:     

        #if p < 6.25 or p > 60.: 
        #    raise ValueError("Interpolation: p scalar out of bounds.")

        f0PA0PN = - 64./(5*(p**3))
        F1 = (self.F1_interp(p)/p**4)+f0PA0PN
        
        return F1


    def interpolate_flux_scalar(
              self, p: float
              ) -> float:     

        #if p < 6.25 or p > 60.: 
        #    raise ValueError("Interpolation: p scalar out of bounds.")

        u_scal = 1./np.sqrt(p-0.9*6)
        pdot_scal = - (self.additional_args[0])*(self.pdot_interp_scal(u_scal))/4.

        return pdot_scal  


    def F2PN(self, r0):
        pi = np.pi
        EulerGamma = np.euler_gamma

        x = 1.0 / r0

        x2 = x * x
        x4 = x2 * x2
        x5 = x4 * x
        x6 = x5 * x
        x7 = x6 * x
        x8 = x7 * x
        x9 = x8 * x

        x15_2 = x7 * np.sqrt(x)
        x17_2 = x15_2 * x
        x19_2 = x17_2 * x

        log_r0 = -np.log(r0)
        log2 = np.log(2.0)
        log3 = np.log(3.0)

        term = (
            - (56.0 / 3.0) * x6
            + (37084.0 / 315.0) * x7
            + (
                -134543.0 / 1215.0
                + (82.0 / 15.0) * pi**2
            ) * x8
            + (
                -1452202403629.0 / 229209750.0
                + (1327296.0 / 1225.0) * EulerGamma
                - (267127.0 / 720.0) * pi**2
                + (15329984.0 / 11025.0) * log2
                + (37908.0 / 49.0) * log3
                + (663648.0 / 1225.0) * log_r0
            ) * x9
            - (2332.0 / 15.0) * pi * x15_2
            + (42949.0 / 54.0) * pi * x17_2
            + (
                2062241.0 / 3465.0
                + (328.0 / 15.0) * pi**2
            ) * pi * x19_2
            )

        return term


    #def DeltaEPN_prime(self, nu, r0):
    #    numerator = nu**2 * ( -852390 * np.pi**2 + 3888 * r0**2 + 120 * r0 * (1674 + 7 * nu) - 35 * (-640863 + 774 * nu + 11 * nu**2))
    #    denominator = 62208 * r0**4
    #    return numerator / denominator
   
    def EPN_prime(self, nu, r0):
        pi = np.pi
        x0 = 1.0/r0
        x0_2 = x0*x0
        x0_3 = x0_2*x0
        x0_4 = x0_2*x0_2
        EulerGamma = 0.57721566490153286060
        
        c1 = 103680.0 * (9.0 + nu)

        c2 = 77760.0  * ((nu - 57.0) * nu + 81)

        c3 = 2400.0*(
            ((7.0*nu+1674.0)*nu
                + 9 * (-6889.0 + 246.0 * pi**2) 
            )* nu
                +10935.0 
            )

        c4 = (
            (((-7700.0 * nu - 541800.0) * nu
            + (448604100.0 - 17047800.0 * pi**2)) * nu
            + (48202884.0
             - 185794560.0 * EulerGamma
             - 18299925.0 * pi**2
              - 92897280.0 * (np.log(16.0) - np.log(r0))
             )
                ) * nu
            + 96446700.0
            )
        
        numerator = (
            - 622080.0 
            + c1 * x0
            + c2 * x0_2
            + c3 * x0_3 
            + c4 * x0_4
        )    

        denominator = 1244160 

        return numerator / denominator


    def E1PN_prime(self, r0):
        pi = np.pi
        EulerGamma = 0.57721566490153286060
        
        x0 = 1.0 / r0
        x0_2 = x0 * x0
        x0_3 = x0_2 * x0
        x0_4 = x0_2 * x0_2

        log_term = np.log(16.0) - np.log(r0)

        c1 = 3840.0
        c2 = -(164160.0)
        c3 = -(5511200.0 - 196800.0 * pi**2)
        c4 = 1785292.0 - 6881280.0 * EulerGamma - 677775.0 * pi**2 - 3440640.0 * log_term

        expr = (
            c1 * x0
            + c2 * x0_2
            + c3 * x0_3
            + c4 * x0_4
            ) / 46080.0

        return expr    

    def F0HybridExtended(self, nu, r0):
        # NOTE ON NAMING: the local F1/F2 below are the *numerical grid* columns
        # (interpolate_flux_grids' raw script-flux data, i.e. the paper's calligraphic
        # F0^GSF/F1^GSF in Eq. 18), unrelated to this *method's* name F0HybridExtended or to
        # F1HybridExtended just below - three different things share similar names here.
        # DENOMINATOR EXPANSION (paper Eq. 18/30, hence the class name): a(r_hat) = term1 =
        # (dE0/dr_hat)^-1 for circular Schwarzschild. F0HybridExtended = num*a(r_hat) is the
        # O(nu^0) piece; the O(nu) correction to the denominator (term2, from dE1/dr_hat) is
        # deliberately NOT included here - it is instead folded into F1HybridExtended via
        # -F1_flux*term2/term1**2 (the first-order Taylor expansion of 1/(term1+nu*term2)),
        # so FHybridExtended = F0 + nu*F1 already carries it at the correct order.

        if 6.25 <= r0 <= 30.:
            F1, F2, EFLx_prime = self.interpolate_flux_grids(r0)
            num = - F1
            term1 = -( (1 - 6 / r0) / (3 * (1 - 3 / r0)**(3/2) * np.sqrt(1 / r0)) )
            term2 = nu * (2/3) * r0**(1/2) * EFLx_prime
            #term3 = (2/3) * r0**(1/2) * self.DeltaEPN_prime(nu, r0)
            return num/(term1)

        elif r0 > 30.:
            # (an analytic F1PN PN-extension was tried here but is unused/removed - superseded
            # by the extended numerical grid in interpolate_flux_grav0pa, valid up to p=50)
            F1 = self.interpolate_flux_grav0pa(r0) 
            num = - F1
            term1 = -( (1 - 6 / r0) / (3 * (1 - 3 / r0)**(3/2) * np.sqrt(1 / r0)) )
            term2 = nu * (2/3) * r0**(1/2) * self.E1PN_prime(r0)
            #term3 = (2/3) * r0**(1/2) * self.DeltaEPN_prime(nu, r0)
            return  num / (term1)

        else:
            raise ValueError("Interpolation: p out of bounds.")

    def F1HybridExtended(self, nu, r0):
        #x = 1 / r0 
    
        if 6.25 <= r0 <= 30.:
            F1, F2, EFLx_prime = self.interpolate_flux_grids(r0)
            num = - F2
            term1 = -( (1 - 6 / r0) / (3 * (1 - 3 / r0)**(3/2) * np.sqrt(1 / r0)) )
            term2 = nu * (2/3) * r0**(1/2) * EFLx_prime
            #term3 = (2/3) * r0**(1/2) * self.DeltaEPN_prime(nu, r0)
            return num/(term1) - F1*term2/(term1**2)  # denominator-expansion cross term, see note in F0HybridExtended above

        elif r0 > 30.:
            F1 = self.interpolate_flux_grav0pa(r0)
            num = - self.F2PN(r0)
            term1 = -( (1 - 6 / r0) / (3 * (1 - 3 / r0)**(3/2) * np.sqrt(1 / r0)) )
            term2 = nu * (2/3) * r0**(1/2) * self.E1PN_prime(r0)
            #term3 = (2/3) * r0**(1/2) * self.DeltaEPN_prime(nu, r0)
            return num/(term1) - F1*term2/(term1**2)  # denominator-expansion cross term, see note in F0HybridExtended above
            
        else:
            raise ValueError("Interpolation: p out of bounds.")

    def FHybridExtended(self,nu, r0):
        return  self.F0HybridExtended(nu, r0) + nu * self.F1HybridExtended(nu, r0)

    def evaluate_rhs(
        self, y: Union[list[float], np.ndarray]
    ) -> list[float]:
        if self.use_ELQ:
            E, L, Q = y[:3]
            p, e, x = ELQ_to_pex(self.a, E, L, Q)

        else:
            p, e, x = y[:3]
        #eps=mu/M
        nu = self.massratio #here eps = new_mu/new_M = \nu! 
        #nu = eps/((1+eps)**2) #if input M, m
        Omega_phi, Omega_theta, Omega_r = get_fundamental_frequencies(self.a, p, e, x)
        omegadot = self.F0HybridExtended(nu, p)
        domegadr = -(3/(2*p**(5/2)))
        pdot  =  omegadot/domegadr
        #pdot = pdot/(eps) #if input M, m
        pdot = pdot #if input new_M=m+M, new_m = mu*M/(mu+M)
        pdot_scal = self.interpolate_flux_scalar(p)
        return [pdot+pdot_scal, 0., 0., Omega_phi, Omega_theta, Omega_r]    




##################################################################
######## VACUUM GENERAL RELATIVITY ###############################
##################################################################
