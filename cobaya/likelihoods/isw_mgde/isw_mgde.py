"""
ISW likelihood for Modified Gravity and Dark Energy models.
Based on the MontePython ISW_MGDE_simpleBias likelihood.
"""

from cobaya.likelihood import Likelihood
import numpy as np
import os
from scipy import integrate, interpolate

from cobaya.log import LoggedError

def patch_asscalar(a): # changed by me, fixing the issue of np.asscalar being deprecated
    return a.item()

setattr(np, "asscalar", patch_asscalar)

class isw_mgde(Likelihood):
    """
    ISW likelihood for Modified Gravity and Dark Energy models.
    Uses cross-correlation between CMB and galaxy surveys.
    """

    path: str
    data_directory: str
    zmax: float
    l_min_cross: int
    l_max_cross: int
    n_bins_cross: int

    use_sdss: bool
    use_qso: bool
    use_mpz: bool
    use_wisc: bool
    use_nvss: bool

    theory_code: str

    sdss_bins: dict
    qso_bins: dict
    mpz_bins: dict
    wisc_bins: dict
    nvss_bins: dict
    
    # Optional: Installation options
    install_options = {
        "github_repository": "", # Add repository if needed
        "download_url": ""      # Add download URL if needed
    }
    
    # Type for categorization
    #type = ["CMB", "LSS"]  # Both CMB and LSS since it's cross-correlation

    def initialize(self):
        """Set up the likelihood."""
        # better than hasattr(self, "path") because path could be None. bool(None) evaluates to False
        if not getattr(self, "path", None):
            raise LoggedError(
                self.log, "No path given to ISW data. Set the likelihood property "
                          "'path'.")
        # If no path specified, use the external packages path

        if self.theory_code not in ["hi_class", "CLASS"]:
            raise LoggedError(self.log, f"ISW likelihood not implemented for the theory code {self.theory_code}. Choose one from {str(['hi_class','CLASS'])}.")

        # Read data files
        #self.data_directory = os.path.join(self.packages_path, "data")
        self.data_directory = os.path.normpath(getattr(self, "path", None) or
                                          os.path.join(self.packages_path, "isw_mgde/data"))
        self.datadir = os.path.join(self.path, self.data_directory)
        
        # # Initialize surveys and redshift bins
        # self.surveys = {
        #     'sdss': {'bins': 5, 'files': []},
        #     'qso': {'bins': 3, 'files': []},
        #     'mpz': {'bins': 3, 'files': []},
        #     'wisc': {'bins': 3, 'files': []},
        #     'nvss': {'bins': 1, 'files': []}
        # }
        
        # Load data files for each survey
        # for survey, info in self.surveys.items():
        #     for i in range(info['bins']):
        #         filepath = os.path.join(self.data_directory, f"{survey}_bin{i}.txt")
        #         info['files'].append(filepath)
                
        # Load window functions, covariance matrices, etc.
        # self._load_survey_data()

    # def _load_survey_data(self):
    #     """Load survey-specific data and window functions."""
    #     # This would load window functions, redshift distributions, etc.
    #     # Implementation specific to your data format
    #     pass

    def get_requirements(self):
        """
        Define theory requirements for the likelihood.
        """
        return {
            "CLASS_background": None,
            "Pk_interpolator": {
                "z": np.linspace(0, self.zmax, 100),
                "k_max": 2.0,
                "nonlinear": False,
                "vars_pairs": (
                    [("delta_nonu", "delta_nonu")]), # nonu for cdm_baryons is safer than total matter
            },
            "Cl": {"tt": 2500},  # CMB temperature power spectrum
            #"H": {"z_max": 5.1, "z_array": np.linspace(0, 5.1, 100)},
            #"Pk_interpolator": {"z_max": 5.1, "nonlinear": False},
            # "ns": None,
            "Omega_m": None,
            # "H0": None, # changed by me, printing H0 out for debugging
            # "h": None,
            # # # "H0",
            # "Omega_Lambda": None,
            # # "omegac": {"z": 0},
            # "omegabh2": None,
            # # # "Omega_m",
            # # # "Omega_k": None,
            # "rs_drag": None,
            # "tau_reio": None,
            # "z_reio": None,
            # "z_rec": None,
            # "tau_rec": None,
            # "m_ncdm_tot": None,
            # "Neff": None,
            # "YHe": None,
            # "age": None,
            # "conformal_age": None,
            # "sigma8": None,
            # "sigma8_cb": None,
            # "theta_s_100": None,
        }

    # def _compute_sigma(self):
    #     """Calculate sigma(z) for growth rate."""
    #     # z_arr = theory.get_z_array()
    #     # sigma = np.zeros_like(z_arr)
        
    #     # for i, z in enumerate(z_arr):
    #     #     # Get power spectrum at this redshift
    #     #     k = np.logspace(-4, 1, 1000)
    #     #     pk = theory.get_pk_interpolator()(k, z)
            
    #     #     # Calculate sigma
    #     #     integrand = pk * k**2
    #     #     sigma[i] = np.sqrt(np.trapz(integrand, k) / (2 * np.pi**2))
            
    #     return sigma

    def _compute_F(self, sigma, zr):
        """Calculate growth rate function F(z)."""

        #Pkinterp = self.provider.get_Pk_interpolator(nonlinear=False)
        Pkinterp = self.provider.get_Pk_interpolator(("delta_nonu", "delta_nonu"), nonlinear=False)
        D = np.array([np.sqrt(Pkinterp.P(z, 1.)/Pkinterp.P(0., 1.)) for z in zr])

        interp_log_D_cross_sigma=interpolate.UnivariateSpline(list(reversed(zr)),list(reversed(np.log(D*sigma))),k=3,s=0)
        prime_log_D_sigma=interp_log_D_cross_sigma.derivative(1)
        interpsigma=interpolate.UnivariateSpline(list(reversed(zr)),list(reversed(sigma)),k=3,s=0)

        return(prime_log_D_sigma,interpsigma)

        # z_arr = theory.get_z_array()
        # D = theory.get_growth_factor(z_arr)
        
        # # Calculate derivative of log(D) with respect to log(a)
        # log_a = -np.log(1 + z_arr)
        # dlogD_dloga = np.gradient(np.log(D), log_a)
        
        # # Create interpolator for F(z)
        # F_interp = interpolate.interp1d(z_arr, dlogD_dloga, 
        #                               bounds_error=False, 
        #                               fill_value="extrapolate")
        
        # return F_interp

    def _integrand_cross(self, z, l, F_z, H_of_z, D_of_z, dndz, norm):
        """Integrand for ISW-galaxy cross correlation."""
        H0 = self.provider.get_param("H0")
        Om = self.provider.get_param("Omega_m")
        c = 299792.458  # Speed of light in km/s, same as what the Montepython version uses from scipy.constants
        interpsigma = F_z[1]
        prime_log_D_sigma = F_z[0]

        #Pkinterp = self.provider.get_Pk_interpolator(nonlinear=False)
        Pkinterp = self.provider.get_Pk_interpolator(("delta_nonu", "delta_nonu"), nonlinear=False)

        k_of_z = lambda z: (l + 0.5)/(D_of_z(z)*(1 + z))

        # I have checked that both Pkinterp.P and cosmo.Pk in the Montepython version use 1/Mpc units, not h/Mpc
        return (3*Om*H0**2)/((c**2)*(l+0.5)**2)*dndz(z)*H_of_z(z)*interpsigma(z)*(prime_log_D_sigma(z)*(1+z)+1)*Pkinterp.P(z, k_of_z(z))*(1+z)/norm
        
        
        # # Angular diameter distance
        # da = theory.get_angular_diameter_distance(z)
        
        # # Scale factor and growth
        # a = 1 / (1 + z)
        # D = theory.get_growth_factor(z)
        
        # # Get k mode
        # k = (l + 0.5) / (da * (1 + z))
        
        # # Get power spectrum
        # pk = theory.get_pk_interpolator()(k, 0)
        
        # # Window function term
        # W = dndz(z) / self.norm
        
        # # ISW term
        # f = F_z(z)
        # isw_term = -(3 * H0**2 * Om / (c**2 * (l + 0.5)**2))
        
        # return (isw_term * theory.get_H(z) * 
        #         (da * (1 + z))**2 * D * W * 
        #         (f * (1 + z) + 1) * np.sqrt(pk))
    
    def _integrand_auto(self, z, l, H_of_z, D_of_z, dndz, norm):
        """Integrand for galaxy auto-correlation."""
        k_of_z = lambda z: (l + 0.5)/(D_of_z(z)*(1 + z))

        #Pkinterp = self.provider.get_Pk_interpolator(nonlinear=False)
        Pkinterp = self.provider.get_Pk_interpolator(("delta_nonu", "delta_nonu"), nonlinear=False)

        return (dndz(z))**2*Pkinterp.P(z, k_of_z(z))*H_of_z(z)*(1+z)**2/(D_of_z(z)*(1+z))**2/norm**2

    def logp(self, **params_values):
        """
        Compute log-likelihood for the ISW signal.
        """
        # theory = self.provider
        
        # # Get required functions from theory
        # sigma = self.get_sigma(theory)
        # zr = theory.get_z_array()
        
        # # Limit to z < 5.1
        # zmax = 5.1
        # zmax_ind = np.argmin(np.abs(zr - zmax))
        # sigma = sigma[zmax_ind:]
        # zr = zr[zmax_ind:]
        # F_z = self.F(sigma, theory)
        
        # # Initialize log-likelihood
        # loglkl = 0
        
        # Compute for each survey and bin
        # for survey, info in self.surveys.items():
        #     for bin_num in range(info['bins']):
        #         # Get bias parameter
        #         b = params_values[f'b_{survey}']
                
        #         # Load bin-specific data
        #         data = self._load_bin_data(survey, bin_num)
                
        #         # Compute theoretical Cl
        #         cl_theory = self._compute_cl(theory, b, sigma, F_z, zr, data)
                
        #         # Add to likelihood
        #         delta = data['cl_obs'] - cl_theory
        #         loglkl += -0.5 * delta.T @ np.linalg.solve(data['cov'], delta)

        #sigma = self._compute_sigma()

        zr = self.provider.get_CLASS_background()["z"]
        if self.theory_code == "hi_class":
            sigma = self.provider.get_CLASS_background()["Sigma_smg"]
        elif self.theory_code == "CLASS":
            sigma = np.ones(zr.shape)
        zmax = self.zmax
        zmax_ind = np.argmin(np.abs(zr - zmax))
        zr = zr[zmax_ind:]
        sigma = sigma[zmax_ind:]
        F_z = self._compute_F(sigma, zr)
        A = params_values.get("A_ISW", 1.0)

        Hr = self.provider.get_CLASS_background()["H [1/Mpc]"]
        Hr = Hr[zmax_ind:]
        H_of_z = interpolate.make_interp_spline(zr[::-1], Hr[::-1]) # [::-1] because make_interp_spline needs increasing x
        Dr = self.provider.get_CLASS_background()["ang.diam.dist."]
        Dr = Dr[zmax_ind:]
        D_of_z = interpolate.make_interp_spline(zr[::-1], Dr[::-1])

        loglkl = 0
        if self.use_sdss:
            #self.log.info(f"DEBUGGING. The bin dict prints as {self.sdss_bins}")
            loglkl += self.compute_survey_loglkl(self.sdss_bins, params_values["b_sdss"], F_z, H_of_z, D_of_z, A)
        
        if self.use_qso:
            loglkl += self.compute_survey_loglkl(self.qso_bins, params_values["b_qso"], F_z, H_of_z, D_of_z, A)
        
        if self.use_mpz:
            loglkl += self.compute_survey_loglkl(self.mpz_bins, params_values["b_mpz"], F_z, H_of_z, D_of_z, A)
        
        if self.use_wisc:
            loglkl += self.compute_survey_loglkl(self.wisc_bins, params_values["b_wisc"], F_z, H_of_z, D_of_z, A)
        
        if self.use_nvss:
            loglkl += self.compute_survey_loglkl(self.nvss_bins, params_values["b_nvss"], F_z, H_of_z, D_of_z, A)
        
        #self.log.info(f"DEBUGGING. The value of H0 is {self.provider.get_param('H0')}.")

        # names = [
        #     "ns",
        #     # "h",
        #     "H0",
        #     "Omega_Lambda",
        #     "Omega_cdm",
        #     "Omega_b",
        #     "omegabh2",
        #     "Omega_m",
        #     # "Omega_k",
        #     "rs_drag",
        #     "tau_reio",
        #     "z_reio",
        #     "z_rec",
        #     "tau_rec",
        #     "m_ncdm_tot",
        #     "Neff",
        #     "YHe",
        #     "age",
        #     "conformal_age",
        #     "sigma8",
        #     "sigma8_cb",
        #     "theta_s_100",
        # ]
        
        # for param in names:
        #     if param == "Omega_cdm":
        #         print(f"DEBUGGING. The value of {param} is {self.provider.get_CLASS_background()["(.)rho_cdm"][-1]/self.provider.get_CLASS_background()["(.)rho_crit"][-1]}.")
        #     elif param == "Omega_b":
        #         print(f"DEBUGGING. The value of {param} is {self.provider.get_CLASS_background()["(.)rho_b"][-1]/self.provider.get_CLASS_background()["(.)rho_crit"][-1]}.")
        #     else:
        #         print(f"DEBUGGING. The value of {param} is {self.provider.get_param(param)}.")
        
        return loglkl

    # def _compute_cl(self, theory, bias, sigma, F_z, zr, data):
    #     """Compute theoretical Cl for a given bin."""
    #     l_arr = data['ell']
    #     cl = np.zeros_like(l_arr)
        
    #     for i, l in enumerate(l_arr):
    #         integrand = lambda z: self.integrand_cross(z, l, sigma, F_z, 
    #                                                  theory, data['dndz'])
    #         cl[i] = bias * integrate.quad(integrand, data['z_min'], 
    #                                    data['z_max'])[0]
        
    #     # Bin the Cl values if necessary
    #     if 'bins' in data:
    #         cl = self._bin_cl(l_arr, cl, data['bins'])
        
    #     return cl
    
    def compute_survey_loglkl(self, bins, bias, F_z, H_of_z, D_of_z, A):
        """Compute log-likelihood for a survey given its bins."""
        survey_loglkl = 0
        for bin in bins:
            #self.log.info(f"DEBUGGING. The bin prints as {bin}")
            #survey_loglkl += self.compute_bin_loglkl(bin, bias, F_z, H_of_z, D_of_z, A)
            bin_loglkl = self.compute_bin_loglkl(bins[bin], bias, F_z, H_of_z, D_of_z, A)
            survey_loglkl += bin_loglkl
            #self.log.info(f"DEBUGGING. The loglkl for bin {bin} is {bin_loglkl}.")
        return survey_loglkl
    
    def compute_bin_loglkl(self, bin, b, F_z, H_of_z, D_of_z, A):

        l_cross,cl_cross=np.loadtxt(os.path.join(self.datadir,bin["cl_cross_file"]),unpack=True,usecols=(0,1))
        l_auto,cl_auto=np.loadtxt(os.path.join(self.datadir,bin["cl_auto_file"]),unpack=True,usecols=(0,1))
        cov_cross=np.loadtxt(os.path.join(self.datadir,bin["cov_cross_file"]))
        cov_auto=np.loadtxt(os.path.join(self.datadir,bin["cov_auto_file"]))

        l_cross = l_cross[self.l_min_cross:self.l_max_cross + 1]
        cl_cross = cl_cross[self.l_min_cross:self.l_max_cross + 1]
        l_auto = l_auto[bin["l_min_auto"]:bin["l_max_auto"] + 1]
        cl_auto = cl_auto[bin["l_min_auto"]:bin["l_max_auto" ] + 1]
        cov_cross=cov_cross[self.l_min_cross:self.l_max_cross+1,self.l_min_cross:self.l_max_cross+1]
        cov_auto=cov_auto[bin["l_min_auto"]:bin["l_max_auto"]+1,bin["l_min_auto"]:bin["l_max_auto"]+1]

        bins_cross=np.ceil(np.logspace(np.log10(self.l_min_cross),np.log10(self.l_max_cross),self.n_bins_cross+1))
        bins_auto=np.ceil(np.logspace(np.log10(bin["l_min_auto"]),np.log10(bin["l_max_auto"]),bin["n_bins_auto"]+1))

        l_binned_cross, cl_binned_cross, cov_binned_cross = self._bin_cl(l_cross, cl_cross, bins_cross, cov_cross)
        l_binned_auto, cl_binned_auto, cov_binned_auto = self._bin_cl(l_auto, cl_auto, bins_auto, cov_auto)

        zz,dndz=np.loadtxt(os.path.join(self.datadir,bin["dndz_file"]),unpack=True,usecols=(0,1))
        dndz=interpolate.interp1d(zz,dndz,kind='cubic')
        norm=integrate.quad(dndz,bin["z_min"],bin["z_max"])[0]

        #A = params_values.get("A_ISW", 1.0)
        #sigma = self._compute_sigma()
        #F_z = self._compute_F(sigma)

        cl_binned_cross_theory=np.array([(integrate.quad(self._integrand_cross,bin["z_min"],bin["z_max"],args=(bins_cross[ll], F_z, H_of_z, D_of_z, dndz, norm))[0]+integrate.quad(self._integrand_cross,bin["z_min"],bin["z_max"],args=(bins_cross[ll+1], F_z, H_of_z, D_of_z, dndz, norm))[0]+integrate.quad(self._integrand_cross,bin["z_min"],bin["z_max"],args=(l_binned_cross[ll], F_z, H_of_z, D_of_z, dndz, norm))[0])/3 for ll in range(self.n_bins_cross)])
        #cl_binned_cross_theory=np.array([(integrate.quad(self._integrand_cross,bin["z_min"],bin["z_max"],args=(bins_cross[ll], F_z, H_of_z, D_of_z, dndz, norm), epsabs=0, epsrel=1e-5)[0]+integrate.quad(self._integrand_cross,bin["z_min"],bin["z_max"],args=(bins_cross[ll+1], F_z, H_of_z, D_of_z, dndz, norm), epsabs=0, epsrel=1e-5)[0]+integrate.quad(self._integrand_cross,bin["z_min"],bin["z_max"],args=(l_binned_cross[ll], F_z, H_of_z, D_of_z, dndz, norm), epsabs=0, epsrel=1e-5)[0])/3 for ll in range(self.n_bins_cross)])

        cl_binned_auto_theory=np.array([integrate.quad(self._integrand_auto,bin["z_min"],bin["z_max"],args=(ll, H_of_z, D_of_z, dndz, norm),epsrel=1e-8)[0] for ll in l_binned_auto])
        #cl_binned_auto_theory=np.array([integrate.quad(self._integrand_auto,bin["z_min"],bin["z_max"],args=(ll, H_of_z, D_of_z, dndz, norm),epsabs=0, epsrel=1e-5)[0] for ll in l_binned_auto])

        chi2_cross=np.asscalar(np.dot(cl_binned_cross-A*b*cl_binned_cross_theory,np.dot(np.linalg.inv(cov_binned_cross),cl_binned_cross-A*b*cl_binned_cross_theory)))
        chi2_auto=np.asscalar(np.dot(cl_binned_auto-b**2*cl_binned_auto_theory,np.dot(np.linalg.inv(cov_binned_auto),cl_binned_auto-b**2*cl_binned_auto_theory)))

        #self.log.info(f"DEBUGGING. loglkl_cross for this bin is {-0.5*chi2_cross}.")
        #self.log.info(f"DEBUGGING. loglkl_auto for this bin is {-0.5*chi2_auto}.")
        #self.log.info(f"DEBUGGING. cl_binned_cross_theory for this bin is {[f'{cl:.2e}' for cl in cl_binned_cross_theory]}.")
        #self.log.info(f"DEBUGGING. cl_binned_auto_theory for this bin is {[f'{cl:.2e}' for cl in cl_binned_auto_theory]}.")
        #self.log.info(f"DEBUGGING. The value of bias for this bin is {b}.")

        bin_loglkl=-0.5*(chi2_cross+chi2_auto)

        return bin_loglkl

    def _bin_cl(self, l, cl, bins, cov=None):
        # This function bins l,C_l, and the covariance matrix in given bins in l
        B=[]
        for i in range(1,len(bins)):
            if i!=len(bins)-1:
                a=np.where((l<bins[i])&(l>=bins[i-1]))[0]
            else:
                a=np.where((l<=bins[i])&(l>=bins[i-1]))[0]
            c=np.zeros(len(l))
            c[a]=1./len(a)
            B.append(c)
        l_binned=np.dot(B,l)
        cl_binned=np.dot(B,cl)
        if cov is not None:
            cov_binned=np.dot(B,np.dot(cov,np.transpose(B)))
            return l_binned,cl_binned,cov_binned
        else:
            return l_binned,cl_binned