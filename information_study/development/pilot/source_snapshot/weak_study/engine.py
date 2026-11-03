"""Space-time weak blocks on a periodic observed grid.

Spatial tests are Fourier modes, temporal tests are compact polynomials. All
spatial and time derivatives act on known test functions, never on observations.
The interface has no reference equation, true noise scale, or future field.
"""
from __future__ import annotations
import numpy as np
from scipy.integrate import newton_cotes
from pde_discovery import generic_engine as baseline
from l1_study import engine as sparse

VARIANTS = {
    'compact': {'time_fraction': .25, 'mode_fraction': .125},
    'low': {'time_fraction': .25, 'mode_fraction': .03125},
    'resolved': {'time_fraction': .25, 'mode_fraction': .25},
}
PROTOCOL = {'version':1, 'variants':VARIANTS, 'terms':list(sparse.TERM_NAMES),
    'spatial_tests':'nonnegative Fourier modes; real/imaginary rows, nonzero modes sqrt(2) weighted',
    'temporal_test':'(1-z*z)^4 on [-1,1], zero outside; analytic first derivative',
    'products':'2N zero-padded real trigonometric interpolation; retain selected observed-grid Fourier modes',
    'noise_correction':'none', 'no_observed_derivatives':True,
    'spatial_mode_rule':'min(Nx//4, max(8, floor(Nx*variant_fraction)))',
    'quadrature':'composite degree-nine Newton-Cotes on original samples, eight subintervals per panel',
    'window_stride':'max(1, window_intervals//4)',
    'selection':'unchanged L1 path, normalized OLS debias and development forecast selection',
    'reference_used_for_selection':False}


def _spectra(u):
    """Return Fourier coefficients of u, u² and u³ with 2N de-aliasing.

    The real Nyquist coefficient is split between +/-N/2 in the interpolant.
    """
    n=u.shape[1]
    spectrum=np.fft.fft(u,axis=1)/n
    padded=np.zeros((len(u),2*n),complex)
    positive=(n+1)//2
    padded[:,:positive]=spectrum[:,:positive]
    negative=(n-1)//2
    if negative:padded[:,-negative:]=spectrum[:,-negative:]
    if n%2==0:
        padded[:,n//2]=spectrum[:,n//2]/2
        padded[:,-n//2]=spectrum[:,n//2]/2
    interpolated=np.fft.ifft(padded*(2*n),axis=1).real
    return spectrum,np.fft.fft(interpolated**2,axis=1)/(2*n),np.fft.fft(interpolated**3,axis=1)/(2*n)


def weak_block(data,variant='compact'):
    if variant not in VARIANTS:raise ValueError('Unknown weak-block variant.')
    if not isinstance(data,dict) or not all(k in data for k in ('x','t','u')):
        raise ValueError('Need x,t,u observations.')
    if any(np.iscomplexobj(np.asarray(data.get(key))) for key in ('x','t','u')):
        raise ValueError('Weak observations must be real-valued.')
    x,t,u=baseline._validate(data)
    spec=VARIANTS[variant];n=len(x)
    width=max(8,8*int(round((len(t)-1)*spec['time_fraction']/8)))
    width=min(width,8*((len(t)-1)//8))
    starts=np.arange(0,len(t)-width,max(1,width//4))
    max_mode=min(n//4,max(8,int(n*spec['mode_fraction'])))
    modes=np.arange(0,max_mode+1)
    wave=2*np.pi*modes/(n*(x[1]-x[0]))
    uh,square,cube=_spectra(u)
    uh,square,cube=uh[:,modes],square[:,modes],cube[:,modes]
    constant=np.zeros_like(uh);constant[:,0]=1
    library=np.stack([constant,uh,square,cube,*[(1j*wave)**order*uh for order in range(1,5)],.5j*wave*square],axis=-1)
    features=[];targets=[]
    for start in starts:
        local_t=t[start:start+width+1]
        half=(local_t[-1]-local_t[0])/2
        z=(local_t-(local_t[-1]+local_t[0])/2)/half
        phi=(1-z*z)**4
        derivative=-8*z*(1-z*z)**3/half
        weights=np.zeros(width+1)
        for offset in range(0,width,8):
            weights[offset:offset+9]+=newton_cotes(8,equal=1)[0]*(local_t[1]-local_t[0])
        norm=float(weights@phi)
        X=np.einsum('t,tmj->mj',weights*phi,library[start:start+width+1])/norm
        y=-np.einsum('t,tm->m',weights*derivative,uh[start:start+width+1])/norm
        features.extend([X[0:1].real,np.sqrt(2)*X[1:].real,np.sqrt(2)*X[1:].imag])
        targets.extend([y[0:1].real,np.sqrt(2)*y[1:].real,np.sqrt(2)*y[1:].imag])
    block={'X':np.vstack(features),'y':np.concatenate(targets),'metadata':{
        'derivative_mode':'space_time_weak_fourier','variant':variant,'window_intervals':width,
        'integration_windows':len(starts),'max_spatial_mode':int(modes[-1]),
        'time_test_degree':8,'no_observed_derivatives':True,'training_frames':len(t),'spatial_points':n,
        'overlapping_windows':True,'noise_correction':'none','reference_used':False}}
    sparse.validate_block(block)
    return block
