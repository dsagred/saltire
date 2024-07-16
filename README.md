# *saltire* 



### <img src="saltire_icon.jpg"  width="40" height="35"> A High-resolution Cross-correlation Model

*saltire* is a simplistic model, designed to fit cross-correlation maps from high resolution cross-correlation spectroscopy (HRCCS), typically derived to detect atomic or molecular species in exoplanet atmospheres. The name *saltire* is inspired by the Saint Andrew's Cross, due to the cross-like structure of typical correlation signals.

This model can also be used to derive dynamical masses of high-contrast binaries [Sebastian et al. 2024](https://ui.adsabs.harvard.edu/abs/2024arXiv240206449S/abstract)
Use cases for *saltire* are:

- Fit CCF maps to estimate the companions semi amplitude and systems restframe.
- Predict the expected 'shape' of CCF maps depending from the planets orbit and observed phases.

**Installation**:
Download the whole archive using 'git clone https://github.com/dsagred/saltire.git'.

Dependencies:
- numpy, astropy, 
- For fitting: lmfit, emcee, multiprocessing

**Desciption and tutorials**:
Please refer to the [saltire paper](https://ui.adsabs.harvard.edu/abs/2024MNRAS.527.10921/abstract) for Details. The two tutorial notebooks present practial examples on how to simulate CCF maps and to measure parameters from observed CCF maps. 

**Development**:
- The current version supports weighting of observations i.e. to model a noise driven weighting usually applied during CCF map creation. This can also be used to model observations without or with minimised signal. 
- An option to add a time dependent CCF contrast will be added in future.

**License**:

The software is freely available. If you use it for your research, please cite [Sebastian et al. 2023](https://ui.adsabs.harvard.edu/abs/2024MNRAS.527.10921/abstract). Feedback and contributions are very welcome.
