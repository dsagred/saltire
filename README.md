# *saltire* 



### <img src="saltire_icon.jpg"  width="40" height="35"> A High-resolution Cross-correlation Model

*saltire* is a simplistic model, designed to fit cross-correlation maps from high resolution cross-correlation spectroscopy (HRCCS), typically derived to detect atomic or molecular species in exoplanet atmospheres. The name *saltire* is inspired by the Saint Andrew's Cross, due to the cross-like structure of typical correlation signals.

This model can also be used to derive dynamical masses of high-contrast binaries (Sebastian et al. 2023b, in prep.).

Use cases for *saltire* are:

- Fit CCF maps to estimate the companions semi amplitude and systems restframe.
- Predict the expected 'shape' of retrieval maps depending from the planets orbit and observed phases.

**Installation**:
Download the whole archive using git clone.


**Development**:
- The current version supports weighting of observations i.e. to model a noise driven weighting usually applied during CCF map creation. This can also be used to model observations without or with minimised signal. 
- A physical phase curve model for inferior conjunction observations will be added in future.

**License**:

The software is freely available. If you use it for your research, please cite Sebastian et al. (2023, MNRAS, submitted). Feedback and contributions are very welcome.
