## Project: WE RF Pulse Optimization Framework (“HydrOptiFrame”)

This repository contains code developed to implement the flexible framework for designing and optimizing water-excitation (WE) radio-frequency (RF) pulses, as described in Sieber et al. (2025). 

## Overview

- The goal is to design efficient WE RF pulses that suprees fat, are robust to B0 inhomogeneity, and can be applied at multiple field strengths (e.g., 1.5 T, 3 T). 
- The core of the framework uses B-spline interpolation to parameterize the RF waveform (“BSIO” pulses: B-spline interpolated optimized pulses). 
- An evolutionary optimization algorithm (e.g., genetic algorithm or differential evolution) is used to minimize a composite loss function based on Bloch‐equation simulations of fat/water contrast. 
- The resulting pulses were validated in phantom / volunteer studies in knee (3 T) and free-running cardiovascular imaging (1.5 T) to demonstrate improved fat suppression / contrast / vessel length.

## Testing 

To test the script change the parameters of your pulse and run main.py. It will start to optimize a pulse and display the best pulse found at the end. You can also use the UI with streamlit to get a browser tab that allows you to run optimizations and see the results


## Citation

If you use the code or framework in your publication, please cite the original paper:
Sieber X., Romanin L., Bastiaansen J.A.M., Roy C.W., Yerly J., Wenz D., Richiardi J., Stuber M., van Heeswijk R.B. A flexible framework for the design and optimization of water-excitation RF pulses using B-spline interpolation. Magn Reson Med. 2025;93(5):1896-1910.

## License

This project is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)**.

You may share and adapt the material for non-commercial purposes, with appropriate attribution.

See: https://creativecommons.org/licenses/by-nc/4.0/
