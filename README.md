# MagGen

A graph-aided conditional generative model, based on variational autoencoder architecture, used to inverse design permanent magnet candiadtes.


## Overview

MagGen implements three imporatant features: 

1. **IRCR** : It uses invertible real space crystallographic representaion (IRCR) to encode raw crystal structure data.

2. **Conditional VAE** : It is based on a variational autoencoder (VAE) model, conditioned simultaneously on two target properties.

3. **Local Perturbation** : It uses Lp scheme to generate new materials by perturbing a parent material.

  
<img src="images/img1.png" alt="MagGen Schematic" width="550">

---

## New Material Generation

Schematic diagram illustrating the generation of new materials via local perturbation across a parent material, based on different scale factors for the perturbation.

<img src="images/scale-factor.png" alt="New material generation" width="550">

---


## License

This project is licensed under the **MIT License**.

See the [LICENSE](LICENSE) file for details.

Developed by: [Sourav Mal](https://github.com/SouravMal) at Harish-Chandra Research Institute (HRI) (https://www.hri.res.in/), Prayagraj, India.


## Citation

Please consider citing our work if you find it helpful:

```bibtex
@Article{Mal2024,
author={Mal, Sourav
and Seal, Gaurav
and Sen, Prasenjit},
title={MagGen: A Graph-Aided Deep Generative Model for Inverse Design of Permanent Magnets},
journal={The Journal of Physical Chemistry Letters},
year={2024},
month={Mar},
day={28},
publisher={American Chemical Society},
volume={15},
number={12},
pages={3221-3228},
doi={10.1021/acs.jpclett.4c00068},
url={https://doi.org/10.1021/acs.jpclett.4c00068}
}
```


## Contact

If you have any questions, feel free to reach us at:
**Sourav Mal** <souravmal492@gmail.com> 
