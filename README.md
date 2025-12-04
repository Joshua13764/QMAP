# Adding a new module checklist

## 1) Create file directory in packages
```
...\AO33\packages\bennu-feature-extractor-{package name}
```

## 2) Add to workspace
In AO33.code-workspace add the path to your package
```
{
  "folders": [
    { "path": "packages/bennu-feature-extractor" },
    { "path": "packages/bennu-feature-extractor-PDS" },
    { "path": "packages/bennu-feature-extractor-BoulderNet" },
    { "path": "packages/bennu-feature-extractor-{package name}" },
    { "path": "integration-tests" }
  ]
}
```

## 3) Setup directory

The directory needs to have the following basic structure

```
.
├── pyproject.toml
├── pytest.ini
├── poetry.lock
├── README.md
├── requirements.txt
├── src/
│   └── Boulder_Statistics_{package name}/
│       ├── __init__.py
│       ├── ... package files
├── tests/
│   ├── __init__.py
│   └── ... tests
```
### ```pyproject.toml```

An example of which has been listed below, it should look broadly like this
```
[build-system]
requires = ["poetry-core>=1.8.0"]
build-backend = "poetry.core.masonry.api"

[project]
name = "bennu-feature-extractor-{package name}"
version = "0.1.0"
description = "Foo add-on step for bennu-feature-extractor"
readme = "README.md"
requires-python = ">=3.11,<4.0"

dependencies = [
  "prefect>=2.14",
  "assertpy (==1.1)",
  "pytest (==8.4.2)",
  "requests (==2.32.5)"
   ... other dependencies (it is better to add using poetry add {package name} in most cases)
]

[tool.poetry]
packages = [{ include = "Boulder_Statistics_{package name}", from = "src" }]

[tool.poetry.dependencies]
bennu-feature-extractor = { path = "../bennu-feature-extractor", develop = true }
... other packages in this repo

[tool.poetry.group.dev.dependencies]
pytest = "^8.0"
pytest-cov = "^4.1"
assertpy = "^1.1"
```

### ```pytest.ini```
Tends to always look the same unless doing something specific
```
[pytest]
addopts = -ra
testpaths = tests
python_files = test_*.py
```

### ```README.md```
The readme for the package (is required, so make sure it's included even if left blank)

### ```requirements.txt```
Can be generated in most cases automatically from the following commands (run these in the package directory when ready to publish)
```
pip install pigar
pigar generate -f requirements.txt
Get-Content requirements.txt | ForEach-Object { poetry add $_ }
```

### ```poetry.lock```
This is generated when the environment has been set up

### ```__init__.py```
These need to be in every folder with Python files; otherwise, those files will not be recognised as part of the package (it is ok to leave them blank).

## Setup poetry (add poetry.lock)
```
cd packages\bennu-feature-extractor-{package name}
poetry env list --full-path
poetry env remove --all
poetry config virtualenvs.in-project true
if (Test-Path .\python) { Rename-Item .\python _python_OLD }
py -3.13 --version
poetry env use 3.13
poetry lock
poetry install -v
```
*run these in the package directory


# V1.1 (block 1) project structure

![Flowchart](images/v1.1_images/flowchart.png)

## Enviroment method

### File validation

In the base class we add currently two methods of file validation
1. Metadata checks (last modified, name, file size ... etc) -> Fast but not perfect
2. Hash checks (hashes the entire file) -> Slow but much stronger

# AO33 – Unsupervised Boulder Mapping of Asteroid (101955) Bennu

## Project Overview  
In 2018, NASA’s **OSIRIS-REx** mission arrived at near-Earth Asteroid **Bennu** and carried out a detailed survey before collecting a sample in October 2020, finally returning it to Earth in September 2023.  

During the survey, the mission collected:  
- Surface imagery  
- Detailed 3D topography using LIDAR  
- Visible, near, and thermal infrared spectra  

The **Planetary Experiments Group in Oxford Physics** has been part of the mission team throughout. The group is now working to connect infrared spectroscopic measurements of the returned sample (measured in the lab) to the global measurements of Bennu made by the spacecraft.  

This project focuses on characterising the surface topography of Bennu to help us understand its infrared spectra and returned samples.  

---

## Project Goals  
The student will develop a **machine-learning pipeline** that:  
1. Automatically segments individual boulders using remote sensing data from the asteroid.  
2. Clusters them into **textural families**.  
3. Extracts their **size-frequency distribution (SFD)**.  

---

## Specific Objectives  
1. **Produce the first automated Bennu boulder catalogue** (≥100,000 objects) including:  
   - Diameters  
   - Shapes  
   - Albedo/colour indices  
   - Local roughness metrics  

2. **Quantify how SFD and block packing vary** between the unsupervised boulder classes, testing hypotheses about porosity and strength suggested by *Rozitis et al. 2022*.  

3. **Correlate class-specific SFD parameters** (cumulative slope, maximum block size) with thermal inertia and spectral slope to determine whether mechanical breakdown pathways drive observed variability.  

---

## Skills & Background  
This project would suit a student with:  
- Programming experience (Python or MATLAB preferred)  
- Interest in **Solar System / planetary science**  
- Interest in **autonomous classification systems** (e.g., neural networks or other machine learning techniques)  

---

## Background Reading  

### OSIRIS-REx Mission  
- [Background information on OSIRIS-REx](https://link.springer.com/article/10.1007/s11214-017-0405-1)  

### Sample Sites  
- [Emery paper on sample sites](https://www.sciencedirect.com/science/article/pii/S0019103514000827?casa_token=TMH2QNtCpY8AAAAA:kZMozTSPqCTd_o7mZ-B6i6h8GAXgPFMzAZE5o-wSoGB9-lms2D8FrKWZpnrk9hGo9qostnSunA)  

### Digital Mapping Approaches  
- [Current digital mapping approaches #1](https://www.sciencedirect.com/science/article/pii/S0032063318303805?casa_token=o2i3SQLUJmIAAAAA:qzAnL-RhVa_Yz8MFSsqccVYna1QTJOxJdpRgpvvQekRU8bvYJukhwDGwl76yCI7YJutkJ-4beA)  
- [Current digital mapping approaches #2](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2018EA000382)  

### Boulder Properties  
- [Boulder properties](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2023JE008019)  

### Other Planetary Boulder Mapping Efforts  
- [Comparative planetary boulder mapping](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2023JE008013)  

### Citations
For all of the download likes the data source will need to be cited
