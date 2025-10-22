from bennu_feature_extractor_BoulderNet.docker.BoulderNetCPU.docker_helpers import run_script

script = r"C:\\Users\\Joshu\\Documents\\AO33\\packages\\bennu-feature-extractor-BoulderNet\\src\\bennu_feature_extractor_BoulderNet\\docker\\BoulderNetCPU\\examples\\minimal_demo.py"
code, out = run_script("mltools:py3.10", script)
print("Exit:", code)
print(out)