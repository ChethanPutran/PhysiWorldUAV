# 1. On the server — start the container
docker start -ai isaac-sim

# 2. On your laptop — tunnel port 6080
ssh -L 6080:localhost:6080 chetan@10.24.36.121

# 3. In your browser
#    http://localhost:6080/vnc.html  →  Connect


cd /workspace/PhysiWorldUAV/OmniDrones/scripts
python collect_data.py
# or with overrides:
python collect_data.py wind.type=step wind.step.magnitude=8.0 collection.num_trajectories=5


# Constant 2 N wind in the +X direction
python collect_data_wind_hover.py wind.type=constant \
  wind.constant.magnitude=2.0 \
  wind.constant.direction='[1.0,0.0,0.0]'

# Sudden 4 N disturbance after 5 seconds
python collect_data_wind_hover.py wind.type=step \
  wind.step.magnitude=4.0 \
  wind.step.step_time=5.0

# Oscillating wind
python collect_data_wind_hover.py wind.type=sinusoidal \
  wind.sinusoidal.amplitude=2.0 \
  wind.sinusoidal.frequency_hz=0.2

# Random, correlated turbulence approximation
python collect_data_wind_hover.py wind.type=dryden \
  wind.dryden.sigma=0.8