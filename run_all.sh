# 1. Sim1, 20dB
# SUnSAL
python unmixing.py hsi=sim1 noise=20dB model=SUnSAL model.lambd=0.7 noise.seed=0,1,2,3,4 --multirun
# CLSUnSAL
python unmixing.py hsi=sim1 noise=20dB model=CLSUnSAL model.lambd=0.7 noise.seed=0,1,2,3,4 --multirun
# S2WSU
python unmixing.py hsi=sim1 noise=20dB model=S2WSU model.lambd=0.1 noise.seed=0,1,2,3,4 --multirun
# MUA_SLIC
python unmixing.py hsi=sim1 noise=20dB model=MUA_SLIC model.lambda1=0.03 model.lambda2=0.1 model.beta=30 model.slic_size=6 model.slic_reg=0.005 noise.seed=0,1,2,3,4 --multirun
# SUnCNN
python unmixing.py hsi=sim1 noise=20dB model=SUnCNN SVD_project=True model.n_iters=4000 noise.seed=0,1,2,3,4 --multirun
# SUnAA
python unmixing.py hsi=sim1 noise=20dB model=SUnAA model.T=500 noise.seed=0,1,2,3,4 --multirun
# FaSUn
python unmixing.py hsi=sim1 noise=20dB model=FaSUn model.T=10000 model.TA=5 model.TB=5 model.mu1=50.0 model.mu2=2.0 model.mu3=1.0 noise.seed=0,1,2,3,4 --multirun
# SUnS
python unmixing.py hsi=sim1 noise=20dB model=SUnS model.T=10000 model.TA=5 model.TB=5 model.mu1=50.0 model.mu2=2.0 model.mu3=1.0 model.hard=False model.lambd=0.01 noise.seed=0,1,2,3,4 --multirun
# MiSiSUn
python unmixing.py hsi=sim1 noise=20dB model=MiSiSUn model.T=10000 model.TA=5 model.TB=5 model.mu1=50.0 model.mu2=2.0 model.mu3=1.0 model.lamb=0.001 noise.seed=0,1,2,3,4 --multirun


# 2. Sim1, 30dB
# SUnSAL
python unmixing.py hsi=sim1 noise=30dB model=SUnSAL model.lambd=0.1 noise.seed=0,1,2,3,4 --multirun
# CLSUnSAL
python unmixing.py hsi=sim1 noise=30dB model=CLSUnSAL model.lambd=0.1 noise.seed=0,1,2,3,4 --multirun
# S2WSU
python unmixing.py hsi=sim1 noise=30dB model=S2WSU model.lambd=0.005 noise.seed=0,1,2,3,4 --multirun
# MUA_SLIC
python unmixing.py hsi=sim1 noise=30dB model=MUA_SLIC model.lambda1=0.007 model.lambda2=0.05 model.beta=10 model.slic_size=5 model.slic_reg=0.005 noise.seed=0,1,2,3,4 --multirun
# SUnCNN
python unmixing.py hsi=sim1 noise=30dB model=SUnCNN SVD_project=True model.n_iters=8000 noise.seed=0,1,2,3,4 --multirun
# SUnAA
python unmixing.py hsi=sim1 noise=30dB model=SUnAA model.T=500 noise.seed=0,1,2,3,4 --multirun
# FaSUn
python unmixing.py hsi=sim1 noise=30dB model=FaSUn model.T=10000 model.TA=5 model.TB=5 model.mu1=50.0 model.mu2=2.0 model.mu3=1.0 noise.seed=0,1,2,3,4 --multirun
# SUnS
python unmixing.py hsi=sim1 noise=30dB model=SUnShrink model.T=10000 model.TA=5 model.TB=5 model.mu1=50.0 model.mu2=2.0 model.mu3=1.0 model.hard=False model.lambd=0.01 noise.seed=0,1,2,3,4 --multirun
# MiSiSUn
python unmixing.py hsi=sim1 noise=30dB model=MiSiSUn model.T=10000 model.TA=5 model.TB=5 model.mu1=50.0 model.mu2=2.0 model.mu3=1.0 model.lamb=0.001 noise.seed=0,1,2,3,4 --multirun


# 3. Sim1, 40dB
# SUnSAL
python unmixing.py hsi=sim1 noise=40dB model=SUnSAL model.lambd=0.01 noise.seed=0,1,2,3,4 --multirun
# CLSUnSAL
python unmixing.py hsi=sim1 noise=40dB model=CLSUnSAL model.lambd=0.01 noise.seed=0,1,2,3,4 --multirun
# S2WSU
python unmixing.py hsi=sim1 noise=40dB model=S2WSU model.lambd=0.001 noise.seed=0,1,2,3,4 --multirun
# MUA_SLIC
python unmixing.py hsi=sim1 noise=40dB model=MUA_SLIC model.lambda1=0.001 model.lambda2=0.01 model.beta=10 model.slic_size=5 model.slic_reg=0.01 noise.seed=0,1,2,3,4 --multirun
# SUnCNN
python unmixing.py hsi=sim1 noise=40dB model=SUnCNN SVD_project=True model.n_iters=16000 noise.seed=0,1,2,3,4 --multirun
# SUnAA
python unmixing.py hsi=sim1 noise=40dB model=SUnAA model.T=500 noise.seed=0,1,2,3,4 --multirun
# FaSUn
python unmixing.py hsi=sim1 noise=40dB model=FaSUn model.T=10000 model.TA=5 model.TB=5 model.mu1=50.0 model.mu2=2.0 model.mu3=1.0 noise.seed=0,1,2,3,4 --multirun
# SUnS
python unmixing.py hsi=sim1 noise=40dB model=SUnShrink model.T=10000 model.TA=5 model.TB=5 model.mu1=50.0 model.mu2=2.0 model.mu3=1.0 model.hard=False model.lambd=0.01 noise.seed=0,1,2,3,4 --multirun
# MiSiSUn
python unmixing.py hsi=sim1 noise=40dB model=MiSiSUn model.T=10000 model.TA=5 model.TB=5 model.mu1=50.0 model.mu2=2.0 model.mu3=1.0 model.lamb=0.001 noise.seed=0,1,2,3,4 --multirun
