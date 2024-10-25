python src/main.py env=ma_atari common.devices=1 world_model_env.diffusion_sampler.num_steps_denoising=4
# CUDA_VISIBLE_DEVICES=0 python src/main.py env=atari common.devices=0 collection.train.num_envs=1

# python src/play.py --record --store-denoising-trajectory
