for SAMPLES in 1000 10000 100000
do
	hyperfine 'python numpy_sampler.py --samples '"$SAMPLES"' --chains 1000'
	hyperfine 'python jax_sampler.py --samples '"$SAMPLES"' --chains 1000 --precompile True'
	# hyperfine 'python tfp_sampler.py --samples '"$SAMPLES"' --chains 1000 --xla True'
	# hyperfine 'python tfp_sampler.py --samples '"$SAMPLES"' --chains 1000'
	hyperfine 'python pytorch_sampler.py --samples '"$SAMPLES"' --chains 1000'
	# hyperfine 'python pytorch_sampler_gpu.py --samples '"$SAMPLES"' --chains 1000'
done
