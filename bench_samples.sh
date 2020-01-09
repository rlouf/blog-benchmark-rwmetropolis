for SAMPLES in 1 10 100 1000 10000 100000
do
	hyperfine 'python numpy_sampler.py --samples '"$SAMPLES"' --chains 1000'
	python jax_sampler.py --samples $SAMPLES --chains 1000
	hyperfine 'python jax_sampler.py --samples '"$SAMPLES"' --chains 1000 --compile_only True'
	hyperfine 'python tfp_sampler.py --samples '"$SAMPLES"' --chains 1000 --xla True'
	hyperfine 'python tfp_sampler.py --samples '"$SAMPLES"' --chains 1000'
done
