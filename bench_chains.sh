set +e

for CHAINS in 1 10 100 1000 10000 100000 1000000
do
	hyperfine 'python numpy_sampler.py --samples 1000 --chains '$CHAINS
	hyperfine 'python jax_sampler.py --samples 1000 --chains '"$CHAINS"' --compile_only True'
	python jax_sampler.py --samples 1000 --chains $CHAINS
	hyperfine 'python tfp_sampler.py --samples 1000 --chains '$CHAINS
	hyperfine 'python tfp_sampler.py --samples 1000 --chains '"$CHAINS"' --xla True'
done
