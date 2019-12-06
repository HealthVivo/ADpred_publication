
for i in {1..10}; do 
	#for j in 0,10 11,20 21,30 31,40 41,48; do
	for j in 0,5 6,10 11,15 16,20 21,25 26,30 31,35 36,40 41,45 46,48; do

		out_sufix=$(echo ${j} | tr "," "-")
		#echo "sbatch -p largenode --mem=21500 -c 6 --gres=gpu --wrap=\"python optimization.py -f aa,ss --gpu\""
		#echo "sbatch -p largenode --mem=21500 -c 6 --gres=gpu --wrap=\"python optimization.py -f aa,dis --gpu\""
		#echo "sbatch -p largenode --mem=21500 -c 6 --gres=gpu --wrap=\"python optimization.py -f aa,ss,dis --gpu\""
		#echo "sbatch -p largenode --mem=21500 -c 6 --gres=gpu --wrap=\"python optimization.py -f aa --gpu\""

	done
done
