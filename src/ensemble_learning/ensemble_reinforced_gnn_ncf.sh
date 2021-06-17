rm -rf ../mains/ensemble/main_reinforced_gnn_ncf

python3 ../mains/main_reinforced_gnn_ncf.py --leonhard --ensemble-learning --random-seed 7
python3 ../mains/main_reinforced_gnn_ncf.py --leonhard --ensemble-learning --random-seed 27
python3 ../mains/main_reinforced_gnn_ncf.py --leonhard --ensemble-learning --random-seed 310
python3 ../mains/main_reinforced_gnn_ncf.py --leonhard --ensemble-learning --random-seed 128
python3 ../mains/main_reinforced_gnn_ncf.py --leonhard --ensemble-learning --random-seed 841
python3 ../mains/main_reinforced_gnn_ncf.py --leonhard --ensemble-learning --random-seed 99

python3 get_mean_predictions.py --ensemble-directory ../mains/ensemble/reinforced_gnn_ncf