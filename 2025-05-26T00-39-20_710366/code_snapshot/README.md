# bachelor-thesis

Windows: 
$env:PYTHONHASHSEED = "0"; python main.py --agent dqn --lr 0.0005 --gamma 0.99 --batch_size 512 --mem_size 25000 --layers 17 17 17 --max_steps 100000 --replace 50 --save_interval 10 --render_freq 0

Macbook: 
PYTHONHASHSEED = 0 python3 main.py --agent dqn --lr 0.0005 --gamma 0.99 --batch_size 512 --mem_size 25000 --layers 17 17 17 --max_steps 100000 --replace 50 --save_interval 10 --render_freq 0

PYTHONHASHSEED=0 python3 main.py --agent dqn --lr 0.00025 --epsilon_decay_steps 250000 --gamma 0.99 --batch_size 512 --mem_size 500000 --layers 17 17 17 --max_steps 50000000 --replace_target_steps 2500 --exploration_steps 12500 --save_interval 100 --render_freq 0