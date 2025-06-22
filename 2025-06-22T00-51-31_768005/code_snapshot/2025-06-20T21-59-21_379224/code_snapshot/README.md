# bachelor-thesis

$env:PYTHONHASHSEED = "0"; python main.py --agent dqn --lr 0.0001 --gamma 0.99 --batch_size 512 --mem_size 1000000 --layers 22 22 22 --max_steps 20000000 --replace_target_steps 2500 --epsilon_decay_steps 250000 --save_interval 100 --render_freq 0

Macbook: PYTHONHASHSEED=0 python main.py --agent dqn --lr 0.0001 --gamma 0.99 --batch_size 512 --mem_size 1000000 --layers 22 22 22 --max_steps 20000000 --replace_target_steps 2500 --epsilon_decay_steps 250000 --save_interval 100 --render_freq 0
