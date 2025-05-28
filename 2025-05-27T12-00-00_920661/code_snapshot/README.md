# bachelor-thesis

Windows: $env:PYTHONHASHSEED = "0"; python main.py --agent dqn --lr 0.0005 --gamma 0.99 --batch_size 512 --mem_size 25000 --layers 17 17 17 --max_steps 100000 --replace 50 --save_interval 10 --render_freq 0

Macbook: PYTHONHASHSEED = "0" python3 main.py --agent dqn --lr 0.0005 --gamma 0.99 --batch_size 512 --mem_size 25000 --layers 17 17 17 --max_steps 100000 --replace 50 --save_interval 10 --render_freq 0