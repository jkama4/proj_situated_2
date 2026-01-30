# Project Situated AI - RL + Robotics

[Exercise 1](https://github.com/jkama4/proj_situated_2/blob/main/src/robotics_rl/exercise_1_nb.ipynb)
[Exercise 2](https://github.com/jkama4/proj_situated_2/blob/main/src/robotics_rl/exercise_2_nb.ipynb)
[Exercise 3 SAC](https://github.com/jkama4/proj_situated_2/blob/main/src/robotics_rl/exercise_3_sac_train.py)
[Exercise 3 TQC](https://github.com/jkama4/proj_situated_2/blob/main/src/robotics_rl/exercise_3_tqc_train.py)

# Setting up the Environment

## No Virtual env (pip)
If you don't Poetry to manage dependencies, you can simply use the following command

```bash
pip install -r requirements.txt
```

## Using Virtual env (recommended)
First, ensure Poetry is installed

```bash
pip install poetry
```

Then, you should go to the directory where you set up the project

```bash
cd ~/path/to/project
```

Now, you can setup the environment

```bash
poetry install
```

And to use it, you call

```bash
$(poetry env activate)
```

Or on Windows

```bash
poetry env activate
```

Alternatively, you can use

```bash
poetry shell
```

