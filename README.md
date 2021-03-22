# Natural Language Processing - Character Completion

For this project, you will develop a program that takes in a string of character and tries to predict the next character.

## Input format

`example/input.txt` contains an example of what the input to your program will look like.
Each line in this file correspond to a string, for which you must guess what the next character should be.

## Output format

`example/pred.txt` contains an example of what the output of your program must look like.
Each line in this file correspond to guesses by the program of what the next character should be.
In other words, line `i` in `example/pred.txt` corresponds to what character the program thinks should come after the string in line `i` of `example/pred.txt`.
In this case, for each string, the program produces 3 guesses.


## Implementing your program

`src/myprogram.py` contains the example program, along with a simple commandline interface that allows for training and testing.

Let's walk through how to use the example program. First we will train the model, telling the program to save intermediate results in the directory `work`:

```
python src/myprogram.py train --work_dir work
```

Because this model doesn't actually require training, we simply saved a fake checkpoint to `work/model.checkpoint`.
Next, we will generate predictions for the example data in `example/input.txt` and save it in `pred.txt`:

```
python src/myprogram.py test --work_dir work --test_data example/input.txt --test_output pred.txt
```
