# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    Makefile                                           :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ymanzi <marvin@42.fr>                      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2019/10/09 14:05:13 by ymanzi            #+#    #+#              #
#    Updated: 2020/11/22 13:49:15 by ymanzi           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

# git clone https://github.com/Homebrew/brew ~/.linuxbrew/Homebrew
# mkdir ~/.linuxbrew/bin
# ln -s ~/.linuxbrew/Homebrew/bin/brew ~/.linuxbrew/bin
# eval $(~/.linuxbrew/bin/brew shellenv)

env:
	brew install python@3.8
	python -m pip install -U pip
	pip install numpy
	pip install pandas
	pip install matplotlib

corr:
	python3 correction/evaluation.py correction/data.csv

clean:
	rm -f *CE.pickle data.csv data_*.csv

describe:
	python3.8 describe.py srcs/data.csv

show:
	python3.8 scatter_plot.py srcs/data.csv

predict:
	python3 predict.py 1-mini-batch\|Tanh\|Xavier\|CE.pickle data_test.csv
	python3 predict.py 2-Stochastic\|Sigmoid\|xavier\|CE.pickle data_test.csv
	python3 predict.py 3-mini-batch\|Sigmoid\|xavier\|CE.pickle data_test.csv
	python3 predict.py 4-mini-batch\|ReLU\|he\|CE.pickle data_test.csv

train:
	python3.8 train.py data_training.csv