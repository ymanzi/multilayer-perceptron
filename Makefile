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
	rm -f *.pickle data.csv data_*.csv saved_NN/* -y

describe:
	python3.8 describe.py srcs/data.csv

show:
	python3.8 scatter_plot.py srcs/data.csv

predict:
	python3 predict.py "saved_NN/1-mini-batch|Tanh.pickle" data_test.csv
	python3 predict.py "saved_NN/2-5Layers|Tanh.pickle" data_test.csv
	python3 predict.py "saved_NN/3-Stochastic|Sigmoid.pickle" data_test.csv
	python3 predict.py "saved_NN/4-mini-batch|Sigmoid.pickle" data_test.csv
	python3 predict.py "saved_NN/5-mini-batch|ReLU|he.pickle" data_test.csv

train:
	make corr
	python3.8 train.py data_training.csv